"""Vayeron-equivalent edge features from a raw vibration waveform.

The Smart-Idler firmware reports, per analysis window, an overall RMS plus four
Goertzel envelope ratios:

    f_defect = (RPM / 60) * (Factor / 10)
    ratio    = peak amplitude at f_defect / noise-floor amplitude in adjacent bins

External datasets ship raw accelerometer waveforms instead, so those two
quantities have to be reconstructed before Vayeron's control limits mean
anything. This module does that with standard envelope (Hilbert) demodulation,
which is the textbook equivalent of the firmware's Goertzel power filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

# KX-VAY-012 section 5.1: Vayeron's defect multipliers, expressed as orders of
# shaft rate (Factor / 10).
VAYERON_DEFECT_ORDERS: dict[str, float] = {
    "bsf": 2.1,   # ball spin
    "bpfo": 3.1,  # outer race
    "ftf": 4.1,   # cage / "Factor-40"
    "bpfi": 4.9,  # inner race
}


def geometric_defect_orders(
    n_balls: int, ball_diameter: float, pitch_diameter: float, contact_angle_deg: float = 0.0
) -> dict[str, float]:
    """True BPFO/BPFI/BSF/FTF orders (multiples of shaft rate) from bearing geometry.

    Prefer this over :data:`VAYERON_DEFECT_ORDERS` whenever the bearing under
    test is known: Vayeron's factors describe the Smart-Idler's own bearing, not
    a lab test bearing.
    """
    ratio = (ball_diameter / pitch_diameter) * np.cos(np.deg2rad(contact_angle_deg))
    return {
        "ftf": 0.5 * (1.0 - ratio),
        "bpfo": 0.5 * n_balls * (1.0 - ratio),
        "bpfi": 0.5 * n_balls * (1.0 + ratio),
        "bsf": 0.5 * (pitch_diameter / ball_diameter) * (1.0 - ratio**2),
    }


@dataclass
class FeatureConfig:
    """How a raw waveform snapshot is reduced to Vayeron-equivalent channels."""

    fs: float                                   # sample rate, Hz
    rpm: float                                  # shaft speed during the snapshot
    defect_orders: dict[str, float] = field(
        default_factory=lambda: dict(VAYERON_DEFECT_ORDERS)
    )
    # Envelope demodulation band. Bearing impacts excite structural resonances
    # well above shaft harmonics; the default keeps the top usable octaves.
    band_hz: tuple[float, float] | None = None
    band_order: int = 4
    detrend: bool = True
    # Peak search half-width around the defect line, as a fraction of it, plus a
    # few FFT bins to absorb speed jitter and leakage.
    peak_tolerance: float = 0.02
    peak_extra_bins: int = 2
    # Noise floor is the median magnitude in +/- noise_span around the line,
    # excluding the peak zone and the zones of the other tracked lines.
    noise_span: float = 0.30
    n_harmonics: int = 1

    def resolved_band(self) -> tuple[float, float]:
        if self.band_hz is not None:
            return self.band_hz
        nyq = self.fs / 2.0
        return (min(1000.0, 0.05 * nyq), 0.80 * nyq)

    def defect_frequencies(self) -> dict[str, float]:
        shaft_hz = self.rpm / 60.0
        return {k: shaft_hz * v for k, v in self.defect_orders.items()}


def bandpass(x: np.ndarray, fs: float, band: tuple[float, float], order: int = 4) -> np.ndarray:
    lo, hi = band
    nyq = fs / 2.0
    lo = max(lo, 1e-6)
    hi = min(hi, nyq * 0.999)
    if lo >= hi:
        return x
    sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def envelope_spectrum(x: np.ndarray, fs: float, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectrum of the Hilbert envelope of the band-passed signal."""
    sig = np.asarray(x, dtype=float)
    if cfg.detrend:
        sig = sig - sig.mean()
    sig = bandpass(sig, fs, cfg.resolved_band(), cfg.band_order)
    env = np.abs(hilbert(sig))
    env = env - env.mean()
    n = env.size
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(env * win)) * (2.0 / np.sum(win))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spec


def _band_peak(freqs: np.ndarray, spec: np.ndarray, f0: float, half_width: float) -> float:
    sel = (freqs >= f0 - half_width) & (freqs <= f0 + half_width)
    return float(spec[sel].max()) if sel.any() else 0.0


def envelope_ratios(
    x: np.ndarray, cfg: FeatureConfig, freqs: np.ndarray | None = None,
    spec: np.ndarray | None = None
) -> dict[str, float]:
    """Peak-to-noise-floor ratio at each tracked defect line."""
    if freqs is None or spec is None:
        freqs, spec = envelope_spectrum(x, cfg.fs, cfg)
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    lines = cfg.defect_frequencies()

    ratios: dict[str, float] = {}
    for name, f0 in lines.items():
        if f0 <= 0 or f0 >= freqs[-1]:
            ratios[name] = float("nan")
            continue
        half = cfg.peak_tolerance * f0 + cfg.peak_extra_bins * df

        peak = 0.0
        used = 0
        for h in range(1, cfg.n_harmonics + 1):
            fh = f0 * h
            if fh + half >= freqs[-1]:
                break
            peak += _band_peak(freqs, spec, fh, half)
            used += 1
        if used == 0:
            ratios[name] = float("nan")
            continue
        peak /= used

        # Noise floor: median magnitude around the line, with every tracked
        # line (and its harmonics) excluded so a neighbouring defect does not
        # inflate the reference.
        lo, hi = f0 * (1.0 - cfg.noise_span), f0 * (1.0 + cfg.noise_span)
        sel = (freqs >= lo) & (freqs <= hi)
        for other in lines.values():
            for h in range(1, cfg.n_harmonics + 2):
                fh = other * h
                w = cfg.peak_tolerance * fh + cfg.peak_extra_bins * df
                sel &= ~((freqs >= fh - w) & (freqs <= fh + w))
        floor_vals = spec[sel]
        if floor_vals.size < 3:
            sel = (freqs >= lo) & (freqs <= hi)
            floor_vals = spec[sel]
        floor = float(np.median(floor_vals)) if floor_vals.size else 0.0
        ratios[name] = peak / floor if floor > 0 else float("nan")

    return ratios


def waveform_features(x: np.ndarray, cfg: FeatureConfig) -> dict[str, float]:
    """Overall RMS plus the four envelope ratios for one acquisition snapshot."""
    sig = np.asarray(x, dtype=float)
    sig = sig[np.isfinite(sig)]
    if sig.size == 0:
        return {"rms": float("nan"), **{k: float("nan") for k in cfg.defect_orders}}
    ac = sig - sig.mean()
    feats: dict[str, float] = {
        "rms": float(np.sqrt(np.mean(ac**2))),
        "peak": float(np.max(np.abs(ac))),
        "kurtosis": float(np.mean(ac**4) / (np.mean(ac**2) ** 2)) if np.mean(ac**2) > 0 else float("nan"),
    }
    feats["crest_factor"] = feats["peak"] / feats["rms"] if feats["rms"] > 0 else float("nan")
    feats.update(envelope_ratios(sig, cfg))
    return feats
