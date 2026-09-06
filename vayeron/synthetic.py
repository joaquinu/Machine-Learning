"""Physically-modelled run-to-failure surrogate.

This is a stand-in for the Mendeley/KAIST accelerated life test, used to
exercise and verify the scoring pipeline end to end when the real archive is
not reachable. It is NOT a substitute for the real validation: the lead-time
numbers it produces describe this model's degradation law, not a physical
bearing.

Model: a constant-speed rig with shaft harmonics and broadband noise, into
which a localised outer-race spall is introduced part-way through life. The
spall produces a decaying-resonance impulse train at the outer-race defect
frequency whose amplitude grows exponentially, with the usual late-stage
smearing (jitter and rising broadband noise as the raceway breaks up). Bearing
temperature follows friction, rising to the 85 C trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import VAYERON_DEFECT_ORDERS


@dataclass
class SurrogateConfig:
    fs: float = 25600.0
    rpm: float = 1770.0
    snapshot_seconds: float = 1.0
    acquisition_interval_s: float = 600.0   # one snapshot every 10 min
    n_acquisitions: int = 240               # -> 40 h of test time
    defect_order: float = VAYERON_DEFECT_ORDERS["bpfo"]
    onset_fraction: float = 0.55            # spall initiates here, in fraction of life
    baseline_noise: float = 0.35            # m/s^2 rms broadband, healthy
    shaft_amplitude: float = 0.45
    resonance_hz: float = 3500.0
    resonance_decay: float = 1400.0
    final_impulse_amplitude: float = 9.0
    late_smearing: float = 0.35             # jitter fraction at end of life
    temp_ambient_c: float = 32.0
    temp_final_c: float = 88.0
    temp_asymmetry_c: float = 6.0           # driven end runs hotter once damaged
    # Healthy-phase realism. A real rig is not stationary to four decimal
    # places: belt/coupling load wanders, the housing warms, and the odd
    # acquisition catches a transient. Without these the healthy MAD collapses
    # and every sigma limit becomes hypersensitive.
    load_wander_rel: float = 0.06           # slow random walk on excitation level
    load_wander_tau_acq: float = 25.0       # correlation length, in acquisitions
    ambient_drift_c: float = 3.0
    transient_rate: float = 0.02            # fraction of acquisitions with a knock
    transient_gain: float = 2.5
    seed: int = 20260906


def _impulse_response(fs: float, resonance_hz: float, decay: float) -> np.ndarray:
    t = np.arange(0, 5.0 / decay, 1.0 / fs)
    return np.exp(-decay * t) * np.sin(2.0 * np.pi * resonance_hz * t)


def _severity_curve(frac: float, onset: float) -> float:
    """Damage amplitude in [0, 1] as a function of life fraction."""
    if frac <= onset:
        return 0.0
    x = (frac - onset) / (1.0 - onset)
    # Exponential propagation, normalised to 1.0 at end of life.
    return float((np.exp(3.0 * x) - 1.0) / (np.exp(3.0) - 1.0))


def _ou_walk(n: int, tau: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Mean-reverting walk, used for slow load/tension drift across the test."""
    a = np.exp(-1.0 / max(tau, 1e-6))
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = a * out[i - 1] + np.sqrt(1 - a * a) * rng.normal()
    return sigma * out


def generate_snapshot(frac: float, cfg: SurrogateConfig, rng: np.random.Generator,
                      load: float = 1.0, transient: float = 1.0) -> np.ndarray:
    n = int(cfg.snapshot_seconds * cfg.fs)
    t = np.arange(n) / cfg.fs
    shaft_hz = cfg.rpm / 60.0
    damage = _severity_curve(frac, cfg.onset_fraction)

    sig = load * cfg.shaft_amplitude * np.sin(2.0 * np.pi * shaft_hz * t + rng.uniform(0, 2 * np.pi))
    sig += 0.25 * load * cfg.shaft_amplitude * np.sin(2.0 * np.pi * 2 * shaft_hz * t + rng.uniform(0, 2 * np.pi))
    sig += rng.normal(0.0, load * transient * cfg.baseline_noise * (1.0 + 1.5 * damage), n)

    if damage > 0:
        fd = shaft_hz * cfg.defect_order
        period = cfg.fs / fd
        jitter = cfg.late_smearing * damage
        train = np.zeros(n)
        k = 0
        while True:
            centre = k * period * (1.0 + rng.normal(0.0, jitter * 0.05))
            idx = int(round(centre))
            if idx >= n:
                break
            # Load-zone modulation: outer-race defects on a rotating-inner-race
            # bearing sit in a fixed load zone, so amplitude is near-constant.
            train[idx] = 1.0 + rng.normal(0.0, 0.08)
            k += 1
        resp = _impulse_response(cfg.fs, cfg.resonance_hz, cfg.resonance_decay)
        sig += load * cfg.final_impulse_amplitude * damage * np.convolve(train, resp)[:n]

    return sig


def generate_run(cfg: SurrogateConfig | None = None,
                 start: str = "2026-05-01 00:00:00") -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Return per-acquisition metadata and the raw waveform snapshots."""
    cfg = cfg or SurrogateConfig()
    rng = np.random.default_rng(cfg.seed)
    t0 = pd.Timestamp(start)

    load = 1.0 + _ou_walk(cfg.n_acquisitions, cfg.load_wander_tau_acq, cfg.load_wander_rel, rng)
    ambient = cfg.temp_ambient_c + _ou_walk(
        cfg.n_acquisitions, cfg.load_wander_tau_acq, cfg.ambient_drift_c, rng)
    knock = np.where(rng.random(cfg.n_acquisitions) < cfg.transient_rate, cfg.transient_gain, 1.0)

    meta, waves = [], []
    for i in range(cfg.n_acquisitions):
        frac = i / max(cfg.n_acquisitions - 1, 1)
        wave = generate_snapshot(frac, cfg, rng, load=float(load[i]), transient=float(knock[i]))
        damage = _severity_curve(frac, cfg.onset_fraction)
        base_temp = ambient[i] + (cfg.temp_final_c - cfg.temp_ambient_c) * damage**1.4
        meta.append({
            "timestamp": t0 + pd.Timedelta(seconds=i * cfg.acquisition_interval_s),
            "acquisition": i,
            "rpm": cfg.rpm + rng.normal(0.0, 2.0),
            "temp1": base_temp + cfg.temp_asymmetry_c * damage + rng.normal(0, 0.25),
            "temp2": base_temp + rng.normal(0, 0.25),
            "life_fraction": frac,
            "true_damage": damage,
            "load_factor": float(load[i]),
            "is_transient": bool(knock[i] > 1.0),
        })
        waves.append(wave)
    return pd.DataFrame(meta), waves


def write_run(out_dir: str | Path, cfg: SurrogateConfig | None = None) -> Path:
    """Write the surrogate as per-acquisition CSV files, the way a real archive ships."""
    cfg = cfg or SurrogateConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta, waves = generate_run(cfg)
    for row, wave in zip(meta.itertuples(), waves):
        ts = pd.Timestamp(row.timestamp).strftime("%Y%m%d_%H%M%S")
        frame = pd.DataFrame({
            "vibration": wave,
            "temp1": np.full(wave.size, row.temp1),
            "temp2": np.full(wave.size, row.temp2),
        })
        frame.to_csv(out / f"acq_{row.acquisition:05d}_{ts}.csv", index=False)
    meta.to_csv(out / "_ground_truth.csv", index=False)
    return out
