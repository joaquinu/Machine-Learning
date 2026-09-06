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

# How each physical failure mode shows up in the envelope spectrum.
#   order      - which Vayeron defect line carries the impulse train
#   modulation - amplitude modulation of the train, in orders of shaft rate:
#                a defect fixed in the load zone (outer race) is unmodulated; an
#                inner-race defect passes through the load zone once per shaft
#                revolution; a rolling-element defect is modulated at cage rate.
#   channel    - the channel the scorer should attribute the fault to
#   asymmetry_c - end-to-end temperature divergence at full damage, in C. A
#                 localised spall raises both raceways much alike; it is
#                 lubrication and seal failures that run one end hot. Coupling
#                 asymmetry to damage in every mode would let a thermal channel
#                 "detect" faults it has no physical claim on.
FAULT_MODES: dict[str, dict] = {
    "outer_race": {"order": VAYERON_DEFECT_ORDERS["bpfo"], "modulation": 0.0,
                   "channel": "bpfo", "asymmetry_c": 0.8},
    "inner_race": {"order": VAYERON_DEFECT_ORDERS["bpfi"], "modulation": 1.0,
                   "channel": "bpfi", "asymmetry_c": 0.8},
    "ball_spin": {"order": VAYERON_DEFECT_ORDERS["bsf"], "modulation": 0.4,
                  "channel": "bsf", "asymmetry_c": 0.8},
    "cage": {"order": VAYERON_DEFECT_ORDERS["ftf"], "modulation": 0.0,
             "channel": "ftf", "asymmetry_c": 1.2},
    # Grease dry-out / seal failure: friction and heat, no localised spall, so
    # no defect line at all. The spectral limits have nothing to bite on, and
    # this is the mode that genuinely runs one end hot.
    "thermal": {"order": None, "modulation": 0.0, "channel": "rms",
                "asymmetry_c": 7.0},
}


@dataclass
class SurrogateConfig:
    fs: float = 25600.0
    rpm: float = 1770.0
    snapshot_seconds: float = 1.0
    acquisition_interval_s: float = 600.0   # one snapshot every 10 min
    n_acquisitions: int = 240               # -> 40 h of test time
    fault_mode: str = "outer_race"
    defect_order: float | None = None   # None -> taken from fault_mode
    onset_fraction: float = 0.55            # spall initiates here, in fraction of life
    baseline_noise: float = 0.35            # m/s^2 rms broadband, healthy
    shaft_amplitude: float = 0.45
    resonance_hz: float = 3500.0
    resonance_decay: float = 1400.0
    final_impulse_amplitude: float = 9.0
    late_smearing: float = 0.35             # jitter fraction at end of life
    temp_ambient_c: float = 32.0
    temp_final_c: float = 88.0
    temp_asymmetry_c: float | None = None   # None -> taken from fault_mode
    # Healthy-phase realism. A real rig is not stationary to four decimal
    # places: belt/coupling load wanders, the housing warms, and the odd
    # acquisition catches a transient. Without these the healthy MAD collapses
    # and every sigma limit becomes hypersensitive.
    load_wander_rel: float = 0.06           # slow random walk on excitation level
    load_wander_tau_acq: float = 25.0       # correlation length, in acquisitions
    ambient_drift_c: float = 3.0
    transient_rate: float = 0.02            # fraction of acquisitions with a knock
    transient_gain: float = 2.5
    # Grease dry-out raises friction noise without producing a defect line.
    thermal_noise_growth: float = 4.0
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
    mode = FAULT_MODES[cfg.fault_mode]

    # Grease dry-out drives friction noise up without any localised impact.
    noise_growth = cfg.thermal_noise_growth if mode["order"] is None else 1.5
    sig = load * cfg.shaft_amplitude * np.sin(2.0 * np.pi * shaft_hz * t + rng.uniform(0, 2 * np.pi))
    sig += 0.25 * load * cfg.shaft_amplitude * np.sin(2.0 * np.pi * 2 * shaft_hz * t + rng.uniform(0, 2 * np.pi))
    sig += rng.normal(0.0, load * transient * cfg.baseline_noise * (1.0 + noise_growth * damage), n)

    order = cfg.defect_order if cfg.defect_order is not None else mode["order"]
    if damage > 0 and order:
        fd = shaft_hz * order
        period = cfg.fs / fd
        jitter = cfg.late_smearing * damage
        train = np.zeros(n)
        k = 0
        while True:
            centre = k * period * (1.0 + rng.normal(0.0, jitter * 0.05))
            idx = int(round(centre))
            if idx >= n:
                break
            amp = 1.0 + rng.normal(0.0, 0.08)
            if mode["modulation"]:
                # Defect passing in and out of the load zone.
                phase = 2.0 * np.pi * mode["modulation"] * shaft_hz * (idx / cfg.fs)
                amp *= 0.15 + 0.85 * (0.5 + 0.5 * np.cos(phase))
            train[idx] = amp
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
    asymmetry_c = (cfg.temp_asymmetry_c if cfg.temp_asymmetry_c is not None
                   else FAULT_MODES[cfg.fault_mode]["asymmetry_c"])

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
            "temp1": base_temp + asymmetry_c * damage + rng.normal(0, 0.25),
            "temp2": base_temp + rng.normal(0, 0.25),
            "life_fraction": frac,
            "true_damage": damage,
            "load_factor": float(load[i]),
            "is_transient": bool(knock[i] > 1.0),
            "fault_mode": cfg.fault_mode,
            "expected_channel": FAULT_MODES[cfg.fault_mode]["channel"],
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


# --- field-shaped surrogate ---------------------------------------------------

FIELD_ROLLERS = {
    # name: (operating_angle_deg, baseline rms, ftf weighting)
    "C_CNT": (0.0, 55.6, 1.35),
    "R0": (0.0, 54.8, 1.30),
    "C_LHS": (35.0, 48.8, 1.00),
    "C_RHS": (35.0, 49.3, 1.00),
}
RPM_REGIMES = ((275.0, 0.04), (545.0, 0.18), (577.0, 0.22), (606.0, 0.56))


@dataclass
class FieldSurrogateConfig:
    """A record shaped like the Vayeron field export, with known label noise.

    Used to exercise the label audit against a dataset whose knock count is
    known by construction. It is not a model of the mine - it reproduces the
    cadence, the roller geometry split, the rpm regimes, the transient rate and
    one sustained multi-channel event, which is what the audit reads.
    """

    rows_per_roller: int = 20_400          # ~81.6k total, matching KX-VAY-012
    cadence_seconds: float = 626.0
    knock_rate: float = 0.004              # isolated single-sample spikes
    knock_gain: float = 6.0                # in baseline MADs
    transient_rate: float = 0.028          # rms == 0 speed-change artefacts
    event_roller: str = "C_RHS"
    event_start_fraction: float = 0.47
    event_days: float = 11.0
    event_rms_gain: float = 1.35           # +35%, per KX-VAY-012 section 6.2
    event_bpfi_gain: float = 1.40          # +40%
    regime_drift: bool = True
    seed: int = 20260903


def generate_field_record(cfg: FieldSurrogateConfig | None = None,
                          start: str = "2026-03-04") -> pd.DataFrame:
    cfg = cfg or FieldSurrogateConfig()
    rng = np.random.default_rng(cfg.seed)
    t0 = pd.Timestamp(start)
    frames = []

    for roller, (angle, rms_base, ftf_w) in FIELD_ROLLERS.items():
        n = cfg.rows_per_roller
        ts = t0 + pd.to_timedelta(np.arange(n) * cfg.cadence_seconds, unit="s")
        frac = np.arange(n) / n

        speeds, weights = zip(*RPM_REGIMES)
        if cfg.regime_drift:
            # The operational mix drifts across the deployment (section 3.1).
            late = np.array([0.02, 0.10, 0.14, 0.74])
            mix = np.outer(1 - frac, weights) + np.outer(frac, late)
            mix /= mix.sum(axis=1, keepdims=True)
            pick = np.array([rng.choice(len(speeds), p=m) for m in mix])
        else:
            pick = rng.choice(len(speeds), size=n, p=weights)
        rpm = np.array(speeds)[pick] + rng.normal(0, 3.0, n)

        # Vibration scales with speed; the health signal is the normalised part.
        rms = rms_base * (rpm / 600.0) * (1 + rng.normal(0, 0.035, n))
        ratios = {
            "bpfo": 1.85 + rng.normal(0, 0.30, n),
            "bpfi": 1.90 + rng.normal(0, 0.28, n),
            "bsf": 1.70 + rng.normal(0, 0.26, n),
            "ftf": 1.75 * ftf_w + rng.normal(0, 0.30, n),
        }

        # One sustained, coherent multi-channel event on a single roller.
        is_event = np.zeros(n, dtype=bool)
        if roller == cfg.event_roller:
            i0 = int(cfg.event_start_fraction * n)
            i1 = min(i0 + int(cfg.event_days * 86400 / cfg.cadence_seconds), n)
            ramp = np.zeros(n)
            ramp[i0:i1] = np.sin(np.linspace(0, np.pi, i1 - i0)) ** 0.6
            rms *= 1 + (cfg.event_rms_gain - 1) * ramp
            ratios["bpfi"] *= 1 + (cfg.event_bpfi_gain - 1) * ramp
            is_event = ramp > 0.05

        # Isolated single-sample knocks - the label-noise source under audit.
        knock = rng.random(n) < cfg.knock_rate
        knock &= ~is_event
        rms = np.where(knock, rms * (1 + cfg.knock_gain * 0.035), rms)
        for k in ratios:
            spike = knock & (rng.random(n) < 0.5)
            ratios[k] = np.where(spike, ratios[k] + cfg.knock_gain * 0.28, ratios[k])

        # Speed-change transients: firmware emits zeros (section 4.2).
        transient = rng.random(n) < cfg.transient_rate
        rms = np.where(transient, 0.0, rms)
        for k in ratios:
            ratios[k] = np.where(transient, 0.0, ratios[k])

        base_temp = 43.6 - 9.2 * np.sin(np.pi * frac) + rng.normal(0, 0.4, n)
        frames.append(pd.DataFrame({
            "timestamp": ts,
            "roller_id": roller,
            "operating_angle_deg": angle,
            "rpm": rpm,
            "rms": rms,
            **ratios,
            "temp1": base_temp + rng.normal(0, 0.2, n),
            "temp2": base_temp + rng.normal(0, 0.2, n),
            "injected_knock": knock,
            "injected_transient": transient,
            "injected_event": is_event,
        }))

    return pd.concat(frames, ignore_index=True)
