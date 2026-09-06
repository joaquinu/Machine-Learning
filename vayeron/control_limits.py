"""Vayeron statistical control-limit scoring (KX-VAY-012 section 7).

Implements the calibrated bearing-health index Y in [0, 1]:

    step 1  z_i,c   = (x_i,c - median_c) / (MAD_c * 1.4826)      robust deviation
    step 2  s_i,c   = z_i,c / z_alert,c                          severity vs limit
            S_i     = max_c s_i,c
    step 3  S_bar_i = EMA(S_i, tau ~ 2 h)                        persistence
    step 4  Y_i     = 1 / (1 + exp(-3.0 * (S_bar_i - 1.0)))      logistic map
    step 5  tiers   normal <0.35, watch <0.50, alert <0.80, critical

Control limits are the Vayeron industrial process-control values: 3.5 sigma on
overall vibration, 4.0 sigma on the Goertzel spectral defect ratios.

The module is dataset-agnostic: it scores any frame that carries a timestamp and
some subset of the channels {rms, bpfo, bpfi, bsf, ftf}. It is used both for
Vayeron field telemetry and for external run-to-failure datasets, so the healthy
baseline can be selected either as a leading fraction of the record (the usual
choice for run-to-failure data, where the machine starts healthy) or by an
explicit mask.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# KX-VAY-012 section 5.2 / 7: firmware alert constants replaced by robust
# statistical control limits.
VAYERON_ALERT_LIMITS: dict[str, float] = {
    "rms": 3.5,   # overall vibration
    "bpfo": 4.0,  # outer race
    "bpfi": 4.0,  # inner race
    "bsf": 4.0,   # ball spin
    "ftf": 4.0,   # cage / Factor-40
}

SPECTRAL_CHANNELS: tuple[str, ...] = ("bpfo", "bpfi", "bsf", "ftf")

# Y as delivered has no thermal term at all, although the Smart-Idler datasheet
# specifies a dedicated Temp Alert byte and KX-VAY-012 section 6.2 calls a
# >4 C end-to-end divergence an anomaly in its own right. These two derived
# channels close that gap when enabled.
#   temp_max   - the hotter raceway. Carries real information but is confounded
#                by ambient: section 6.1 item 3 rules out seasonal drift as an
#                anomaly, and both ends cool together, so this channel alone
#                will chase the seasons.
#   temp_delta - |T1 - T2|. Ambient-invariant by construction, which is exactly
#                why the datasheet monitors both ends separately.
THERMAL_CHANNELS: tuple[str, ...] = ("temp_max", "temp_asymmetry")
THERMAL_ALERT_LIMITS: dict[str, float] = {"temp_max": 3.5, "temp_asymmetry": 3.5}

# KX-VAY-012 section 7 step 5.
TIER_EDGES: tuple[float, float, float] = (0.35, 0.50, 0.80)
TIER_NAMES: tuple[str, str, str, str] = ("normal", "watch", "alert", "critical")

_MAD_TO_SIGMA = 1.4826


@dataclass
class ScoringConfig:
    """Every knob of the KX-VAY-012 target construction, in one place."""

    alert_limits: Mapping[str, float] = field(
        default_factory=lambda: dict(VAYERON_ALERT_LIMITS)
    )

    # --- healthy baseline -------------------------------------------------
    # Run-to-failure records start healthy, so the baseline is taken from the
    # leading slice. Expressed in elapsed time (not row count) so that an
    # irregular acquisition cadence does not distort the window.
    healthy_window_ratio: float = 0.10
    healthy_window_hours: float | None = None  # overrides the ratio when set
    healthy_mask_column: str | None = None     # overrides both when set
    min_healthy_samples: int = 30

    # --- quality gates (KX-VAY-012 section 9) -----------------------------
    # 350-1500 rpm is the Vayeron conveyor envelope. Lab rigs run outside it;
    # override rather than silently gating an entire dataset away.
    rpm_gate: tuple[float, float] | None = (350.0, 1500.0)
    drop_zero_rms: bool = True  # supposition 6: rms == 0 is a speed transient

    # --- normalisation ----------------------------------------------------
    # Specific vibration intensity, RMS_norm = RMS / (RPM / 600). A no-op at
    # constant speed (the robust z-score is scale invariant), but kept so the
    # same code path serves variable-speed field data.
    normalize_rms_by_rpm: bool = True
    rpm_reference: float = 600.0

    # --- persistence ------------------------------------------------------
    # A single-sample spike survives the EMA: the smoother spreads it over
    # ~span samples, so it clears a "3 consecutive samples" persistence rule and
    # registers as a real alert episode. A short median prefilter on the raw
    # severity removes it before smoothing. 0 = off, i.e. KX-VAY-012 as written.
    prefilter_median_samples: int = 0
    ema_tau_hours: float = 2.0
    # "span": pandas span EMA with span = tau * samples_per_hour, matching the
    #         delivered Vayeron implementation on a regular cadence.
    # "time": alpha_i = 1 - exp(-dt_i / tau), correct under irregular sampling.
    ema_mode: str = "span"
    irregular_cadence_cv: float = 0.25  # warn above this coefficient of variation

    # --- logistic mapping -------------------------------------------------
    logistic_gain: float = 3.0
    logistic_center: float = 1.0
    clip_negative_severity: bool = False
    # KX-VAY-012 section 7 aggregates severity as max over channels. Section 6.2
    # describes a real anomaly as a *coherent multi-channel* excursion, so
    # "kth_max" (k=2 -> the second-highest channel drives the score) is offered
    # as a drift-resistant alternative. Default stays at the delivered spec.
    aggregation: str = "max"
    aggregation_k: int = 2
    # How fault_channel picks a driver. For an amplitude-modulated defect
    # (inner race, ball spin) the correct channel leads the others by only
    # ~1.5x, so the choice of rule decides the answer.
    #   "max_severity"      - KX-VAY-012 as written: argmax over all channels.
    #   "spectral_priority" - if any spectral channel is above its own control
    #                         limit, attribute to the highest of those;
    #                         otherwise fall back to rms. Reflects how an
    #                         analyst reads it: RMS says something is wrong, a
    #                         defect line says what. Counteracts the structural
    #                         bias toward rms, whose baseline dispersion is
    #                         4-7x tighter than the spectral channels'.
    attribution: str = "max_severity"
    # Which thermal channels to fold into S. () reproduces KX-VAY-012, which
    # has none. ("temp_asymmetry",) is the ambient-robust choice.
    thermal_channels: tuple[str, ...] = ()
    # Section 6.2 gives an absolute engineering threshold for asymmetry. When
    # set, a divergence this large counts as being at the control limit however
    # tight the statistical baseline is.
    temp_asymmetry_floor_c: float | None = 4.0

    # --- numerical guards -------------------------------------------------
    mad_floor_relative: float = 1e-6  # numerical guard: floor MAD at this fraction of |median|
    mad_floor_absolute: float = 1e-12
    # Engineering guard, off by default. A lab rig's healthy phase can be so
    # stationary that MAD collapses to a fraction of a percent of the median,
    # which makes a 3.5 sigma limit trip on a ~1% change. Setting this to e.g.
    # 0.01 declares "changes below 1% of baseline are not resolvable" and keeps
    # the limits comparable with field deployments.
    min_mad_fraction_of_median: float = 0.0

    # --- crossing detection ------------------------------------------------
    persistence_samples: int = 3  # consecutive samples required for a crossing

    def limits_for(self, columns: Iterable[str]) -> dict[str, float]:
        merged = {**THERMAL_ALERT_LIMITS, **self.alert_limits}
        return {c: float(merged[c]) for c in columns if c in merged}


@dataclass
class ChannelBaseline:
    channel: str
    source_column: str
    median: float
    mad: float
    mad_floored: bool
    n_samples: int

    def as_dict(self) -> dict:
        return {
            "channel": self.channel,
            "source_column": self.source_column,
            "median": self.median,
            "mad": self.mad,
            "sigma_equivalent": self.mad * _MAD_TO_SIGMA,
            "mad_floored": self.mad_floored,
            "n_samples": self.n_samples,
        }


@dataclass
class ScoringReport:
    baselines: dict[str, ChannelBaseline]
    healthy_rows: int
    healthy_span_hours: float
    total_rows: int
    gated_rows: int
    median_cadence_seconds: float
    cadence_cv: float
    ema_mode: str
    ema_span_samples: float | None
    channels_scored: list[str]

    def as_dict(self) -> dict:
        return {
            "baselines": {k: v.as_dict() for k, v in self.baselines.items()},
            "healthy_rows": self.healthy_rows,
            "healthy_span_hours": self.healthy_span_hours,
            "total_rows": self.total_rows,
            "gated_rows": self.gated_rows,
            "median_cadence_seconds": self.median_cadence_seconds,
            "cadence_cv": self.cadence_cv,
            "ema_mode": self.ema_mode,
            "ema_span_samples": self.ema_span_samples,
            "channels_scored": self.channels_scored,
        }


def _elapsed_hours(ts: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(ts):
        # Already an elapsed-time column; assume hours unless it looks like s.
        return ts.to_numpy(dtype=float)
    t = pd.to_datetime(ts, utc=False)
    return (t - t.iloc[0]).dt.total_seconds().to_numpy() / 3600.0


def _cadence(elapsed_hours: np.ndarray) -> tuple[float, float]:
    if elapsed_hours.size < 2:
        return float("nan"), float("nan")
    dt = np.diff(elapsed_hours) * 3600.0
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(dt))
    cv = float(np.std(dt) / med) if med > 0 else float("nan")
    return med, cv


def _time_aware_ema(values: np.ndarray, elapsed_hours: np.ndarray, tau: float,
                    updatable: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    state = np.nan
    prev_t = elapsed_hours[0] if elapsed_hours.size else 0.0
    for i, (v, t, ok) in enumerate(zip(values, elapsed_hours, updatable)):
        if ok and np.isfinite(v):
            if not np.isfinite(state):
                state = float(v)
            else:
                dt = max(float(t) - float(prev_t), 0.0)
                alpha = 1.0 - np.exp(-dt / tau) if tau > 0 else 1.0
                state = alpha * float(v) + (1.0 - alpha) * state
            prev_t = float(t)
        out[i] = state
    return out


def _resolve_healthy_mask(df: pd.DataFrame, elapsed_hours: np.ndarray,
                          valid: np.ndarray, cfg: ScoringConfig) -> np.ndarray:
    if cfg.healthy_mask_column is not None:
        if cfg.healthy_mask_column not in df.columns:
            raise KeyError(f"healthy_mask_column {cfg.healthy_mask_column!r} not in frame")
        mask = df[cfg.healthy_mask_column].to_numpy(dtype=bool)
        return mask & valid

    total_span = float(elapsed_hours[-1] - elapsed_hours[0]) if elapsed_hours.size else 0.0
    if cfg.healthy_window_hours is not None:
        span = float(cfg.healthy_window_hours)
    else:
        span = total_span * float(cfg.healthy_window_ratio)
    mask = (elapsed_hours - elapsed_hours[0]) <= span
    mask &= valid

    if mask.sum() < cfg.min_healthy_samples:
        # Fall back to the first min_healthy_samples valid rows so a short or
        # coarsely sampled record still yields a baseline.
        idx = np.flatnonzero(valid)[: cfg.min_healthy_samples]
        if idx.size == 0:
            raise ValueError("no valid rows available to establish a healthy baseline")
        mask = np.zeros(len(df), dtype=bool)
        mask[idx] = True
        warnings.warn(
            f"healthy window held only {int((elapsed_hours - elapsed_hours[0] <= span).sum())} "
            f"valid rows; widened to the first {idx.size} valid rows",
            RuntimeWarning,
            stacklevel=2,
        )
    return mask


def score_run(
    df: pd.DataFrame,
    config: ScoringConfig | None = None,
    timestamp_column: str = "timestamp",
    channels: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, ScoringReport]:
    """Apply the Vayeron control limits to one machine record.

    Returns a copy of ``df`` with the intermediate quantities and the target
    columns (``anomaly_score``, ``is_anomaly``, ``anomaly_tier``,
    ``fault_channel``) appended, plus a :class:`ScoringReport` describing the
    baseline that was fitted.
    """
    cfg = config or ScoringConfig()
    if timestamp_column not in df.columns:
        raise KeyError(f"timestamp column {timestamp_column!r} not in frame")

    out = df.sort_values(timestamp_column, kind="mergesort").reset_index(drop=True).copy()
    elapsed = _elapsed_hours(out[timestamp_column])
    out["elapsed_hours"] = elapsed
    cadence_s, cadence_cv = _cadence(elapsed)

    requested = list(channels) if channels is not None else [
        c for c in cfg.alert_limits if c not in THERMAL_CHANNELS]

    # --- derived thermal channels ------------------------------------------
    if cfg.thermal_channels and {"temp1", "temp2"}.issubset(out.columns):
        t1 = out["temp1"].astype(float)
        t2 = out["temp2"].astype(float)
        out["temp_max"] = np.maximum(t1, t2)
        out["temp_asymmetry"] = (t1 - t2).abs()
        for ch in cfg.thermal_channels:
            if ch not in THERMAL_CHANNELS:
                raise ValueError(f"unknown thermal channel {ch!r}")
            requested.append(ch)
    elif cfg.thermal_channels:
        raise KeyError("thermal_channels requested but temp1/temp2 are not in the frame")

    # --- speed normalisation ------------------------------------------------
    source_columns: dict[str, str] = {}
    if "rms" in requested and "rms" in out.columns:
        if cfg.normalize_rms_by_rpm and "rpm" in out.columns:
            speed = out["rpm"].to_numpy(dtype=float) / float(cfg.rpm_reference)
            with np.errstate(divide="ignore", invalid="ignore"):
                out["rms_norm"] = np.where(speed > 0, out["rms"].to_numpy(float) / speed, np.nan)
            source_columns["rms"] = "rms_norm"
        else:
            source_columns["rms"] = "rms"
    for ch in requested:
        if ch != "rms" and ch in out.columns:
            source_columns[ch] = ch

    scored = [c for c in requested if c in source_columns]
    if not scored:
        raise ValueError(
            f"none of the requested channels {requested} are present; frame has {list(out.columns)}"
        )

    # --- quality gates ------------------------------------------------------
    valid = np.ones(len(out), dtype=bool)
    if cfg.rpm_gate is not None and "rpm" in out.columns:
        lo, hi = cfg.rpm_gate
        rpm = out["rpm"].to_numpy(dtype=float)
        valid &= np.isfinite(rpm) & (rpm >= lo) & (rpm <= hi)
    if cfg.drop_zero_rms and "rms" in out.columns:
        rms = out["rms"].to_numpy(dtype=float)
        valid &= np.isfinite(rms) & (rms > 0)
    for ch in scored:
        valid &= np.isfinite(out[source_columns[ch]].to_numpy(dtype=float))
    out["valid_analysis"] = valid

    if not valid.any():
        raise ValueError(
            "quality gates rejected every row - check rpm_gate "
            f"{cfg.rpm_gate} against the record's actual speed range"
        )

    # --- healthy baseline ---------------------------------------------------
    healthy = _resolve_healthy_mask(out, elapsed, valid, cfg)
    out["is_healthy_baseline"] = healthy

    baselines: dict[str, ChannelBaseline] = {}
    severity_cols: list[str] = []
    limits = cfg.limits_for(scored)

    for ch in scored:
        col = source_columns[ch]
        x = out[col].to_numpy(dtype=float)
        ref = x[healthy]
        ref = ref[np.isfinite(ref)]
        med = float(np.median(ref))
        mad = float(np.median(np.abs(ref - med)))
        floor = max(
            cfg.mad_floor_relative * abs(med),
            cfg.mad_floor_absolute,
            cfg.min_mad_fraction_of_median * abs(med) / _MAD_TO_SIGMA,
        )
        floored = mad < floor
        if floored:
            mad = floor
        baselines[ch] = ChannelBaseline(ch, col, med, mad, floored, int(ref.size))

        z = (x - med) / (mad * _MAD_TO_SIGMA)
        out[f"z_{ch}"] = z
        sev = z / limits[ch]
        if ch == "temp_asymmetry" and cfg.temp_asymmetry_floor_c:
            # Section 6.2 gives an absolute engineering threshold: a >4 C
            # divergence is an anomaly however tight the statistical baseline
            # is. Expressed on the same severity scale, |dT| = 4 C is s = 1.0.
            sev = np.maximum(sev, x / float(cfg.temp_asymmetry_floor_c))
        if cfg.clip_negative_severity:
            sev = np.maximum(sev, 0.0)
        out[f"severity_{ch}"] = sev
        severity_cols.append(f"severity_{ch}")

    # --- aggregate, smooth, map ---------------------------------------------
    sev_matrix = out[severity_cols].to_numpy(dtype=float)
    sev_masked = np.where(valid[:, None], sev_matrix, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if cfg.aggregation == "max":
            agg = np.nanmax(sev_masked, axis=1)
        elif cfg.aggregation == "kth_max":
            k = min(max(int(cfg.aggregation_k), 1), sev_masked.shape[1])
            ordered = -np.sort(-np.nan_to_num(sev_masked, nan=-np.inf), axis=1)
            agg = ordered[:, k - 1]
            agg = np.where(np.isfinite(agg), agg, np.nan)
        else:
            raise ValueError("aggregation must be 'max' or 'kth_max'")
    out["max_severity"] = agg

    if cfg.ema_mode not in {"span", "time"}:
        raise ValueError("ema_mode must be 'span' or 'time'")
    if cfg.ema_mode == "span" and np.isfinite(cadence_cv) and cadence_cv > cfg.irregular_cadence_cv:
        warnings.warn(
            f"acquisition cadence is irregular (cv={cadence_cv:.2f}); "
            "ema_mode='time' tracks the 2 h time constant more faithfully",
            RuntimeWarning,
            stacklevel=2,
        )

    if cfg.prefilter_median_samples and cfg.prefilter_median_samples > 1:
        out["max_severity"] = (
            out["max_severity"]
            .rolling(int(cfg.prefilter_median_samples), center=True, min_periods=1)
            .median()
        )

    ema_span: float | None = None
    if cfg.ema_mode == "span":
        samples_per_hour = 3600.0 / cadence_s if cadence_s and np.isfinite(cadence_s) else 1.0
        ema_span = max(float(cfg.ema_tau_hours) * samples_per_hour, 1.0)
        series = out["max_severity"].where(out["valid_analysis"])
        smoothed = series.ewm(span=ema_span, adjust=False, ignore_na=True).mean().to_numpy()
    else:
        smoothed = _time_aware_ema(
            out["max_severity"].to_numpy(dtype=float), elapsed, float(cfg.ema_tau_hours), valid
        )
    out["smoothed_severity"] = smoothed

    y = 1.0 / (1.0 + np.exp(-cfg.logistic_gain * (smoothed - cfg.logistic_center)))
    y = np.where(np.isfinite(y), y, 0.0)
    y = np.where(valid, y, 0.0)  # gated rows exit with Y = 0 (KX-VAY-012 section 9)
    out["anomaly_score"] = y

    out["anomaly_tier"] = pd.Categorical(
        np.select(
            [y < TIER_EDGES[0], y < TIER_EDGES[1], y < TIER_EDGES[2]],
            list(TIER_NAMES[:3]),
            default=TIER_NAMES[3],
        ),
        categories=list(TIER_NAMES),
        ordered=True,
    )
    out["is_anomaly"] = (y >= TIER_EDGES[1]).astype(int)

    attribution_source = out[severity_cols]
    if cfg.attribution == "max_severity":
        driver = attribution_source.idxmax(axis=1).astype("object")
    elif cfg.attribution == "spectral_priority":
        spectral = [f"severity_{c}" for c in SPECTRAL_CHANNELS if f"severity_{c}" in severity_cols]
        driver = attribution_source.idxmax(axis=1).astype("object")
        if spectral:
            spec_src = attribution_source[spectral]
            spec_best = spec_src.idxmax(axis=1).astype("object")
            spec_above = spec_src.max(axis=1) >= 1.0
            driver = driver.where(~spec_above, spec_best)
    else:
        raise ValueError("attribution must be 'max_severity' or 'spectral_priority'")
    driver = driver.where(driver.isna(), driver.str.replace("severity_", "", regex=False))
    driver[~valid] = "none"
    driver[y < TIER_EDGES[0]] = "none"
    out["fault_channel"] = driver.fillna("none")

    # --- thermal asymmetry diagnostic (KX-VAY-012 section 6.2) --------------
    if {"temp1", "temp2"}.issubset(out.columns):
        out["temp_delta"] = out["temp1"].astype(float) - out["temp2"].astype(float)
        out["thermal_asymmetry_flag"] = (out["temp_delta"].abs() > 4.0) & valid

    report = ScoringReport(
        baselines=baselines,
        healthy_rows=int(healthy.sum()),
        healthy_span_hours=float(elapsed[healthy].max() - elapsed[healthy].min())
        if healthy.any() else 0.0,
        total_rows=int(len(out)),
        gated_rows=int((~valid).sum()),
        median_cadence_seconds=cadence_s,
        cadence_cv=cadence_cv,
        ema_mode=cfg.ema_mode,
        ema_span_samples=ema_span,
        channels_scored=scored,
    )
    return out, report


def apply_vayeron_control_limits(
    df: pd.DataFrame,
    healthy_window_ratio: float = 0.10,
    ema_span_hours: float = 2.0,
    samples_per_hour: float | None = None,  # noqa: ARG001 - compatibility only
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Backwards-compatible wrapper matching the original prototype signature.

    ``samples_per_hour`` is accepted for compatibility and ignored: the cadence
    is measured from the timestamps instead of being asserted.
    """  # noqa: ARG001
    cfg = ScoringConfig(
        healthy_window_ratio=healthy_window_ratio,
        ema_tau_hours=ema_span_hours,
        **kwargs,
    )
    scored, report = score_run(df, cfg)
    baselines = {k: {"median": v.median, "mad": v.mad} for k, v in report.baselines.items()}
    return scored, baselines


def first_sustained_crossing(
    df: pd.DataFrame,
    threshold: float,
    persistence: int = 3,
    score_column: str = "anomaly_score",
    valid_column: str = "valid_analysis",
) -> int | None:
    """Index of the first sample beginning ``persistence`` consecutive samples at/above ``threshold``.

    Single-sample spikes are explicitly not anomalies (KX-VAY-012 section 6.1
    item 6), so a crossing must persist.
    """
    y = df[score_column].to_numpy(dtype=float)
    if valid_column in df.columns:
        ok = df[valid_column].to_numpy(dtype=bool)
    else:
        ok = np.ones_like(y, dtype=bool)
    above = (y >= threshold) & ok
    if persistence <= 1:
        idx = np.flatnonzero(above)
        return int(idx[0]) if idx.size else None
    run = 0
    for i, flag in enumerate(above):
        run = run + 1 if flag else 0
        if run >= persistence:
            return int(i - persistence + 1)
    return None
