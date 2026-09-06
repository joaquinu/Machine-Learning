"""Unit tests for the KX-VAY-012 target construction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vayeron.control_limits import (  # noqa: E402
    TIER_EDGES,
    VAYERON_ALERT_LIMITS,
    ScoringConfig,
    first_sustained_crossing,
    score_run,
)
from vayeron.features import (  # noqa: E402
    FeatureConfig,
    geometric_defect_orders,
    waveform_features,
)


def make_frame(n=400, rpm=600.0, seed=0, rms_scale=1.0, ramp=0.0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="10min")
    trend = np.linspace(0.0, ramp, n)
    return pd.DataFrame({
        "timestamp": ts,
        "rpm": np.full(n, rpm),
        "rms": (50.0 + rng.normal(0, 1.0, n)) * rms_scale + trend,
        "bpfo": 2.0 + rng.normal(0, 0.2, n),
        "bpfi": 1.5 + rng.normal(0, 0.15, n),
        "bsf": 1.8 + rng.normal(0, 0.18, n),
        "ftf": 1.6 + rng.normal(0, 0.16, n),
    })


def test_limits_match_the_specification():
    assert VAYERON_ALERT_LIMITS["rms"] == 3.5
    assert all(VAYERON_ALERT_LIMITS[c] == 4.0 for c in ("bpfo", "bpfi", "bsf", "ftf"))


def test_logistic_is_calibrated_so_severity_one_maps_to_half():
    cfg = ScoringConfig()
    y = 1.0 / (1.0 + np.exp(-cfg.logistic_gain * (1.0 - cfg.logistic_center)))
    assert y == pytest.approx(0.50)


def test_healthy_run_stays_normal():
    scored, report = score_run(make_frame(), ScoringConfig())
    assert (scored["anomaly_tier"] == "normal").mean() > 0.95
    assert report.channels_scored == ["rms", "bpfo", "bpfi", "bsf", "ftf"]


def test_sustained_excursion_reaches_alert_and_is_attributed():
    df = make_frame()
    # A coherent, sustained inner-race excursion in the last quarter.
    df.loc[300:, "bpfi"] += 3.0
    scored, _ = score_run(df, ScoringConfig(healthy_window_ratio=0.25))
    tail = scored.iloc[320:]
    assert tail["anomaly_score"].median() >= TIER_EDGES[1]
    assert tail["fault_channel"].mode().iat[0] == "bpfi"
    assert tail["is_anomaly"].mean() > 0.9


def test_scoring_is_invariant_to_channel_rescaling():
    """Robust z-scores use median/MAD, so a constant gain must not move Y."""
    base, _ = score_run(make_frame(), ScoringConfig())
    scaled, _ = score_run(make_frame(rms_scale=7.0), ScoringConfig())
    np.testing.assert_allclose(base["anomaly_score"], scaled["anomaly_score"], atol=1e-9)


def test_rpm_normalisation_cancels_a_pure_speed_change():
    df = make_frame(n=300)
    faster = slice(150, None)
    df.loc[faster, "rpm"] = 900.0
    df.loc[faster, "rms"] = df.loc[faster, "rms"] * 1.5  # vibration scales with speed
    cfg = ScoringConfig(rpm_gate=(350.0, 1500.0), healthy_window_ratio=0.4)
    scored, _ = score_run(df, cfg)
    assert (scored["anomaly_tier"] == "normal").mean() > 0.95

    unnormalised, _ = score_run(df, ScoringConfig(healthy_window_ratio=0.4,
                                                  normalize_rms_by_rpm=False))
    assert unnormalised["is_anomaly"].sum() > scored["is_anomaly"].sum()


def test_quality_gates_zero_the_score():
    df = make_frame()
    df.loc[10:14, "rms"] = 0.0        # speed-change transient
    df.loc[20:24, "rpm"] = 120.0      # below the conveyor envelope
    scored, report = score_run(df, ScoringConfig())
    assert not scored.loc[10:14, "valid_analysis"].any()
    assert not scored.loc[20:24, "valid_analysis"].any()
    gated = ~scored["valid_analysis"]
    assert (scored.loc[gated, "anomaly_score"] == 0).all()
    assert (scored.loc[gated, "fault_channel"] == "none").all()
    assert report.gated_rows == 10


def test_all_rows_gated_is_an_error_not_a_silent_zero():
    df = make_frame(rpm=1770.0)  # a lab rig, outside the conveyor gate
    with pytest.raises(ValueError, match="rpm_gate"):
        score_run(df, ScoringConfig())
    scored, _ = score_run(df, ScoringConfig(rpm_gate=(1500.0, 2000.0)))
    assert scored["valid_analysis"].all()


def test_tier_boundaries():
    df = make_frame(n=200)
    df.loc[100:, "bpfo"] += 5.0
    scored, _ = score_run(df, ScoringConfig(healthy_window_ratio=0.4))
    y = scored["anomaly_score"].to_numpy()
    tiers = scored["anomaly_tier"].astype(str).to_numpy()
    assert set(tiers[y < 0.35]) <= {"normal"}
    assert set(tiers[(y >= 0.35) & (y < 0.50)]) <= {"watch"}
    assert set(tiers[(y >= 0.50) & (y < 0.80)]) <= {"alert"}
    assert set(tiers[y >= 0.80]) <= {"critical"}
    np.testing.assert_array_equal(scored["is_anomaly"].to_numpy(), (y >= 0.50).astype(int))


def test_persistence_rejects_a_single_sample_crossing():
    df = pd.DataFrame({
        "anomaly_score": [0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1],
        "valid_analysis": True,
    })
    assert first_sustained_crossing(df, 0.5, persistence=1) == 1
    assert first_sustained_crossing(df, 0.5, persistence=3) == 4
    assert first_sustained_crossing(df, 0.5, persistence=5) is None


def test_median_prefilter_removes_a_lone_spike_but_keeps_a_real_excursion():
    df = make_frame(n=300)
    df.loc[120, "bpfo"] += 12.0        # single-sample knock
    df.loc[240:, "bpfo"] += 4.0        # sustained defect

    spec, _ = score_run(df, ScoringConfig(healthy_window_ratio=0.3))
    filt, _ = score_run(df, ScoringConfig(healthy_window_ratio=0.3,
                                          prefilter_median_samples=3))
    # The EMA smears the lone spike across many samples under the plain spec.
    assert spec.loc[118:135, "anomaly_score"].max() > filt.loc[118:135, "anomaly_score"].max()
    assert filt.loc[118:135, "is_anomaly"].sum() < spec.loc[118:135, "is_anomaly"].sum()
    # The real excursion survives either way.
    assert filt.loc[260:, "is_anomaly"].mean() > 0.9


def test_mad_floor_desensitises_a_hyperstationary_baseline():
    n = 300
    ts = pd.date_range("2026-01-01", periods=n, freq="10min")
    df = pd.DataFrame({
        "timestamp": ts,
        "rpm": np.full(n, 600.0),
        "rms": np.full(n, 50.0) + np.concatenate([np.zeros(200), np.full(100, 1.0)]),
        "bpfo": np.full(n, 2.0),
    })
    df["rms"] += np.random.default_rng(1).normal(0, 1e-3, n)
    df["bpfo"] += np.random.default_rng(2).normal(0, 1e-3, n)

    hot, _ = score_run(df, ScoringConfig(healthy_window_ratio=0.5))
    calm, _ = score_run(df, ScoringConfig(healthy_window_ratio=0.5,
                                          min_mad_fraction_of_median=0.05))
    assert hot["is_anomaly"].tail(50).all()      # 2% step trips a 3.5 sigma limit
    assert not calm["is_anomaly"].tail(50).any()  # ... unless 5% is declared the floor


def test_kth_max_aggregation_ignores_a_single_drifting_channel():
    df = make_frame(n=300)
    df.loc[200:, "bpfo"] += 4.0  # one channel only
    spec, _ = score_run(df, ScoringConfig(healthy_window_ratio=0.3))
    coherent, _ = score_run(df, ScoringConfig(healthy_window_ratio=0.3,
                                              aggregation="kth_max", aggregation_k=2))
    assert spec["is_anomaly"].tail(80).mean() > 0.9
    assert coherent["is_anomaly"].tail(80).mean() < 0.1


def test_irregular_cadence_warns_and_time_ema_is_available():
    df = make_frame(n=200)
    jitter = pd.to_timedelta(np.random.default_rng(3).integers(0, 1800, 200), unit="s")
    df["timestamp"] = df["timestamp"] + jitter
    with pytest.warns(RuntimeWarning, match="irregular"):
        score_run(df.sort_values("timestamp"), ScoringConfig())
    scored, report = score_run(df.sort_values("timestamp"), ScoringConfig(ema_mode="time"))
    assert report.ema_mode == "time"
    assert scored["anomaly_score"].notna().all()


def test_envelope_ratio_finds_the_injected_defect_line():
    fs, rpm = 25600.0, 1770.0
    cfg = FeatureConfig(fs=fs, rpm=rpm)
    n = int(fs)
    t = np.arange(n) / fs
    shaft = rpm / 60.0
    fd = shaft * cfg.defect_orders["bpfo"]

    rng = np.random.default_rng(7)
    resp_t = np.arange(0, 0.004, 1 / fs)
    resp = np.exp(-1500 * resp_t) * np.sin(2 * np.pi * 3500 * resp_t)
    train = np.zeros(n)
    for k in range(int(fd) + 1):
        i = int(k / fd * fs)
        if i < n:
            train[i] = 1.0
    sig = 3.0 * np.convolve(train, resp)[:n] + rng.normal(0, 0.4, n) \
        + 0.5 * np.sin(2 * np.pi * shaft * t)

    feats = waveform_features(sig, cfg)
    assert feats["bpfo"] > 10 * max(feats["bpfi"], feats["bsf"], feats["ftf"])
    assert feats["rms"] > 0


def test_geometric_orders_are_physically_ordered():
    orders = geometric_defect_orders(9, 7.94, 39.04, 0.0)
    assert 0.3 < orders["ftf"] < 0.5
    assert orders["bpfo"] < orders["bpfi"]
    assert orders["bpfo"] + orders["bpfi"] == pytest.approx(9.0, rel=1e-6)


def test_prototype_wrapper_still_works():
    from vayeron.control_limits import apply_vayeron_control_limits

    scored, baselines = apply_vayeron_control_limits(make_frame(), samples_per_hour=6)
    assert {"anomaly_score", "anomaly_tier", "fault_channel"} <= set(scored.columns)
    assert set(baselines) == {"rms", "bpfo", "bpfi", "bsf", "ftf"}
