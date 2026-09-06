"""Multi-seed, multi-fault-mode campaign over the run-to-failure surrogate.

A single surrogate run tells you the pipeline executes. It does not tell you
whether the lead time is a property of the method or of one lucky seed and one
fault mode. This runs every modelled failure mode across several seeds, scores
each under the delivered specification and under the median-prefiltered
variant, and reports the spread.

Everything here is modelled data. Its purpose is to state how much confidence
the surrogate can carry, and to say which claims must wait for the real archive.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .control_limits import ScoringConfig, score_run
from .features import FeatureConfig, waveform_features
from .mendeley import failure_index
from .report import attach_ground_truth, lead_time_analysis
from .synthetic import FAULT_MODES, SurrogateConfig, generate_run

# Scoring variants, each a delta on the base config. Feature variants (below)
# are crossed with these.
VARIANTS: dict[str, dict] = {
    "spec": {},                                    # KX-VAY-012 as written
    "prefilter": {"prefilter_median_samples": 3},  # median filter ahead of the EMA
    "prefilter+specattr": {"prefilter_median_samples": 3,
                           "attribution": "spectral_priority"},
}

# Feature variants: what the edge computes. "single_bin" is the Smart-Idler
# firmware's Goertzel ratio; "sideband" recombines the +/-1 modulation
# sidebands that a rotating defect splits its energy into.
FEATURE_VARIANTS: dict[str, dict] = {
    "single_bin": {},
    "sideband": {"use_sidebands": True},
}


def _extract(scfg: SurrogateConfig) -> dict[str, pd.DataFrame]:
    """One waveform pass, one feature table per feature variant."""
    meta, waves = generate_run(scfg)
    configs = {name: FeatureConfig(fs=scfg.fs, rpm=scfg.rpm, **kw)
               for name, kw in FEATURE_VARIANTS.items()}
    rows: dict[str, list[dict]] = {name: [] for name in configs}
    for (_, row), wave in zip(meta.iterrows(), waves):
        base = row.to_dict()
        for name, fcfg in configs.items():
            rec = dict(base)
            rec.update(waveform_features(wave, fcfg))
            rows[name].append(rec)
    return {name: pd.DataFrame(r) for name, r in rows.items()}


def run(
    fault_modes: list[str] | None = None,
    seeds: list[int] | None = None,
    base_config: ScoringConfig | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    modes = fault_modes or list(FAULT_MODES)
    seeds = seeds or [20260906 + 1000 * i for i in range(6)]
    base = base_config or ScoringConfig(
        rpm_gate=(1500.0, 2000.0), rpm_reference=1770.0, healthy_window_ratio=0.15
    )

    rows: list[dict] = []
    for mode in modes:
        expected = FAULT_MODES[mode]["channel"]
        for seed in seeds:
            scfg = SurrogateConfig(fault_mode=mode, seed=seed)
            feature_sets = _extract(scfg)
            for feature_variant, features in feature_sets.items():
                fail_idx = failure_index(features)
                for variant, overrides in VARIANTS.items():
                    cfg = replace(base, **overrides)
                    scored, _ = score_run(features, cfg)
                    lead = attach_ground_truth(
                        scored, lead_time_analysis(scored, fail_idx, cfg))
                    alert = next(c for c in lead.crossings if c.tier == "alert")
                    g = lead.ground_truth
                    rows.append({
                        "fault_mode": mode,
                        "expected_channel": expected,
                        "seed": seed,
                        "features": feature_variant,
                        "variant": variant,
                        "life_hours": float(scored["elapsed_hours"].iloc[fail_idx]),
                        "detected": bool(alert.crossed and alert.terminal_index is not None),
                        "first_alert_lead_h": alert.lead_time_hours,
                        "terminal_alert_lead_h": alert.terminal_lead_time_hours,
                        "terminal_life_fraction": (
                            alert.terminal_lead_time_hours
                            / scored["elapsed_hours"].iloc[fail_idx]
                            if alert.terminal_lead_time_hours is not None else None),
                        "first_driver": alert.driving_channel,
                        "terminal_driver": alert.terminal_driving_channel,
                        "driver_correct": alert.terminal_driving_channel == expected,
                        "recovering_episodes": alert.prior_episodes,
                        "false_alarm_h_pre_onset": g.false_alarm_hours_before_onset if g else None,
                        "detection_delay_after_onset_h": g.detection_delay_hours if g else None,
                        "peak_score": lead.peak_score,
                    })
            if progress:
                tail = rows[-len(VARIANTS) * len(FEATURE_VARIANTS):]
                ok = sum(r["driver_correct"] for r in tail)
                print(f"  {mode:11s} seed={seed}  attribution {ok}/{len(tail)} correct",
                      flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        lead = pd.to_numeric(g["terminal_alert_lead_h"], errors="coerce").dropna()
        frac = pd.to_numeric(g["terminal_life_fraction"], errors="coerce").dropna()
        fa = pd.to_numeric(g["false_alarm_h_pre_onset"], errors="coerce").dropna()
        return pd.Series({
            "runs": len(g),
            "detected": int(g["detected"].sum()),
            "driver_correct": int(g["driver_correct"].sum()),
            "terminal_lead_h_median": lead.median() if not lead.empty else np.nan,
            "terminal_lead_h_min": lead.min() if not lead.empty else np.nan,
            "terminal_lead_h_max": lead.max() if not lead.empty else np.nan,
            "life_fraction_median": frac.median() if not frac.empty else np.nan,
            "false_alarm_h_median": fa.median() if not fa.empty else np.nan,
            "false_alarm_h_max": fa.max() if not fa.empty else np.nan,
        })

    return df.groupby(["fault_mode", "features", "variant"], sort=False).apply(
        agg, include_groups=False).reset_index()


def render_markdown(df: pd.DataFrame, summary: pd.DataFrame) -> str:
    n_seeds = df["seed"].nunique()
    modes = df["fault_mode"].nunique()
    lines = [
        "# Multi-seed surrogate campaign",
        "",
        f"{modes} modelled failure modes x {n_seeds} seeds x {len(FEATURE_VARIANTS)} feature "
        f"variants x {len(VARIANTS)} scoring variants = {len(df)} scored runs. **All data is modelled**; this measures how much the "
        "surrogate's lead-time claim depends on the seed and the fault mode, not how a "
        "physical bearing behaves.",
        "",
        "## Detection rate and lead time by failure mode",
        "",
        "| fault mode | variant | detected | driver @ first | driver @ terminal | "
        "terminal lead (h) | % of life | "
        "pre-onset false alarm (h) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lead = ("never" if not np.isfinite(r["terminal_lead_h_median"])
                else f"{r['terminal_lead_h_median']:.2f} "
                     f"({r['terminal_lead_h_min']:.2f}-{r['terminal_lead_h_max']:.2f})")
        frac = ("-" if not np.isfinite(r["life_fraction_median"])
                else f"{100 * r['life_fraction_median']:.0f}%")
        fa = ("-" if not np.isfinite(r["false_alarm_h_median"])
              else f"{r['false_alarm_h_median']:.2f} (max {r['false_alarm_h_max']:.2f})")
        lines.append(
            f"| `{r['fault_mode']}` | {r['features']} | {r['variant']} | "
            f"{int(r['detected'])}/{int(r['runs'])} | "
            f"{int(r['driver_correct'])}/{int(r['runs'])} | {lead} | {frac} | {fa} |")
    lines.append("")
    return "\n".join(lines)


def main(out_dir: str | Path = "reports/campaign") -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = run()
    summary = summarise(df)
    df.to_csv(out / "campaign.csv", index=False)
    summary.to_csv(out / "campaign_summary.csv", index=False)
    md = render_markdown(df, summary)
    (out / "campaign.md").write_text(md)
    print("\n" + md)
    return df


if __name__ == "__main__":
    main()
