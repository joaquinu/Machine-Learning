"""Sensitivity of the lead-time result to the analyst's free choices.

The Vayeron control limits are fixed by KX-VAY-012 (3.5 sigma / 4.0 sigma), but
four things are not: how much of the record is declared healthy, the EMA time
constant, how many consecutive acquisitions a crossing must persist for, and
whether a minimum resolvable change is imposed on the baseline MAD. A lead time
that only survives one particular combination is not a validation result, so
every run sweeps them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd

from .control_limits import ScoringConfig, score_run
from .report import lead_time_analysis


def sweep(
    features: pd.DataFrame,
    base_config: ScoringConfig,
    failure_idx: int,
    healthy_ratios: Sequence[float] = (0.05, 0.10, 0.15, 0.25),
    ema_taus: Sequence[float] = (1.0, 2.0, 4.0),
    persistences: Sequence[int] = (1, 3, 6),
    mad_floors: Sequence[float] = (0.0, 0.01, 0.02),
    limit_scales: Sequence[float] = (1.0,),
) -> pd.DataFrame:
    """One row per parameter combination, carrying the alert lead time."""
    rows: list[dict] = []
    for hr in healthy_ratios:
        for tau in ema_taus:
            for floor in mad_floors:
                for scale in limit_scales:
                    limits = {k: v * scale for k, v in base_config.alert_limits.items()}
                    cfg = replace(
                        base_config,
                        alert_limits=limits,
                        healthy_window_ratio=hr,
                        healthy_window_hours=None,
                        ema_tau_hours=tau,
                        min_mad_fraction_of_median=floor,
                    )
                    try:
                        scored, _ = score_run(features, cfg)
                    except (ValueError, KeyError) as exc:  # pragma: no cover
                        rows.append({"healthy_ratio": hr, "ema_tau_h": tau,
                                     "mad_floor": floor, "limit_scale": scale,
                                     "error": str(exc)})
                        continue
                    for pers in persistences:
                        lead = lead_time_analysis(scored, failure_idx,
                                                  replace(cfg, persistence_samples=pers))
                        by_tier = {c.tier: c for c in lead.crossings}
                        rows.append({
                            "healthy_ratio": hr,
                            "ema_tau_h": tau,
                            "persistence": pers,
                            "mad_floor": floor,
                            "limit_scale": scale,
                            "watch_lead_h": by_tier["watch"].lead_time_hours,
                            "alert_lead_h": by_tier["alert"].lead_time_hours,
                            "critical_lead_h": by_tier["critical"].lead_time_hours,
                            "alert_terminal_lead_h": by_tier["alert"].terminal_lead_time_hours,
                            "alert_prior_episodes": by_tier["alert"].prior_episodes,
                            "alert_driver": by_tier["alert"].driving_channel,
                            "alert_life_fraction": by_tier["alert"].lead_time_life_fraction,
                            "peak_score": lead.peak_score,
                            "alert_share": lead.tier_shares.get("alert", 0.0)
                                           + lead.tier_shares.get("critical", 0.0),
                        })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    lead = pd.to_numeric(df.get("alert_lead_h"), errors="coerce").dropna()
    if lead.empty:
        return {"n_configs": int(len(df)), "alert_never_crossed": True}
    term = pd.to_numeric(df.get("alert_terminal_lead_h"), errors="coerce").dropna()
    return {
        "n_configs": int(len(df)),
        "alert_terminal_lead_h_median": float(term.median()) if not term.empty else None,
        "alert_terminal_lead_h_min": float(term.min()) if not term.empty else None,
        "alert_terminal_lead_h_max": float(term.max()) if not term.empty else None,
        "median_prior_episodes": float(pd.to_numeric(
            df.get("alert_prior_episodes"), errors="coerce").dropna().median())
            if "alert_prior_episodes" in df else None,
        "alert_lead_h_median": float(lead.median()),
        "alert_lead_h_min": float(lead.min()),
        "alert_lead_h_max": float(lead.max()),
        "alert_lead_h_iqr": [float(lead.quantile(0.25)), float(lead.quantile(0.75))],
        "configs_with_no_alert": int(df["alert_lead_h"].isna().sum()),
        "dominant_driver": (df["alert_driver"].mode().iat[0]
                            if "alert_driver" in df and df["alert_driver"].notna().any() else None),
    }


def render_markdown(df: pd.DataFrame, summary: dict) -> str:
    lines = ["## Sensitivity of the lead time to analyst choices", ""]
    lines.append(f"{summary['n_configs']} parameter combinations "
                 "(healthy window x EMA tau x persistence x MAD floor).")
    lines.append("")
    if summary.get("alert_never_crossed"):
        lines.append("**No configuration produced an alert crossing.**")
        return "\n".join(lines)
    lines.append(f"- Alert lead time: median **{summary['alert_lead_h_median']:,.2f} h** "
                 f"({summary['alert_lead_h_median'] / 24:,.2f} d), "
                 f"range {summary['alert_lead_h_min']:,.2f}-{summary['alert_lead_h_max']:,.2f} h, "
                 f"IQR {summary['alert_lead_h_iqr'][0]:,.2f}-{summary['alert_lead_h_iqr'][1]:,.2f} h.")
    if summary.get("alert_terminal_lead_h_median") is not None:
        lines.append(f"- Terminal (irreversible) alert lead time: median "
                     f"**{summary['alert_terminal_lead_h_median']:,.2f} h** "
                     f"({summary['alert_terminal_lead_h_median'] / 24:,.2f} d), range "
                     f"{summary['alert_terminal_lead_h_min']:,.2f}-"
                     f"{summary['alert_terminal_lead_h_max']:,.2f} h.")
    if summary.get("median_prior_episodes") is not None:
        lines.append(f"- Median number of earlier recovering alert episodes: "
                     f"**{summary['median_prior_episodes']:.0f}**.")
    lines.append(f"- Configurations where the alert never fired before failure: "
                 f"**{summary['configs_with_no_alert']}**.")
    lines.append(f"- Most frequent driving channel at the alert crossing: "
                 f"`{summary['dominant_driver']}`.")
    lines.append("")
    piv = df.pivot_table(index="healthy_ratio", columns="ema_tau_h",
                         values="alert_lead_h", aggfunc="median")
    lines.append("Median alert lead time (h), healthy window vs EMA tau:")
    lines.append("")
    lines.append("| healthy window | " + " | ".join(f"tau = {c} h" for c in piv.columns) + " |")
    lines.append("|---" * (len(piv.columns) + 1) + "|")
    for idx, row in piv.iterrows():
        cells = " | ".join("n/a" if not np.isfinite(v) else f"{v:,.2f}" for v in row)
        lines.append(f"| {100 * idx:.0f}% of record | {cells} |")
    lines.append("")
    return "\n".join(lines)
