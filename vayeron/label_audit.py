"""How many `alert` labels in the field dataset are EMA-smeared single knocks?

KX-VAY-012 section 6.1 item 6 says an isolated spike is not an anomaly. The
persistence rule is meant to enforce that, but the 2 h EMA sits *before* it: a
one-sample excursion is spread across roughly the smoother's span, so it clears
any "N consecutive samples" test and lands in the labels as a genuine alert
episode.

This recomputes the target from the raw telemetry two ways - as delivered, and
with a median prefilter on raw severity - and reports how many labelled rows
change. Every model benchmarked in KX-VAY-013/014 was trained against the first
column, so the size of the difference is the size of the label-noise floor
those benchmarks were measuring against.

Whether that floor is large is an empirical question with a sharp answer: a lone
spike only clears the control limit if its severity exceeds ``1 / alpha`` where
``alpha`` is the EMA coefficient (see :func:`survival_threshold`). At the field
cadence of 626 s with tau = 2 h that is about 22 sigma on RMS. Smaller knocks
are harmless; larger ones manufacture whole alert episodes. Run this against the
real CSV to find out which side the field's knocks fall on.

Usage
-----
    python -m vayeron.label_audit --csv vayeron_anomaly_dataset.csv \\
        --group-by roller_id --out reports/label_audit

The CSV needs the raw channels (`rms`, `bpfo`, `bpfi`, `bsf`, `ftf`, `rpm`,
`timestamp`) - not `anomaly_score`, which is the target and is recomputed here.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .control_limits import ScoringConfig, score_run
from .report import _episodes


def _episode_table(scored: pd.DataFrame, cfg: ScoringConfig,
                   threshold: float = 0.50) -> pd.DataFrame:
    """One row per alert episode, with the evidence that produced it."""
    y = scored["anomaly_score"].to_numpy(dtype=float)
    ok = (scored["valid_analysis"].to_numpy(dtype=bool)
          if "valid_analysis" in scored.columns else np.ones_like(y, dtype=bool))
    raw = scored["max_severity"].to_numpy(dtype=float)
    elapsed = scored["elapsed_hours"].to_numpy(dtype=float)

    rows = []
    for start, end in _episodes((y >= threshold) & ok, cfg.persistence_samples):
        seg_raw = raw[start:end]
        # How much of the episode is supported by raw severity actually being
        # above the control limit, rather than by the smoother's memory of one
        # sample?
        above = np.count_nonzero(seg_raw >= 1.0)
        rows.append({
            "start": start,
            "end": end,
            "n_rows": end - start,
            "duration_h": float(elapsed[min(end, len(elapsed) - 1)] - elapsed[start]),
            "raw_samples_above_limit": int(above),
            "raw_support_fraction": above / (end - start),
            "peak_score": float(y[start:end].max()),
            "peak_raw_severity": float(np.nanmax(seg_raw)),
            # A knock-driven episode is one the raw severity supports for a
            # single sample (or none) while the smoothed score runs for many.
            "knock_driven": bool(above <= 1 and (end - start) >= 3),
        })
    return pd.DataFrame(rows)


def audit_one(df: pd.DataFrame, cfg: ScoringConfig,
              prefilter: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    spec, _ = score_run(df, cfg)
    filt, _ = score_run(df, replace(cfg, prefilter_median_samples=prefilter))

    eps = _episode_table(spec, cfg)
    knock_rows = int(eps.loc[eps["knock_driven"], "n_rows"].sum()) if not eps.empty else 0
    alert_rows = int((spec["anomaly_score"] >= 0.50).sum())

    stats = {
        "rows": int(len(spec)),
        "alert_rows_spec": alert_rows,
        "alert_rows_prefiltered": int((filt["anomaly_score"] >= 0.50).sum()),
        "alert_episodes_spec": int(len(eps)),
        "alert_episodes_knock_driven": int(eps["knock_driven"].sum()) if not eps.empty else 0,
        "alert_rows_knock_driven": knock_rows,
        "knock_row_share_of_alerts": knock_rows / alert_rows if alert_rows else 0.0,
        "tier_changed_rows": int((spec["anomaly_tier"].astype(str)
                                  != filt["anomaly_tier"].astype(str)).sum()),
        "score_mae": float((spec["anomaly_score"] - filt["anomaly_score"]).abs().mean()),
    }
    for tier in ("normal", "watch", "alert", "critical"):
        stats[f"share_{tier}_spec"] = float((spec["anomaly_tier"] == tier).mean())
        stats[f"share_{tier}_prefiltered"] = float((filt["anomaly_tier"] == tier).mean())
    return spec, eps, stats


def audit(df: pd.DataFrame, cfg: ScoringConfig | None = None,
          group_by: str | None = None, prefilter: int = 3
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or ScoringConfig()
    groups = [(k, g) for k, g in df.groupby(group_by)] if group_by else [("all", df)]

    per_group, all_eps = [], []
    for name, g in groups:
        _, eps, stats = audit_one(g.reset_index(drop=True), cfg, prefilter)
        stats["group"] = name
        per_group.append(stats)
        if not eps.empty:
            eps = eps.assign(group=name)
            all_eps.append(eps)
    return (pd.DataFrame(per_group).set_index("group").reset_index(),
            pd.concat(all_eps, ignore_index=True) if all_eps else pd.DataFrame())


def render_markdown(summary: pd.DataFrame, source: str) -> str:
    tot = summary[["rows", "alert_rows_spec", "alert_rows_prefiltered",
                   "alert_episodes_spec", "alert_episodes_knock_driven",
                   "alert_rows_knock_driven", "tier_changed_rows"]].sum()
    share = tot["alert_rows_knock_driven"] / tot["alert_rows_spec"] if tot["alert_rows_spec"] else 0

    lines = [
        "# Label audit: EMA-smeared knocks in the alert class",
        "",
        f"Source: `{source}`. The target is recomputed from raw telemetry, once as "
        "KX-VAY-012 delivers it and once with a 3-sample median prefilter on raw "
        "severity ahead of the EMA. An **episode** is a run of consecutive rows at or "
        "above Y = 0.50; it is **knock-driven** when the raw (unsmoothed) severity "
        "clears the control limit for at most one sample while the smoothed score "
        "runs for three or more.",
        "",
        "## Totals",
        "",
        f"- Rows: **{int(tot['rows']):,}**",
        f"- Alert rows as delivered: **{int(tot['alert_rows_spec']):,}**; "
        f"with the prefilter: **{int(tot['alert_rows_prefiltered']):,}**",
        f"- Alert episodes: **{int(tot['alert_episodes_spec']):,}**, of which "
        f"**{int(tot['alert_episodes_knock_driven']):,}** are knock-driven",
        f"- Alert rows attributable to knocks: **{int(tot['alert_rows_knock_driven']):,}** "
        f"(**{100 * share:.1f}%** of the alert class)",
        f"- Rows whose tier changes under the prefilter: **{int(tot['tier_changed_rows']):,}**",
        "",
        "## Tier distribution, as delivered vs prefiltered",
        "",
        "| tier | as delivered | prefiltered |",
        "|---|---|---|",
    ]
    weights = summary["rows"] / summary["rows"].sum()
    for tier in ("normal", "watch", "alert", "critical"):
        a = float((summary[f"share_{tier}_spec"] * weights).sum())
        b = float((summary[f"share_{tier}_prefiltered"] * weights).sum())
        lines.append(f"| {tier} | {100 * a:.2f}% | {100 * b:.2f}% |")
    lines += ["", "## Per group", "",
              "| group | rows | alert rows | knock-driven alert rows | share |",
              "|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(
            f"| `{r['group']}` | {int(r['rows']):,} | {int(r['alert_rows_spec']):,} | "
            f"{int(r['alert_rows_knock_driven']):,} | "
            f"{100 * r['knock_row_share_of_alerts']:.1f}% |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--group-by", default=None,
                   help="column identifying the roller, so baselines are per-asset")
    p.add_argument("--prefilter", type=int, default=3)
    p.add_argument("--persistence", type=int, default=3)
    p.add_argument("--healthy-ratio", type=float, default=0.10)
    p.add_argument("--ema-tau-hours", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=Path("reports/label_audit"))
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    cfg = ScoringConfig(healthy_window_ratio=args.healthy_ratio,
                        ema_tau_hours=args.ema_tau_hours,
                        persistence_samples=args.persistence)

    summary, episodes = audit(df, cfg, args.group_by, args.prefilter)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    episodes.to_csv(out / "episodes.csv", index=False)
    md = render_markdown(summary, str(args.csv))
    (out / "label_audit.md").write_text(md)
    print(md)
    print(f"\nWritten to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --- how large must a knock be to manufacture an alert? -----------------------

def survival_threshold(ema_span_samples: float, alert_limit: float = 3.5) -> dict:
    """Closed form for when a single-sample spike survives the EMA.

    A lone spike of severity ``S`` against a baseline near zero leaves the
    smoother at ``alpha * S`` where ``alpha = 2 / (span + 1)``. It registers as
    an alert only if that clears the control limit, so ``S >= 1 / alpha``. It
    then decays as ``(1 - alpha)^k``, so the episode lasts
    ``k <= ln(alpha * S) / -ln(1 - alpha)`` samples - which is what has to reach
    the persistence requirement.
    """
    alpha = 2.0 / (ema_span_samples + 1.0)
    s_min = 1.0 / alpha

    def episode_length(s: float) -> float:
        return np.log(alpha * s) / -np.log(1 - alpha) if alpha * s > 1 else 0.0

    return {
        "ema_span_samples": ema_span_samples,
        "alpha": alpha,
        "min_severity_to_cross": s_min,
        "min_sigma_to_cross": s_min * alert_limit,
        "min_severity_for_3_samples": next(
            (s / 100 for s in range(int(100 * s_min), int(100 * s_min * 20))
             if episode_length(s / 100) >= 3), float("nan")),
    }


def knock_threshold_sweep(knock_gains=(6, 12, 20, 30, 45, 60, 90),
                          config: ScoringConfig | None = None,
                          group_by: str = "roller_id") -> pd.DataFrame:
    """How much of the alert class a given knock magnitude manufactures.

    Regenerates the field-shaped surrogate at each knock size and audits it, so
    the answer to "do isolated knocks corrupt the labels" comes back as a curve
    against knock magnitude rather than a yes/no.
    """
    from .synthetic import FieldSurrogateConfig, generate_field_record

    cfg = config or ScoringConfig(persistence_samples=3)
    rows = []
    for gain in knock_gains:
        df = generate_field_record(FieldSurrogateConfig(knock_gain=gain))
        summary, _ = audit(df, cfg, group_by)
        alert_rows = int(summary["alert_rows_spec"].sum())
        knock_rows = int(summary["alert_rows_knock_driven"].sum())
        rows.append({
            "knock_gain_mad": gain,
            "alert_rows": alert_rows,
            "knock_driven_episodes": int(summary["alert_episodes_knock_driven"].sum()),
            "knock_driven_alert_rows": knock_rows,
            "share_of_alert_class": knock_rows / alert_rows if alert_rows else 0.0,
        })
    return pd.DataFrame(rows)
