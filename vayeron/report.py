"""Lead-time analysis and reporting for a scored run-to-failure record.

The validation question from the strategy note is a single number per tier:
how long before physical seizure does Vayeron's statistical target first,
and durably, cross that tier?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .control_limits import TIER_EDGES, ScoringConfig, ScoringReport, first_sustained_crossing


def _episodes(above: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Contiguous runs of ``above`` at least ``min_len`` long, as [start, end) pairs."""
    out: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(above) - start >= min_len:
        out.append((start, len(above)))
    return out


@dataclass
class Crossing:
    tier: str
    threshold: float
    crossed: bool
    index: int | None
    timestamp: str | None
    elapsed_hours: float | None
    lead_time_hours: float | None
    lead_time_life_fraction: float | None
    driving_channel: str | None
    # The first crossing can be a transient excursion that recovers - baseline
    # drift, a load change, a knock. The terminal crossing is the start of the
    # final, unbroken run above the threshold that reaches failure: that is the
    # one that corresponds to irreversible degradation.
    terminal_index: int | None = None
    terminal_elapsed_hours: float | None = None
    terminal_lead_time_hours: float | None = None
    terminal_driving_channel: str | None = None
    prior_episodes: int = 0
    prior_episode_hours: float = 0.0


@dataclass
class GroundTruth:
    """Only available for the surrogate, where the damage onset is known."""

    onset_index: int
    onset_elapsed_hours: float
    detection_delay_hours: float | None
    false_alarm_hours_before_onset: float


@dataclass
class LeadTimeReport:
    failure_index: int
    failure_timestamp: str
    failure_elapsed_hours: float
    total_acquisitions: int
    persistence_samples: int
    crossings: list[Crossing]
    peak_score: float
    peak_score_hours_to_failure: float
    tier_shares: dict[str, float]
    ground_truth: GroundTruth | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["crossings"] = [asdict(c) for c in self.crossings]
        return d


def attach_ground_truth(scored: pd.DataFrame, lead: LeadTimeReport,
                        damage_column: str = "true_damage") -> LeadTimeReport:
    """Score the run against a known damage onset (surrogate runs only)."""
    if damage_column not in scored.columns:
        return lead
    dmg = pd.to_numeric(scored[damage_column], errors="coerce").to_numpy(float)
    onset = np.flatnonzero(dmg > 0)
    if onset.size == 0:
        return lead
    i0 = int(onset[0])
    elapsed = scored["elapsed_hours"].to_numpy(dtype=float)
    t0 = float(elapsed[i0])

    alert = next((c for c in lead.crossings if c.tier == "alert"), None)
    detect_at = alert.terminal_elapsed_hours if alert and alert.terminal_index is not None else None
    delay = detect_at - t0 if detect_at is not None else None

    y = scored["anomaly_score"].to_numpy(dtype=float)[:i0]
    pre = elapsed[:i0]
    fa = float(np.sum(np.diff(pre, prepend=pre[0])[y >= 0.50])) if y.size else 0.0

    lead.ground_truth = GroundTruth(i0, t0, delay, fa)
    return lead


def lead_time_analysis(
    scored: pd.DataFrame,
    failure_idx: int,
    config: ScoringConfig | None = None,
) -> LeadTimeReport:
    cfg = config or ScoringConfig()
    elapsed = scored["elapsed_hours"].to_numpy(dtype=float)
    t_fail = float(elapsed[failure_idx])
    life = t_fail - float(elapsed[0]) if t_fail > elapsed[0] else float("nan")

    crossings: list[Crossing] = []
    window = scored.iloc[: failure_idx + 1]
    y_win = window["anomaly_score"].to_numpy(dtype=float)
    ok_win = (window["valid_analysis"].to_numpy(dtype=bool)
              if "valid_analysis" in window.columns else np.ones_like(y_win, dtype=bool))

    for tier, thr in zip(("watch", "alert", "critical"), TIER_EDGES):
        idx = first_sustained_crossing(window, thr, cfg.persistence_samples)
        if idx is None:
            crossings.append(Crossing(tier, thr, False, None, None, None, None, None, None))
            continue
        te = float(elapsed[idx])
        lead = t_fail - te

        eps = _episodes((y_win >= thr) & ok_win, cfg.persistence_samples)
        terminal = eps[-1] if eps and eps[-1][1] >= len(y_win) else None
        prior = [e for e in eps if terminal is None or e[0] < terminal[0]]

        crossings.append(Crossing(
            tier=tier,
            threshold=thr,
            crossed=True,
            index=int(idx),
            timestamp=str(scored["timestamp"].iloc[idx]),
            elapsed_hours=te,
            lead_time_hours=lead,
            lead_time_life_fraction=lead / life if np.isfinite(life) and life > 0 else None,
            driving_channel=str(scored["fault_channel"].iloc[idx]),
            terminal_index=int(terminal[0]) if terminal else None,
            terminal_elapsed_hours=float(elapsed[terminal[0]]) if terminal else None,
            terminal_lead_time_hours=float(t_fail - elapsed[terminal[0]]) if terminal else None,
            terminal_driving_channel=str(scored["fault_channel"].iloc[terminal[0]]) if terminal else None,
            prior_episodes=len(prior),
            prior_episode_hours=float(sum(elapsed[min(e[1], len(elapsed) - 1)] - elapsed[e[0]]
                                          for e in prior)),
        ))

    # Peak is measured up to the failure point; anything after it is post-mortem.
    y = window["anomaly_score"].to_numpy(dtype=float)
    peak_i = int(np.nanargmax(y))
    shares = (
        scored["anomaly_tier"].value_counts(normalize=True).reindex(
            ["normal", "watch", "alert", "critical"], fill_value=0.0
        ).to_dict()
    )

    return LeadTimeReport(
        failure_index=int(failure_idx),
        failure_timestamp=str(scored["timestamp"].iloc[failure_idx]),
        failure_elapsed_hours=t_fail,
        total_acquisitions=int(len(scored)),
        persistence_samples=int(cfg.persistence_samples),
        crossings=crossings,
        peak_score=float(y[peak_i]),
        peak_score_hours_to_failure=float(t_fail - elapsed[peak_i]),
        tier_shares={k: float(v) for k, v in shares.items()},
    )


def _fmt_hours(h: float | None) -> str:
    if h is None or not np.isfinite(h):
        return "n/a"
    days = h / 24.0
    return f"{h:,.2f} h ({days:,.2f} d)"


def render_markdown(
    scored: pd.DataFrame,
    scoring: ScoringReport,
    lead: LeadTimeReport,
    cfg: ScoringConfig,
    title: str,
    source_note: str,
) -> str:
    lines: list[str] = []
    a = lines.append
    a(f"# {title}")
    a("")
    a(source_note)
    a("")
    a("## Control limits applied")
    a("")
    a("| channel | z_alert | baseline median | baseline MAD | sigma-equivalent |")
    a("|---|---|---|---|---|")
    for ch, b in scoring.baselines.items():
        a(f"| `{ch}` | {cfg.alert_limits[ch]:.1f} | {b.median:,.4f} | {b.mad:,.4f} | "
          f"{b.mad * 1.4826:,.4f} |")
    a("")
    a(f"- Healthy baseline: first **{scoring.healthy_rows}** valid acquisitions "
      f"(**{scoring.healthy_span_hours:,.2f} h** of run time).")
    a(f"- Acquisition cadence: median **{scoring.median_cadence_seconds:,.1f} s** "
      f"(cv {scoring.cadence_cv:.3f}).")
    a(f"- Persistence smoothing: EMA `{scoring.ema_mode}`, tau = {cfg.ema_tau_hours} h"
      + (f", span = {scoring.ema_span_samples:.1f} samples." if scoring.ema_span_samples else "."))
    a(f"- Quality gates: rpm in {cfg.rpm_gate}, rms > 0 -> "
      f"**{scoring.gated_rows}** of {scoring.total_rows} rows gated to Y = 0.")
    a("")
    a("## Time to failure")
    a("")
    a(f"Failure reference: acquisition **{lead.failure_index}** of {lead.total_acquisitions} "
      f"at `{lead.failure_timestamp}` (t = {lead.failure_elapsed_hours:,.2f} h).")
    a("")
    a(f"A crossing counts only when it persists for **{lead.persistence_samples} consecutive "
      "acquisitions**; single-sample spikes are not anomalies (KX-VAY-012 6.1).")
    a("")
    a("| tier | Y threshold | first sustained crossing | lead time | % of life remaining | driver |")
    a("|---|---|---|---|---|---|")
    for c in lead.crossings:
        if not c.crossed:
            a(f"| **{c.tier}** | {c.threshold:.2f} | never | - | - | - |")
            continue
        pct = f"{100 * c.lead_time_life_fraction:.1f}%" if c.lead_time_life_fraction is not None else "-"
        a(f"| **{c.tier}** | {c.threshold:.2f} | acq {c.index} @ {c.elapsed_hours:,.2f} h | "
          f"{_fmt_hours(c.lead_time_hours)} | {pct} | `{c.driving_channel}` |")
    a("")
    a("The first crossing can be a recoverable excursion (baseline drift, a load change, a "
      "knock). The **terminal crossing** is the start of the final unbroken run above the "
      "threshold that reaches failure - the one that corresponds to irreversible degradation. "
      "The gap between the two columns is the false-alarm exposure an operator would live with.")
    a("")
    a("| tier | terminal crossing | terminal lead time | driver | earlier recovering episodes | time spent in them |")
    a("|---|---|---|---|---|---|")
    for c in lead.crossings:
        if not c.crossed:
            a(f"| **{c.tier}** | never | - | - | - | - |")
            continue
        if c.terminal_index is None:
            a(f"| **{c.tier}** | score recovers before failure | - | - | {c.prior_episodes} | "
              f"{c.prior_episode_hours:,.2f} h |")
            continue
        a(f"| **{c.tier}** | acq {c.terminal_index} @ {c.terminal_elapsed_hours:,.2f} h | "
          f"{_fmt_hours(c.terminal_lead_time_hours)} | `{c.terminal_driving_channel}` | "
          f"{c.prior_episodes} | {c.prior_episode_hours:,.2f} h |")
    a("")
    a(f"- Peak anomaly score: **{lead.peak_score:.3f}**, reached "
      f"{_fmt_hours(lead.peak_score_hours_to_failure)} before failure.")
    a("")
    if lead.ground_truth is not None:
        g = lead.ground_truth
        a("## Against the known damage onset (surrogate only)")
        a("")
        a(f"- Modelled spall initiates at acquisition **{g.onset_index}** "
          f"(t = {g.onset_elapsed_hours:,.2f} h).")
        a(f"- Terminal alert fires **{_fmt_hours(g.detection_delay_hours)}** after onset."
          if g.detection_delay_hours is not None else
          "- Terminal alert never establishes before failure.")
        a(f"- Time spent at or above the alert threshold *before* any damage exists "
          f"(false-alarm exposure): **{g.false_alarm_hours_before_onset:,.2f} h**.")
        a("")
    a("## Tier distribution over the whole run")
    a("")
    a("| tier | share of acquisitions |")
    a("|---|---|")
    for k, v in lead.tier_shares.items():
        a(f"| {k} | {100 * v:.2f}% |")
    a("")
    return "\n".join(lines)


def plot_run(scored: pd.DataFrame, lead: LeadTimeReport, path: str | Path,
             title: str = "Vayeron control limits on a run-to-failure record") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = scored["elapsed_hours"].to_numpy(dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    ax = axes[0]
    if "rms_norm" in scored.columns:
        ax.plot(t, scored["rms_norm"], lw=1.1, label="rms_norm")
    else:
        ax.plot(t, scored["rms"], lw=1.1, label="rms")
    ax.set_ylabel("vibration")
    ax2 = ax.twinx()
    for c, style in (("temp1", "-"), ("temp2", "--")):
        if c in scored.columns:
            ax2.plot(t, scored[c], style, lw=0.9, alpha=0.7, color="tab:red", label=c)
    ax2.set_ylabel("temperature (C)", color="tab:red")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    for ch in ("bpfo", "bpfi", "bsf", "ftf"):
        col = f"severity_{ch}"
        if col in scored.columns:
            ax.plot(t, scored[col], lw=1.0, label=ch)
    if "severity_rms" in scored.columns:
        ax.plot(t, scored["severity_rms"], lw=1.4, color="k", label="rms")
    ax.axhline(1.0, color="crimson", ls=":", lw=1.2, label="alert control limit")
    ax.set_ylabel("severity  s = z / z_alert")
    ax.legend(ncol=5, fontsize=8)

    ax = axes[2]
    ax.plot(t, scored["anomaly_score"], lw=1.6, color="tab:blue")
    for thr, name, color in zip(TIER_EDGES, ("watch", "alert", "critical"),
                                ("goldenrod", "darkorange", "crimson")):
        ax.axhline(thr, color=color, ls="--", lw=1.0)
        ax.text(t[0], thr + 0.012, name, color=color, fontsize=8)
    for c in lead.crossings:
        if c.crossed:
            ax.axvline(c.elapsed_hours, color="grey", ls=":", lw=1.0)
    ax.axvline(scored["elapsed_hours"].iloc[lead.failure_index], color="black", lw=1.6)
    ax.text(scored["elapsed_hours"].iloc[lead.failure_index], 0.05, " failure",
            fontsize=9, ha="right", rotation=90, va="bottom")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("anomaly score Y")
    ax.set_xlabel("elapsed test time (h)")

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def write_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path
