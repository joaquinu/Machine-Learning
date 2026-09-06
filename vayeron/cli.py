"""Apply the Vayeron statistical control limits to a run-to-failure record.

Examples
--------
Real Mendeley/KAIST archive already downloaded on the machine::

    python -m vayeron.cli --data /path/to/mendeley_bearing_run_to_failure \
        --fs 25600 --rpm 1770 --preset mendeley --out reports/mendeley

Surrogate run (no download needed), to verify the pipeline end to end::

    python -m vayeron.cli --synthetic --out reports/surrogate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .control_limits import ScoringConfig, VAYERON_ALERT_LIMITS, score_run
from .features import VAYERON_DEFECT_ORDERS, geometric_defect_orders
from .mendeley import LoadConfig, build_feature_table, failure_index
from .report import (attach_ground_truth, lead_time_analysis, plot_run,
                     render_markdown, write_json)

# The Vayeron conveyor gate (350-1500 rpm) excludes a 1770 rpm lab rig outright.
# The preset widens the gate to the rig's own envelope and leaves every
# statistical parameter of KX-VAY-012 untouched.
PRESETS = {
    "vayeron": dict(rpm_gate=(350.0, 1500.0), rpm_reference=600.0),
    "mendeley": dict(rpm_gate=(1500.0, 2000.0), rpm_reference=1770.0),
    "none": dict(rpm_gate=None, rpm_reference=600.0),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("input")
    src.add_argument("--data", type=Path,
                     help="directory of acquisition files, a long waveform file, "
                          "or an already-reduced feature CSV")
    src.add_argument("--synthetic", action="store_true",
                     help="generate and score a run-to-failure surrogate instead")
    src.add_argument("--features-csv", type=Path,
                     help="skip extraction and score this feature table directly")

    acq = p.add_argument_group("acquisition")
    acq.add_argument("--fs", type=float, default=25600.0, help="sample rate, Hz")
    acq.add_argument("--rpm", type=float, default=1770.0, help="shaft speed, rpm")
    acq.add_argument("--interval", type=float, default=600.0,
                     help="seconds between acquisitions when files are not timestamped")
    acq.add_argument("--snapshot-seconds", type=float, default=None,
                     help="chop a single long record into snapshots of this length")
    acq.add_argument("--vibration-column", default=None,
                     help="column name or index holding the accelerometer channel")
    acq.add_argument("--max-snapshots", type=int, default=None)
    acq.add_argument("--band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                     help="envelope demodulation band, Hz")

    geo = p.add_argument_group("defect frequencies")
    geo.add_argument("--orders", choices=("vayeron", "geometry"), default="vayeron",
                     help="Vayeron's Factor/10 multipliers, or compute from bearing geometry")
    geo.add_argument("--geometry", type=float, nargs=4, default=None,
                     metavar=("NBALLS", "BALL_D", "PITCH_D", "CONTACT_DEG"))
    geo.add_argument("--sidebands", action="store_true",
                     help="recombine the +/-1 modulation sidebands into each defect ratio. "
                          "Tested on the surrogate campaign: no attribution benefit "
                          "(2 fixed / 2 broken, p=1.0). Kept for checking against real data, "
                          "where the sideband structure is physical rather than modelled.")

    sc = p.add_argument_group("scoring")
    sc.add_argument("--preset", choices=tuple(PRESETS), default="mendeley")
    sc.add_argument("--healthy-ratio", type=float, default=0.10)
    sc.add_argument("--healthy-hours", type=float, default=None)
    sc.add_argument("--ema-tau-hours", type=float, default=2.0)
    sc.add_argument("--ema-mode", choices=("span", "time"), default="span")
    sc.add_argument("--persistence", type=int, default=3,
                    help="consecutive acquisitions required to count a tier crossing")
    sc.add_argument("--rms-limit", type=float, default=VAYERON_ALERT_LIMITS["rms"])
    sc.add_argument("--spectral-limit", type=float, default=VAYERON_ALERT_LIMITS["bpfo"])
    sc.add_argument("--no-rpm-normalisation", action="store_true")
    sc.add_argument("--min-mad-fraction", type=float, default=0.0,
                    help="minimum resolvable change as a fraction of the baseline median; "
                         "guards against a lab rig's near-zero healthy dispersion")
    sc.add_argument("--prefilter-median", type=int, default=0,
                    help="median prefilter width on raw severity, in samples; suppresses "
                         "single-sample knocks that the EMA would otherwise smear into a "
                         "multi-sample alert episode (0 = KX-VAY-012 as written)")
    sc.add_argument("--attribution", choices=("max_severity", "spectral_priority"),
                    default="max_severity",
                    help="how fault_channel picks a driver. 'spectral_priority' prefers a "
                         "defect line over broadband rms whenever one is above its own limit; "
                         "on the surrogate it fixed 5 modulated-fault attributions and broke "
                         "none (p=0.0625).")
    sc.add_argument("--thermal", choices=("none", "asymmetry", "all"), default="none",
                    help="fold thermal channels into Y. 'asymmetry' adds |T1-T2| only and is "
                         "ambient-safe; 'all' also adds the hotter raceway, which chased a "
                         "seasonal swing to a 71%% alert rate on a field-shaped record and is "
                         "not recommended.")
    sc.add_argument("--aggregation", choices=("max", "kth_max"), default="max",
                    help="'max' is the KX-VAY-012 spec; 'kth_max' requires k channels to be "
                         "excursing together, which resists single-channel baseline drift")
    sc.add_argument("--aggregation-k", type=int, default=2)
    sc.add_argument("--sensitivity", action="store_true",
                    help="sweep healthy window, EMA tau, persistence and MAD floor")

    fail = p.add_argument_group("failure reference")
    fail.add_argument("--failure-temp-c", type=float, default=85.0)
    fail.add_argument("--failure-rms", type=float, default=9.0)
    fail.add_argument("--failure-index", type=int, default=None,
                      help="override the detected failure acquisition")

    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, default=Path("reports/run"))
    out.add_argument("--title", default=None)
    out.add_argument("--no-plot", action="store_true")
    return p


def resolve_orders(args) -> dict[str, float]:
    if args.orders == "geometry":
        if not args.geometry:
            raise SystemExit("--orders geometry requires --geometry NBALLS BALL_D PITCH_D CONTACT_DEG")
        n, bd, pd_, ang = args.geometry
        return geometric_defect_orders(int(n), bd, pd_, ang)
    return dict(VAYERON_DEFECT_ORDERS)


def load_features(args) -> tuple[pd.DataFrame, str]:
    if args.features_csv:
        df = pd.read_csv(args.features_csv)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, f"Feature table: `{args.features_csv}`"

    if args.synthetic:
        from .synthetic import SurrogateConfig, generate_run
        from .features import FeatureConfig, waveform_features

        scfg = SurrogateConfig(fs=args.fs, rpm=args.rpm,
                               acquisition_interval_s=args.interval)
        meta, waves = generate_run(scfg)
        fcfg = FeatureConfig(fs=scfg.fs, rpm=scfg.rpm, defect_orders=resolve_orders(args),
                             band_hz=tuple(args.band) if args.band else None,
                             use_sidebands=args.sidebands)
        rows = []
        for (_, row), wave in zip(meta.iterrows(), waves):
            rec = row.to_dict()
            rec.update(waveform_features(wave, fcfg))
            rows.append(rec)
        note = ("Source: **modelled run-to-failure surrogate** "
                f"({scfg.n_acquisitions} acquisitions, {scfg.rpm:.0f} rpm, "
                f"{scfg.fs / 1000:.1f} kHz, outer-race spall from "
                f"{100 * scfg.onset_fraction:.0f}% of life). Not physical data.")
        return pd.DataFrame(rows), note

    if not args.data:
        raise SystemExit("one of --data, --features-csv or --synthetic is required")

    lcfg = LoadConfig(
        fs=args.fs,
        rpm=args.rpm,
        acquisition_interval_s=args.interval,
        snapshot_seconds=args.snapshot_seconds,
        vibration_column=args.vibration_column,
        max_snapshots=args.max_snapshots,
        defect_orders=resolve_orders(args),
        feature_overrides={**({"band_hz": tuple(args.band)} if args.band else {}),
                           "use_sidebands": args.sidebands},
    )
    df = build_feature_table(args.data, lcfg, progress=True)
    return df, f"Source: `{args.data}` ({len(df)} acquisitions, {args.rpm:.0f} rpm)."


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    features, source_note = load_features(args)

    limits = {
        "rms": args.rms_limit,
        "bpfo": args.spectral_limit,
        "bpfi": args.spectral_limit,
        "bsf": args.spectral_limit,
        "ftf": args.spectral_limit,
    }
    cfg = ScoringConfig(
        alert_limits=limits,
        healthy_window_ratio=args.healthy_ratio,
        healthy_window_hours=args.healthy_hours,
        ema_tau_hours=args.ema_tau_hours,
        ema_mode=args.ema_mode,
        persistence_samples=args.persistence,
        normalize_rms_by_rpm=not args.no_rpm_normalisation,
        min_mad_fraction_of_median=args.min_mad_fraction,
        prefilter_median_samples=args.prefilter_median,
        attribution=args.attribution,
        thermal_channels={"none": (), "asymmetry": ("temp_asymmetry",),
                          "all": ("temp_max", "temp_asymmetry")}[args.thermal],
        aggregation=args.aggregation,
        aggregation_k=args.aggregation_k,
        **PRESETS[args.preset],
    )

    scored, scoring = score_run(features, cfg)

    fail_idx = args.failure_index if args.failure_index is not None else failure_index(
        scored, temp_limit_c=args.failure_temp_c, rms_limit=args.failure_rms)
    lead = attach_ground_truth(scored, lead_time_analysis(scored, fail_idx, cfg))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "scored.csv", index=False)
    write_json({"scoring": scoring.as_dict(), "lead_time": lead.as_dict(),
                "config": {k: str(v) for k, v in vars(cfg).items()}},
               out / "summary.json")
    title = args.title or "Vayeron control limits applied to a run-to-failure record"
    md = render_markdown(scored, scoring, lead, cfg, title, source_note)

    if args.sensitivity:
        from . import sensitivity as sens

        sweep = sens.sweep(features, cfg, fail_idx)
        sweep.to_csv(out / "sensitivity.csv", index=False)
        summary = sens.summarise(sweep)
        write_json(summary, out / "sensitivity_summary.json")
        md += "\n" + sens.render_markdown(sweep, summary)

    (out / "report.md").write_text(md)
    if not args.no_plot:
        plot_run(scored, lead, out / "run_to_failure.png", title)

    print(md)
    print(f"\nWritten to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
