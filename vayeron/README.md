# Vayeron control limits on external run-to-failure data

Applies the KX-VAY-012 statistical anomaly target — **3.5 sigma on overall
vibration, 4.0 sigma on the spectral defect ratios** — to a run-to-failure
record, and measures how much warning it gives before the bearing physically
seizes.

The Vayeron field dataset has no run-to-failure ground truth, so `anomaly_score`
is a statistical proxy: models trained on it learn to predict an upstream
equation, not a confirmed mechanical failure. Scoring an external record that
*does* end in destruction is what turns the proxy into a validated one.

## Layout

| file | what it does |
|---|---|
| `control_limits.py` | the KX-VAY-012 target: robust z-score → severity vs limit → EMA → logistic → tiers |
| `features.py` | Vayeron-equivalent edge features (RMS + Goertzel-style envelope ratios) from a raw waveform |
| `mendeley.py` | loader for the Mendeley / KAIST ball-bearing run-to-failure archive |
| `synthetic.py` | modelled run-to-failure surrogate, for verifying the pipeline without the archive |
| `sensitivity.py` | sweeps the analyst's free choices so a lead time is not a one-configuration artefact |
| `report.py` | lead-time analysis, markdown report, diagnostic plot |
| `campaign.py` | multi-seed x multi-fault-mode sweep over the surrogate |
| `label_audit.py` | how much of an existing `alert` class is EMA-smeared knocks |
| `cli.py` | `python -m vayeron.cli` |
| `tests/` | 26 unit tests, `python -m pytest vayeron/tests` |

## Running it against the real archive

The Mendeley dataset is not bundled — download it to the machine first
(`data.mendeley.com` is blocked from Claude Code's remote sandbox, so this step
has to happen on your WSL box):

```bash
python -m vayeron.cli \
    --data /mnt/c/data/mendeley_bearing_run_to_failure \
    --preset mendeley --fs 25600 --rpm 1770 \
    --sensitivity --prefilter-median 3 \
    --out reports/mendeley
```

Outputs land in `--out`: `scored.csv` (every intermediate quantity per
acquisition), `report.md`, `summary.json`, `sensitivity.csv`, and
`run_to_failure.png`.

The loader is format-tolerant — a directory of per-acquisition waveform files
(`.csv`/`.txt`/`.mat`/`.npy`), one long waveform to chop with
`--snapshot-seconds`, or an already-reduced feature table. Pin anything the
heuristics get wrong:

```bash
--vibration-column ch1 --interval 600 --band 1000 10000
```

Verify the pipeline with no data at all:

```bash
python -m vayeron.cli --synthetic --healthy-ratio 0.15 --sensitivity --out reports/surrogate
python -m vayeron.campaign     # 5 fault modes x 6 seeds x 2 variants
```

## Auditing the field labels

Whether the field dataset's `alert` class contains EMA-smeared knocks is one
command against the raw telemetry (not `anomaly_score`, which is recomputed):

```bash
python -m vayeron.label_audit --csv vayeron_anomaly_dataset.csv --group-by roller_id \
    --out reports/label_audit
```

The threshold is sharp — `survival_threshold()` gives it in closed form, ~22
sigma on RMS at the field's 626 s cadence — so the answer is either "none" or
"a large fraction", with little in between.

## Three things the external data forces you to decide

**1. The rpm quality gate excludes the rig.** KX-VAY-012 gates on
`350 <= rpm <= 1500`; the KAIST rig runs at 1770 rpm, so the Vayeron gate
discards every row. `--preset mendeley` widens the gate to `1500-2000` and
leaves every statistical parameter untouched. `score_run` raises rather than
returning an all-zero column when a gate rejects everything.

**2. Defect frequencies are the Smart-Idler's, not the test bearing's.**
`f_defect = (rpm/60) * (Factor/10)` with BSF 2.1, BPFO 3.1, FTF 4.1, BPFI 4.9
describes Vayeron's own bearing. Keeping them (`--orders vayeron`, the default)
applies Vayeron's definition literally; if the test bearing's geometry is known,
`--orders geometry --geometry NBALLS BALL_D PITCH_D CONTACT_DEG` computes the
true orders instead. Comparing the two is itself informative: a fault the
Vayeron multipliers miss is a fault the fielded firmware would also miss.

**3. RMS_norm is a no-op at constant speed.** `RMS / (rpm/600)` is a constant
rescale on a fixed-speed rig, and the robust z-score is scale-invariant, so it
changes nothing. It stays in the path so the same code serves variable-speed
field data.

## Departures from KX-VAY-012, all opt-in

Defaults reproduce the delivered specification exactly. Each of these is off
unless asked for:

| flag | what it changes | why |
|---|---|---|
| `--prefilter-median N` | median filter on raw severity before the EMA | a single-sample knock survives the 2 h EMA — the smoother spreads it over ~span samples, so it clears a "3 consecutive samples" persistence rule and registers as a genuine alert episode |
| `--min-mad-fraction F` | floors the baseline MAD at `F x median` | a lab rig's healthy phase can be stationary to a fraction of a percent, which makes 3.5 sigma trip on a ~1% change |
| `--aggregation kth_max` | the k-th highest channel drives severity, not the highest | section 6.2 calls a real anomaly a *coherent multi-channel* excursion; section 7 aggregates with `max`, which one drifting channel can carry alone |
| `--attribution spectral_priority` | a defect line above its own limit outranks broadband `rms` for `fault_channel` | `rms` baseline dispersion is 4-7x tighter than the spectral channels', so a plain argmax over severity favours it structurally. Fixed 5 modulated-fault attributions and broke none |
| `--thermal asymmetry` | folds \|T1-T2\| into Y | the datasheet has a Temp Alert byte and section 6.2 calls a >4 C divergence an anomaly; Y as delivered has no thermal term at all. Ambient-invariant, unlike absolute temperature |
| `--sidebands` | recombines the +/-1 modulation sidebands into each defect ratio | tested and gave no attribution benefit (2 fixed / 2 broken). Kept for checking against real data |
| `--ema-mode time` | `alpha = 1 - exp(-dt/tau)` instead of a fixed span | correct when the acquisition cadence is irregular; a warning fires when it is |
