# Proposed changes to the Vayeron anomaly target

**Document** KX-VAY-015 (draft) · **Supersedes nothing** · **Amends** KX-VAY-012 §7, §9
**Classification** Internal — Kaspix & Vayeron

---

## Purpose

KX-VAY-012 defines the anomaly target `Y ∈ [0, 1]` and states plainly that it is
a statistical proxy: the site supplied no run-to-failure record, so the models
in KX-VAY-013 learn to predict an upstream equation rather than a confirmed
mechanical failure. Validating that proxy against an external run-to-destruction
dataset was the plan in the validation strategy note.

Building the pipeline to do that surfaced six things worth changing in the target
itself, and four candidate changes that were tested and rejected. This document
is the changelist: what to change, the evidence for it, what it costs, and how
far the evidence actually reaches.

## Status of the evidence

**The Mendeley/KAIST archive has not been scored yet.** It could not be
downloaded in the environment this work was done in. Every measurement below
therefore comes from one of two modelled records, and the distinction between
them matters:

| source | what it is | what it can support |
|---|---|---|
| run-to-failure surrogate | 5 failure modes × 6 seeds, constant-speed rig, 240 acquisitions to seizure | claims about *how the scoring behaves* on a degradation curve |
| field-shaped surrogate | 81,600 rows, 4 rollers, 626 s cadence, the deployment's seasonal swing, one 11-day multi-channel event | claims about *false-alarm exposure* under field conditions |

Neither supports a claim about how a physical bearing degrades. Changes 1–5 are
about the mechanics of the target and are testable on modelled data. Change 6
is not, and is flagged accordingly.

---

## Change 1 — the rpm gate must fail loudly, not silently

**Now.** §9 gates on `350 ≤ rpm ≤ 1500`, the conveyor envelope. Rows outside it
exit with `Y = 0`.

**Problem.** The KAIST rig runs at 1770 rpm. Applied literally, the gate assigns
`Y = 0` to *every row of the dataset* and reports no anomalies — a complete false
negative that looks exactly like a clean bearing. Any external validation record
will sit outside an envelope fitted to one conveyor.

**Change.** Keep the gate; make the envelope a parameter, and raise an error
when it rejects an entire record rather than returning a column of zeros.

**Cost.** None. Covered by a test.

---

## Change 2 — median-prefilter the raw severity before the EMA

**Now.** §7 step 3 smooths severity with a ~2 h EMA. §6.1 item 6 says an isolated
spike is not an anomaly, and temporal persistence is what should enforce that.

**Problem.** The EMA sits *before* the persistence test, so it spreads a lone
spike across roughly its span. The smoothed severity then stays above the limit
for several consecutive samples and clears any "N consecutive samples" rule. The
persistence test cannot see a spike the smoother has already turned into a
plateau.

This has a sharp threshold. A lone spike of severity `S` leaves the smoother at
`α·S` with `α = 2/(span+1)`, so it crosses the limit only above `S = 1/α`, and
sustains a 3-sample episode only above ≈10.6. At the field cadence of 626 s with
τ = 2 h:

| | severity | equivalent on RMS |
|---|---|---|
| spike crosses the alert limit | 6.25 | ≈22 σ |
| spike sustains a 3-sample episode | 10.6 | ≈37 σ |

That bound assumes a spike against a near-zero baseline. Real severity wanders,
and a knock landing on already-elevated ground needs far less. Swept against the
field-shaped surrogate — tuned so its scored tier distribution matches the real
export at 88.07 / 5.82 / 6.11 — the contamination is a gradient, not a cliff:

| knock size (baseline MADs) | 6 | 12 | 20 | 30 | 45 | 90 |
|---|---|---|---|---|---|---|
| share of the alert class it manufactures | 0.8% | 1.5% | 3.0% | 6.9% | 13.8% | 21.9% |

There is no size below which knocks are harmless: even 6 MADs takes 0.8% of the
alert class, because some of those knocks land on stretches where severity is
already elevated. The closed form is an upper bound on what a knock needs in
the quiet case, not a safety threshold.

**Change.** Apply a 3-sample median filter to raw severity ahead of the EMA.

**Cost.** Measured across 6 seeds of the run-to-failure surrogate: pre-onset time
above the alert threshold falls from **3.08 h median to 0.58 h**, and terminal
alert lead time is unchanged at 11.0 h. An 80% cut in false-alarm exposure for no
measurable lead time. It is a reduction, not a cure — three of six seeds still
spend over an hour above the threshold before any damage exists.

**Open.** Whether the field record contains knocks above ~22 σ at all is
unmeasured; the audit that answers it is one command against the raw telemetry.
If it does, the KX-VAY-013 benchmark numbers are partly measuring agreement with
smeared knocks and the models should be retrained after relabelling. If it does
not, the prefilter is cheap insurance and one open question closes.

---

## Change 3 — report the terminal crossing, not the first

**Now.** Lead time is naturally read as "when did `Y` first cross 0.50".

**Problem.** That number is dominated by recoverable excursions — drift, a load
change, a knock — and is not a property of the degradation. Across 108
configurations of the analyst's free choices (healthy window × EMA τ ×
persistence × MAD floor), first-crossing lead varied **9.3–32.8 h**; the
terminal crossing — the start of the final unbroken run above the threshold that
reaches failure — varied **9.3–11.2 h**. One is a measurement, the other is
noise.

**Change.** Report both, and treat the terminal crossing as the lead time. The
gap between them is the operator's false-alarm exposure and is worth reporting
in its own right.

**Cost.** None; it is a reporting change.

---

## Change 4 — attribute to a defect line ahead of broadband RMS

**Now.** §7 attributes `fault_channel` by `argmax` over per-channel severity.

**Problem.** The five channels are not comparable after normalisation. Measured
baseline relative dispersion on the surrogate:

| channel | MAD / median | alert limit |
|---|---|---|
| `rms` | **3.6%** | 3.5 |
| `bpfi` | 15.1% | 4.0 |
| `ftf` | 15.8% | 4.0 |
| `bsf` | 24.5% | 4.0 |
| `bpfo` | 25.6% | 4.0 |

RMS is four to seven times tighter, so the same relative rise buys it four to
seven times more sigma, and `max_c(z_c / z_alert,c)` is biased toward it by
construction. Compounding that, an amplitude-modulated defect — inner race
(load-zone passage, once per shaft revolution) or ball spin (cage rate) — spreads
its energy into sidebands and leads its own channel by only ~1.5×, against ~3.0×
for an unmodulated defect. The result: attribution is correct 6/6 for outer-race
and cage faults, and **1–2 / 6** for the modulated ones, which report `rms`.

The alert still fires and the fitter is still sent. They are told "general
vibration" when the defect is on the inner race.

**Change.** If any spectral channel is above its own control limit, attribute to
the highest of those; fall back to `rms` only when none is. This is how an
analyst reads a spectrum — RMS says something is wrong, a defect line says what.

**Cost.** On the modulated modes, attribution goes from 6/24 to **12/24**. The
paired comparison is the reassuring part: **5 cases fixed, 0 broken** (McNemar
exact, p = 0.0625). It never regresses a case it already got right, and costs
nothing in lead time. It does cost one case on the thermal mode, correctly —
there the true driver is broadband.

**Open.** Still a coin flip on the faults that matter. Four fixes have now been
tried and three failed (below), which suggests the residue is structural at this
window and channel set rather than a tuning gap — the same shape as KX-VAY-013's
finding that three independent architectures plateau at one F1 ceiling. Whether
it holds on a real inner-race fault decides whether this needs Vayeron in the
room or is good enough to ship.

---

## Change 5 — give `Y` a thermal term: asymmetry yes, absolute no

**Now.** `Y` consumes five vibration channels and no thermal one. `temp1` and
`temp2` reach the models as input features but never as evidence in the label —
although the Smart-Idler datasheet specifies a dedicated **Temp Alert** byte with
independent left/right monitoring, and §6.2 calls a >4 °C end-to-end divergence
an anomaly in its own right.

**Problem.** A grease dry-out or seal failure that runs hot without a vibration
signature scores `normal` all the way to seizure.

**Change.** Add `temp_asymmetry = |T₁ − T₂|` as a scored channel, carrying §6.2's
absolute threshold alongside the statistical one, so a 4 °C divergence sits at
severity 1.0 however tight the fitted baseline is.

**Do not add absolute temperature.** §6.1 item 3 already rules out seasonal
drift — "both bearing ends cool equally" — and an absolute-temperature channel
cannot tell that from a fault. Scored against the field-shaped surrogate carrying
the deployment's real 43.6 → 34.4 °C swing:

| baseline fitted | as delivered | + asymmetry | + absolute temperature |
|---|---|---|---|
| first 10%, warm start | 1.54% alert | 1.54% | 1.54% |
| coolest 10%, commissioned in winter | 0.32% alert | 0.34% | **71.2%** |

A sensor commissioned in a cool period would spend most of its life flagged, with
60,483 rows attributed to the temperature channel. Absolute temperature is the
wrong quantity.

**Cost.** Asymmetry moved the adverse-baseline alert rate by 0.02 percentage
points. No lead-time benefit is claimed: the surrogate always couples heat to
broadband noise, so the hot-but-quiet failure this channel exists for is the one
it cannot represent. The case rests on the datasheet and §6.2, not on a measured
gain. What the surrogate does confirm is that the channel is well-behaved — it
never claimed attribution on a spall mode, and claimed it on 2 of 6 thermal runs.

---

## Change 6 — re-derive the tier thresholds against a real failure axis

**Now.** 0.35 / 0.50 / 0.80 produce an 88 / 6 / 6 split on the field record.

**Problem.** Those thresholds were set to make the distribution come out that
way. They were not set to buy a chosen amount of warning, because without a
run-to-failure record there was no time-to-failure axis to set them against.

**Change.** Once the archive is scored, re-derive them from lead time: choose the
alert threshold that buys the inspection window the mine actually needs, and let
the class balance fall where it falls.

**Status.** Cannot be done on modelled data — the answer would be a property of
the assumed degradation law. Flagged, not recommended, pending the archive.

---

## Tested and rejected

Recorded so they are not re-attempted.

| candidate | result |
|---|---|
| **Sideband-aware ratios** — recombine energy at `f_defect ± shaft rate` (BPFI) and `± cage rate` (BSF) | The ±1 sidebands do carry real energy (~1.9× floor against a 2.5× centre) and ±2 carry none, so the family was capped at ±1 and summed. No attribution benefit: **2 fixed, 2 broken, p = 1.0**, at a cost of ~0.3 h lead time. Retained behind a flag only because on real data the sideband structure is physical rather than modelled. |
| **Attribute from smoothed severity** rather than instantaneous | No effect whatever. The ordering problem is dispersion, not noise. Removed. |
| **Attribute at the first crossing** rather than the terminal one | Worse everywhere — 1/6 for outer race against 6/6 — because early crossings are frequently drift or knock episodes, themselves `rms`-driven. |
| **Focal loss / oversampling** for the class imbalance (KX-VAY-013) | Already rejected there; noted for completeness. Plain MSE remains correct. |
| **Minority-class overlapping windows** — slide the window where the label is scarce, recovering the samples non-overlapping windows discard | Works, but not in the direction wanted. Training windows for `watch` and `alert` grow ~7.5x and macro-F1 rises +0.057, entirely by buying precision with recall: the alert class gains 8.6 points of precision and loses 6.2 points of recall. A missed alert is a seized roller; a false one is an inspection. The tier thresholds already trade those off more directly. Implemented in `benchmarks.py`, off by default. |

On the augmentation result, one methodological note worth keeping: generating
overlapping windows *before* splitting fills the test set with shifted copies of
training data. Measured here that mistake was worth only +0.003 macro-F1 — much
less than expected, plausibly because the model is tiny and the surrogate's
episodes are long — but it is free to avoid, so `benchmarks.py` splits first and
admits an augmented window only when every reading it covers already belongs to
a training window. A test asserts it.

Available but **not** campaign-tested: `kth_max` aggregation, which requires two
channels to excurse together per §6.2's "coherent multi-channel" definition. It
has unit coverage only. Do not cite it as evidenced.

---

## What the surrogate does and does not establish

**Establishes** — detection held in **180 of 180** scored runs across five failure
modes, six seeds and six scoring configurations, at a median terminal lead of
11.0 h, **27.8% of life**. The detection instant is stable across every analyst
choice swept; only the false-alarm exposure moves.

**Does not establish** — anything about a physical bearing. Absolute hours do not
transfer from an accelerated test; the fractional lead may. Ball-spin detection
is model-limited (the modelled cage modulation leaves ≤3.3× peak-to-floor at any
sideband, so there is little for any detector to find) and should not be cited.
The thermal channel's value is untestable here by construction.

## Next action

Score the archive. One command, and Changes 1–5 move from "behaves correctly on
modelled data" to "validated", while Change 6 becomes answerable for the first
time.

```
python -m vayeron.cli --data <archive> --preset mendeley --fs 25600 --rpm 1770 \
    --prefilter-median 3 --attribution spectral_priority --thermal asymmetry \
    --sensitivity --out reports/mendeley
python -m vayeron.label_audit --csv vayeron_anomaly_dataset.csv --group-by roller_id
```
