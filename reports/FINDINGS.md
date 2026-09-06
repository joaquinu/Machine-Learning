# Applying the Vayeron control limits to run-to-failure data

**Status of the Mendeley data.** `data.mendeley.com` is blocked by this
session's egress policy (the proxy answered `403` to `CONNECT`), so the KAIST
ball-bearing archive could not be downloaded here. Everything below is the
complete, tested pipeline plus an end-to-end validation run against a
**modelled** run-to-failure surrogate. Point it at your local copy of the
archive on WSL and the same command produces the real numbers:

```bash
python -m vayeron.cli --data /mnt/c/data/mendeley_bearing_run_to_failure \
    --preset mendeley --fs 25600 --rpm 1770 --sensitivity --prefilter-median 3 \
    --out reports/mendeley
```

The surrogate result tells you whether the *method* works. It cannot tell you
whether a physical bearing gives Vayeron days of warning — only the real archive
can, and every number in the surrogate column below should be treated as a
property of the degradation law that was modelled, not of a bearing.

---

## 1. What had to be built before the limits could be applied at all

The Vayeron target consumes five channels: `rms` plus the four Goertzel envelope
ratios. External run-to-failure archives ship **raw accelerometer waveforms**,
so those five channels have to be reconstructed first, or the control limits have
nothing to act on.

`vayeron/features.py` does this with Hilbert envelope demodulation — the textbook
equivalent of the firmware's Goertzel power filter — reproducing the datasheet's
own definition:

```
f_defect = (rpm/60) * (Factor/10)
ratio    = peak amplitude at f_defect / noise-floor amplitude in adjacent bins
```

Verified against a synthetic outer-race impulse train: the BPFO ratio comes back
at **38.5** against 1.2–2.2 on the three untouched channels.

## 2. Three decisions the external data forces, which the field pipeline never had to make

**The rpm gate excludes the rig.** KX-VAY-012 section 9 gates on
`350 <= rpm <= 1500`. The KAIST rig runs at 1770 rpm, so the Vayeron pipeline
applied literally **discards 100% of the dataset** and returns `Y = 0` for every
row — a silent, total false negative. `score_run` now raises instead of
returning an all-zero column, and `--preset mendeley` widens the gate to
1500–2000 rpm while leaving every statistical parameter untouched.

**The defect multipliers belong to the wrong bearing.** BSF 2.1 / BPFO 3.1 /
FTF 4.1 / BPFI 4.9 describe the Smart-Idler's bearing. For a 6205-class test
bearing, geometry gives BPFO ≈ 3.59 and BPFI ≈ 5.42 — a 14% and 10% offset,
comfortably outside the ±2% peak-search window, so a real outer-race fault would
land in the noise band and read as *nothing*. Both are available
(`--orders vayeron | geometry`), and running the pair is itself the useful
experiment: a fault the Vayeron multipliers miss is a fault the fielded firmware
also misses.

**`RMS_norm` is a no-op here.** `RMS/(rpm/600)` is a constant rescale at fixed
speed and the robust z-score is scale-invariant, so it changes no result. It
stays in the path for variable-speed field data. Covered by a test.

## 3. Two defects in the target construction, surfaced by a degradation curve

Both were invisible on field data because the field record has one anomaly
episode and no failure to calibrate against.

**A single-sample knock can produce a multi-sample alert — above a threshold I
can now state exactly.** Section 6.1 item 6 says an isolated spike is not an
anomaly, and the persistence rule is meant to enforce that. It cannot, because
the EMA sits *before* it: the smoother spreads a lone spike across roughly its
span, so the smoothed severity can stay above the limit for several consecutive
samples and clear any "N consecutive samples" test.

Whether that actually happens depends on how big the spike is, and the
arithmetic is closed-form. A lone spike of severity `S` against a near-zero
baseline leaves the smoother at `alpha * S`, with `alpha = 2/(span+1)`. It
crosses the alert limit only if `S >= 1/alpha`, and the resulting episode lasts
`ln(alpha*S) / -ln(1-alpha)` samples. At the field cadence of 626 s with
tau = 2 h (span 11.5, alpha 0.16):

| | severity | equivalent on RMS |
|---|---|---|
| spike crosses the alert limit at all | 6.25 | ~22 sigma |
| spike sustains a 3-sample episode | 10.6 | ~37 sigma |

Measured against a field-shaped surrogate (81,600 rows, 4 rollers, 626 s
cadence, 2.8% transients, one sustained 11-day multi-channel event), sweeping
the knock magnitude reproduces exactly that threshold:

| knock size (baseline MADs) | share of the alert class it manufactures |
|---|---|
| 6 | 0.0% |
| 12 | 0.0% |
| 20 | 6.7% |
| 30 | 41.0% |
| 45 | 51.9% |
| 90 | 62.8% |

Nothing below ~12 MADs; onset around 20; catastrophic by 30. **I do not know
which side the real field knocks fall on** — that needs the actual CSV, and
`python -m vayeron.label_audit --csv vayeron_anomaly_dataset.csv --group-by
roller_id` answers it in one run. What can be said now is that the mechanism is
real, the threshold is sharp, and it is cheap to check.

The 3-sample median prefilter removes the mechanism regardless of knock size.
On the run-to-failure surrogate, across 6 seeds:

| | pre-onset time above the alert threshold | terminal alert lead time |
|---|---|---|
| KX-VAY-012 as written | 3.08 h median, 0.00 – 5.33 h | 11.0 h median |
| `--prefilter-median 3` | **0.58 h median, 0.00 – 3.17 h** | 11.0 h median |

An 80% cut in false-alarm exposure at no measurable cost in lead time. It is a
reduction, not a cure: three of six seeds still spend over an hour above the alert
threshold before any damage exists.

**`max` over channels lets one drifting channel carry the score.** Section 7
aggregates severity with `max`; section 6.2 defines a real anomaly as a
*coherent multi-channel* excursion. Those disagree, and `max` is the weaker of
the two: a slow load drift on `rms` alone drove the surrogate's first (spurious)
alert. `--aggregation kth_max --aggregation-k 2` requires two channels to excurse
together. Off by default — it is a change to the delivered specification, not a
bug fix.

## 4. First crossing is the wrong number to report

The obvious metric — when does `Y` first cross 0.50 — is dominated by recoverable
excursions. The number that matters operationally is the **terminal crossing**:
the start of the final unbroken run above the threshold that reaches failure.
Both are now reported, and their gap *is* the false-alarm exposure.

On the surrogate (39 h to failure, modelled spall initiating at 22 h):

| tier | first crossing | terminal crossing | driver at terminal |
|---|---|---|---|
| watch | 6.17 h — 32.8 h of lead | 28.00 h — **11.00 h** of lead | `bpfo` |
| alert | 10.17 h — 28.8 h of lead | 28.83 h — **10.17 h** of lead | `bpfo` |
| critical | 30.00 h — 9.0 h of lead | 33.00 h — **6.00 h** of lead | `rms` |

The attribution is correct: `bpfo`, the channel the spall was injected on, drives
both the watch and alert crossings, with `rms` only taking over during terminal
break-up.

## 5. The lead time is stable; the false alarms are what move

108 configurations — healthy window (5/10/15/25% of the record) x EMA tau
(1/2/4 h) x persistence (1/3/6 samples) x MAD floor (0/1/2%):

| | median | range |
|---|---|---|
| first-crossing alert lead | 20.00 h | 9.3 – 32.8 h |
| **terminal alert lead** | **10.08 h** | **9.3 – 11.2 h** |

The terminal lead varies by ±9% across every reasonable analyst choice; the
first-crossing lead varies by a factor of 3.5. Nothing in the sweep failed to
alert before failure. This is the shape of result you want: the detection
instant is a property of the degradation, and the tuning knobs only move how
much noise you tolerate on the way there.

Two knobs turn out not to matter on this record and can be left at spec:
persistence changes the terminal lead by nothing at all (10.08 h at 1, 3 and 6
samples), and the MAD floor changes it by nothing (10.08 h at 0%, 1% and 2%).
The MAD floor still earns its place — the surrogate's healthy dispersion is
realistic; a rig held to a tenth of that would make 3.5 sigma trip on a 1%
change, which a test covers.

## 6. Across every modelled failure mode and seed

Five failure modes x six seeds x two feature variants x three scoring variants,
180 scored runs (`reports/campaign/`). Modes differ in how the defect shows up: an outer-race
spall sits in a fixed load zone and is unmodulated; an inner-race defect passes
through the load zone once per shaft revolution; a rolling-element defect is
modulated at cage rate; a cage fault rides the Factor-40 line; a grease dry-out
produces friction heat and broadband noise with no defect line at all.

**Detection held in 180 of 180 runs**, at a median terminal lead of 11.0 h —
27.8% of life, range 22–32% by mode. No mode, seed or variant failed to establish
an alert before failure.

**Attribution did not hold, and I could not fix it.** `fault_channel` is a
delivered output of KX-VAY-012 and is what tells a fitter which bearing element
to inspect. Measured at the terminal crossing:

| fault mode | correct attribution |
|---|---|
| `outer_race` | 6/6 |
| `cage` | 6/6 |
| `inner_race` | **1–2 / 6** — reports `rms` |
| `ball_spin` | **1–2 / 6** — reports `rms` |

The split is exactly the unmodulated/modulated one. An inner-race defect passes
through the load zone once per shaft revolution and a rolling-element defect is
modulated at cage rate, so their energy spreads into sidebands at
`f_defect ± modulation` rather than concentrating in the single bin the Goertzel
ratio watches. Pooled over the degradation phase the correct line still leads —
but at ~1.5x contrast against the other channels, versus ~3.0x for an
unmodulated defect. A test pins that ratio so the claim fails loudly if the
model changes.

Halved contrast alone would not decide the `max`, though. The second half of the
mechanism is that the five channels are not comparable after normalisation.
Baseline relative dispersion, measured on the surrogate's healthy phase:

| channel | MAD / median | alert limit |
|---|---|---|
| `rms` | **3.6%** | 3.5 |
| `bpfi` | 15.1% | 4.0 |
| `ftf` | 15.8% | 4.0 |
| `bsf` | 24.5% | 4.0 |
| `bpfo` | 25.6% | 4.0 |

RMS is four to seven times tighter, so the same *relative* rise buys it four to
seven times more sigma. `S = max_c(z_c / z_alert,c)` is therefore biased toward
RMS by construction, and the 3.5-vs-4.0 limit split offsets only a fraction of
it. A modulated defect at half contrast loses the argmax to broadband RMS even
while its own line is genuinely elevated.

Four fixes were tried. **One helps, three do not.**

*Works, partially* — **attribute to a defect line ahead of broadband RMS**
(`--attribution spectral_priority`). If any spectral channel is above its own
control limit, `fault_channel` names the highest of those; only if none is does
it fall back to `rms`. This is how an analyst reads a spectrum: RMS says
something is wrong, a defect line says what. On the modulated modes it took
attribution from 6/24 to **12/24** — and the paired comparison is the reassuring
part: **5 cases fixed, 0 broken** (McNemar exact p = 0.0625). It never regresses
a case it previously got right, and it costs nothing in lead time. It does cost
one case on the thermal mode, correctly: there the true driver *is* broadband.

Still a coin flip on the faults that matter, so this is a mitigation, not a fix.

*Does not work* — **sideband-aware ratios** (`--sidebands`). The obvious attack
on the contrast half: recombine the energy that modulation split into sidebands
at `f_defect ± shaft rate` (BPFI) and `± cage rate` (BSF). Measured on the
surrogate, the ±1 sidebands do carry real energy (~1.9x the noise floor against
a 2.5x centre line) and ±2 carry none, so the family was capped at ±1 and
summed rather than averaged. It made no difference: **2 fixed, 2 broken,
p = 1.0**, and it cost ~0.3 h of lead time. The option stays because on real
data the sideband structure is physical rather than modelled, and that is worth
one check — but on this evidence it is not the answer.

*Does not work* — **attribute from the smoothed severity** rather than the
instantaneous value. The score is smoothed and `fault_channel` is not, which
looked like an inconsistency worth closing. No effect at all (`inner_race`
stayed at 1/6). The ordering problem is dispersion, not noise.

*Does not work* — **attribute at the first crossing** instead of the terminal
one, on the theory that a defect signature is most specific when it first
emerges. Worse everywhere: 1/6 for `outer_race` against 6/6 at the terminal
crossing, because early crossings are frequently drift or knock episodes, which
are themselves `rms`-driven.

**The thermal mode is the one to treat with suspicion.** It was detected 6/6 —
but through `rms`, never through temperature, because `Y` does not consume
`temp1`/`temp2` at all. It worked only because grease dry-out was modelled as
raising broadband vibration alongside heat. A thermal failure that ran hot
*without* a vibration signature would be invisible to the target, and nothing in
this campaign rules that out. The datasheet has a dedicated Temp Alert byte;
`Y` has no thermal term. That gap is real regardless of what the surrogate says.

**Ball spin is model-limited, not a finding.** The modelled cage modulation
leaves at most 3.3x peak-to-floor at any BSF sideband, so there is little for
any detector to find. Whether a real rolling-element defect is that weak is a
physics question this surrogate cannot settle — check it on the archive before
concluding anything about BSF.

## 7. Y has no thermal term. It should have exactly one.

The Smart-Idler datasheet specifies a dedicated **Temp Alert** byte and
independent left/right raceway monitoring. KX-VAY-012 section 6.2 calls a >4 °C
end-to-end divergence under steady speed an anomaly in its own right. Yet `Y` as
constructed consumes five vibration channels and no thermal one — `temp1` and
`temp2` reach the models only as input features, never as evidence in the label.
A grease dry-out or seal failure that ran hot without a vibration signature
would score `normal` all the way to seizure.

Two thermal channels are now available. The evidence says take one and refuse
the other.

**Refuse `temp_max` (the hotter raceway).** Section 6.1 item 3 already rules out
seasonal drift as an anomaly — "both bearing ends cool equally" — and an
absolute-temperature channel cannot tell that from a fault. Scored against the
field-shaped surrogate, which carries the record's real 43.6 → 34.4 °C seasonal
swing:

| baseline fitted | none (as delivered) | + asymmetry | + `temp_max` |
|---|---|---|---|
| first 10%, warm start | 1.54% alert | 1.54% | 1.54% |
| coolest 10%, commissioned in winter | 0.32% alert | 0.34% | **71.21%** |

A sensor commissioned in a cool period spends most of its life flagged, with
57,571 rows attributed to `temp_max`. This is not a tuning problem; absolute
temperature is the wrong quantity.

**Take `temp_asymmetry` (|T₁ − T₂|).** Ambient-invariant by construction, which
is precisely why the datasheet monitors both ends separately. It moved the
field-surrogate alert rate by 0.02 percentage points under the adverse baseline.
It carries section 6.2's absolute threshold as well as a statistical one: a 4 °C
divergence sits at severity 1.0 however tight the fitted baseline happens to be.

**What the surrogate cannot tell you.** Enabling asymmetry changed terminal lead
time by under 0.2 h on every mode and left attribution unchanged except on the
thermal mode. That is not evidence against it — my surrogate always couples heat
to broadband noise, so the very failure class the channel exists for (hot, quiet)
is one it cannot represent. What the surrogate *can* confirm is that the channel
is well-behaved: after correcting the model so that only lubrication failures
run one end hot, `temp_asymmetry` never claimed attribution on a spall mode, and
claimed it on 2 of 6 thermal runs. The case for enabling it rests on the
datasheet and section 6.2, not on a measured lead-time gain.

## 8. Reading this across to the business question

11 h of warning on a ~39 h accelerated test is **27.8% of total life** (median
over all 180 runs), with the alert landing roughly 40% of the way into the
damage phase. Absolute hours do not transfer — an accelerated
test compresses propagation that takes weeks in a field idler — but the
*fractional* lead does, and 26% of remaining life on a conveyor idler whose
degradation phase runs weeks is the "days to weeks" margin the programme is
after.

That claim is only worth as much as the degradation law behind it, which here is
modelled. Re-run against the real archive before it goes to the client.

## 9. What to run next, in order

1. **The real archive**, with the command at the top. Compare terminal lead
   under `--orders vayeron` against `--orders geometry`; the gap quantifies what
   the fielded firmware's fixed multipliers cost on a bearing they were not cut
   for.
2. **Check inner-race attribution on the archive.** Four fixes are now tested
   (section 6): `spectral_priority` halves the problem, sidebands and the other
   two do nothing. Whether the residual holds on a real inner-race fault decides
   whether this needs Vayeron in the room or is good enough to ship.
3. **Run the label audit on `vayeron_anomaly_dataset.csv`.** One command, and it
   settles whether the spike-through-EMA defect touches the field labels at all
   — the threshold is ~22 sigma and the field's knock distribution decides it.
   If it does, every benchmark number in KX-VAY-013 is partly measuring
   agreement with smeared knocks, and the prefilter should go into KX-VAY-012
   before the models are retrained. If it does not, the prefilter is still worth
   taking as cheap insurance, and one open question closes.
4. **Re-derive the tier thresholds from the real curve.** 0.35 / 0.50 / 0.80 were
   set to make the field distribution come out at 88/6/6, not to hit a lead-time
   target. With a real time-to-failure axis, they can be set to buy a chosen
   number of days instead.
