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

**A single-sample knock produces a multi-sample alert.** Section 6.1 item 6
says an isolated spike is not an anomaly, and the persistence rule is what is
meant to enforce that. It does not: the 2 h EMA *spreads* a lone spike across
roughly its span, so the smoothed severity sits above the limit for many
consecutive samples and clears any persistence test. In the surrogate this
produced two clean alert episodes at 6 h and 10 h, hours before any damage
existed. A 3-sample median prefilter on raw severity, ahead of the EMA, fixes
it:

| | pre-onset time above the alert threshold | terminal alert lead time |
|---|---|---|
| KX-VAY-012 as written | 0.67 h | 10.17 h |
| `--prefilter-median 3` | **0.00 h** | 10.00 h |

All the false-alarm exposure removed, 2% of the lead time given up.

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

## 6. Reading this across to the business question

10 h of warning on a 39 h accelerated test is **26% of total life**, with the
alert landing 40% of the way into the damage phase. Absolute hours do not transfer — an accelerated
test compresses propagation that takes weeks in a field idler — but the
*fractional* lead does, and 26% of remaining life on a conveyor idler whose
degradation phase runs weeks is the "days to weeks" margin the programme is
after.

That claim is only worth as much as the degradation law behind it, which here is
modelled. Re-run against the real archive before it goes to the client.

## 7. What to run next, in order

1. **The real archive**, with the command at the top. Compare terminal lead
   under `--orders vayeron` against `--orders geometry`; the gap quantifies what
   the fielded firmware's fixed multipliers cost on a bearing they were not cut
   for.
2. **Fold the prefilter decision back into KX-VAY-012.** The spike-through-EMA
   defect affects the field dataset's labels too, and therefore every benchmark
   number in KX-VAY-013 — the models are being asked to predict alert episodes
   that are smeared knocks.
3. **Re-derive the tier thresholds from the real curve.** 0.35 / 0.50 / 0.80 were
   set to make the field distribution come out at 88/6/6, not to hit a lead-time
   target. With a real time-to-failure axis, they can be set to buy a chosen
   number of days instead.
