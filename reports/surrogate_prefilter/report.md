# Vayeron control limits with a 3-sample median prefilter

Source: **modelled run-to-failure surrogate** (240 acquisitions, 1770 rpm, 25.6 kHz, outer-race spall from 55% of life). Not physical data.

## Control limits applied

| channel | z_alert | baseline median | baseline MAD | sigma-equivalent |
|---|---|---|---|---|
| `rms` | 3.5 | 0.5010 | 0.0179 | 0.0265 |
| `bpfo` | 4.0 | 2.0244 | 0.5186 | 0.7689 |
| `bpfi` | 4.0 | 1.7330 | 0.2607 | 0.3866 |
| `bsf` | 4.0 | 1.7813 | 0.4357 | 0.6460 |
| `ftf` | 4.0 | 1.6237 | 0.2558 | 0.3792 |

- Healthy baseline: first **36** valid acquisitions (**5.83 h** of run time).
- Acquisition cadence: median **600.0 s** (cv 0.000).
- Persistence smoothing: EMA `span`, tau = 2.0 h, span = 12.0 samples.
- Quality gates: rpm in (1500.0, 2000.0), rms > 0 -> **0** of 240 rows gated to Y = 0.

## Time to failure

Failure reference: acquisition **234** of 240 at `2026-05-02 15:00:00` (t = 39.00 h).

A crossing counts only when it persists for **3 consecutive acquisitions**; single-sample spikes are not anomalies (KX-VAY-012 6.1).

| tier | Y threshold | first sustained crossing | lead time | % of life remaining | driver |
|---|---|---|---|---|---|
| **watch** | 0.35 | acq 168 @ 28.00 h | 11.00 h (0.46 d) | 28.2% | `bpfo` |
| **alert** | 0.50 | acq 174 @ 29.00 h | 10.00 h (0.42 d) | 25.6% | `bpfo` |
| **critical** | 0.80 | acq 180 @ 30.00 h | 9.00 h (0.38 d) | 23.1% | `rms` |

The first crossing can be a recoverable excursion (baseline drift, a load change, a knock). The **terminal crossing** is the start of the final unbroken run above the threshold that reaches failure - the one that corresponds to irreversible degradation. The gap between the two columns is the false-alarm exposure an operator would live with.

| tier | terminal crossing | terminal lead time | driver | earlier recovering episodes | time spent in them |
|---|---|---|---|---|---|
| **watch** | acq 168 @ 28.00 h | 11.00 h (0.46 d) | `bpfo` | 0 | 0.00 h |
| **alert** | acq 174 @ 29.00 h | 10.00 h (0.42 d) | `bpfo` | 0 | 0.00 h |
| **critical** | acq 199 @ 33.17 h | 5.83 h (0.24 d) | `rms` | 1 | 0.83 h |

- Peak anomaly score: **1.000**, reached 0.00 h (0.00 d) before failure.

## Against the known damage onset (surrogate only)

- Modelled spall initiates at acquisition **132** (t = 22.00 h).
- Terminal alert fires **7.00 h (0.29 d)** after onset.
- Time spent at or above the alert threshold *before* any damage exists (false-alarm exposure): **0.00 h**.

## Tier distribution over the whole run

| tier | share of acquisitions |
|---|---|
| normal | 70.00% |
| watch | 2.50% |
| alert | 8.33% |
| critical | 19.17% |
