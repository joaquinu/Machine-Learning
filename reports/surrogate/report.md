# Vayeron control limits applied to a run-to-failure record

Source: **modelled run-to-failure surrogate** (240 acquisitions, 1770 rpm, 25.6 kHz, outer-race spall from 55% of life). Not physical data.

## Control limits applied

| channel | z_alert | baseline median | baseline MAD | sigma-equivalent |
|---|---|---|---|---|
| `rms` | 3.5 | 0.5010 | 0.0179 | 0.0265 |
| `bpfo` | 4.0 | 2.0244 | 0.5186 | 0.7689 |
| `bpfi` | 4.0 | 1.7383 | 0.2894 | 0.4290 |
| `bsf` | 4.0 | 1.7813 | 0.4357 | 0.6460 |
| `ftf` | 4.0 | 1.6680 | 0.2600 | 0.3854 |

- Healthy baseline: first **36** valid acquisitions (**5.83 h** of run time).
- Acquisition cadence: median **600.0 s** (cv 0.000).
- Persistence smoothing: EMA `span`, tau = 2.0 h, span = 12.0 samples.
- Quality gates: rpm in (1500.0, 2000.0), rms > 0 -> **0** of 240 rows gated to Y = 0.

## Time to failure

Failure reference: acquisition **235** of 240 at `2026-05-02 15:10:00` (t = 39.17 h).

A crossing counts only when it persists for **3 consecutive acquisitions**; single-sample spikes are not anomalies (KX-VAY-012 6.1).

| tier | Y threshold | first sustained crossing | lead time | % of life remaining | driver |
|---|---|---|---|---|---|
| **watch** | 0.35 | acq 37 @ 6.17 h | 33.00 h (1.38 d) | 84.3% | `rms` |
| **alert** | 0.50 | acq 174 @ 29.00 h | 10.17 h (0.42 d) | 26.0% | `bpfo` |
| **critical** | 0.80 | acq 180 @ 30.00 h | 9.17 h (0.38 d) | 23.4% | `rms` |

The first crossing can be a recoverable excursion (baseline drift, a load change, a knock). The **terminal crossing** is the start of the final unbroken run above the threshold that reaches failure - the one that corresponds to irreversible degradation. The gap between the two columns is the false-alarm exposure an operator would live with.

| tier | terminal crossing | terminal lead time | driver | earlier recovering episodes | time spent in them |
|---|---|---|---|---|---|
| **watch** | acq 168 @ 28.00 h | 11.17 h (0.47 d) | `bpfo` | 2 | 1.83 h |
| **alert** | acq 174 @ 29.00 h | 10.17 h (0.42 d) | `bpfo` | 0 | 0.00 h |
| **critical** | acq 198 @ 33.00 h | 6.17 h (0.26 d) | `rms` | 1 | 1.50 h |

- Peak anomaly score: **1.000**, reached 0.00 h (0.00 d) before failure.

## Against the known damage onset (surrogate only)

- Modelled spall initiates at acquisition **132** (t = 22.00 h).
- Terminal alert fires **7.00 h (0.29 d)** after onset.
- Time spent at or above the alert threshold *before* any damage exists (false-alarm exposure): **0.50 h**.

## Tier distribution over the whole run

| tier | share of acquisitions |
|---|---|
| normal | 65.42% |
| watch | 5.42% |
| alert | 7.92% |
| critical | 21.25% |

## Sensitivity of the lead time to analyst choices

108 parameter combinations (healthy window x EMA tau x persistence x MAD floor).

- Alert lead time: median **10.33 h** (0.43 d), range 9.50-33.00 h, IQR 10.00-33.00 h.
- Terminal (irreversible) alert lead time: median **10.17 h** (0.42 d), range 9.50-11.33 h.
- Median number of earlier recovering alert episodes: **0**.
- Configurations where the alert never fired before failure: **0**.
- Most frequent driving channel at the alert crossing: `bpfo`.

Median alert lead time (h), healthy window vs EMA tau:

| healthy window | tau = 1.0 h | tau = 2.0 h | tau = 4.0 h |
|---|---|---|---|
| 5% of record | 33.00 | 33.00 | 9.50 |
| 10% of record | 33.00 | 33.00 | 9.50 |
| 15% of record | 29.00 | 10.17 | 9.50 |
| 25% of record | 33.00 | 29.00 | 10.17 |
