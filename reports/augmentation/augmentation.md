# Minority-class overlapping augmentation

Non-overlapping 2-hour windows turn the record into ~6,800 samples and discard every window that straddles a boundary. Sliding the window recovers them; doing it only where the label is scarce grows the minority classes without inflating `normal`.

**All data is modelled.** The field-shaped surrogate is tuned so its scored tier distribution matches the real export (88.07 / 5.82 / 6.11 against KX-VAY-012's 88.06 / 5.86 / 6.08), because that balance is the thing augmentation acts on. It reproduces the cadence, geometry split, rpm regimes, drift and episode structure of the deployment - not the mine.

Small GRU (2667 parameters), checkpoint selected on class-weighted validation loss, 3 seeds. One untouched test set throughout.

| variant | train windows | macro-F1 | accuracy |
|---|---|---|---|
| baseline | 4,080 | **0.672** ± 0.014 | 0.832 |
| leak-free augmentation | 7,222 | **0.730** ± 0.034 | 0.893 |
| LEAKY augmentation | 13,329 | **0.733** ± 0.041 | 0.898 |

Per class - the headline metric and the operational one disagree:

| variant | recall watch | recall alert | precision watch | precision alert | recall normal |
|---|---|---|---|---|---|
| baseline | 0.783 | 0.840 | 0.260 | 0.632 | 0.835 |
| leak-free augmentation | 0.746 | 0.778 | 0.375 | 0.718 | 0.911 |
| LEAKY augmentation | 0.704 | 0.765 | 0.390 | 0.740 | 0.920 |

- Real effect (leak-free − baseline): **+0.057** macro-F1.
- Leak effect (leaky − leak-free): **+0.003** macro-F1.

The leak was expected to inflate the score substantially and did not. Two likely reasons, both specific to this setup: the model is tiny (2.7k parameters) so it has little capacity to memorise a shifted duplicate, and the surrogate's minority episodes are long, so a window shifted by one reading carries almost the same information as its neighbour either way. Neither makes leaking safe - with a larger model, or shorter and sparser episodes, the same mistake could pay much better. The split-then-augment order costs nothing, so there is no reason to take the risk.

## The headline metric and the operational one disagree

Macro-F1 rose, but it did so by buying precision with recall. Augmentation gave the alert class **+8.6** points of precision and cost it **-6.2** points of recall. For predictive maintenance that is the wrong direction: a missed alert is a seized roller and a torn belt, a false alert is an inspection. KX-VAY-013's own recommendation - ship the binary detector at 76% recall - implies recall is what the programme values.

So the honest reading is that this augmentation is **not** a free improvement. It is a knob that trades detections for false alarms, and the tier thresholds already do that more directly and more legibly. Take it only if a precision problem is what you actually have.
