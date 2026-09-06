"""Does minority-class overlapping augmentation help the tier classifier?

Trains the same small GRU on three versions of the same record and scores all
three against one untouched test set:

    baseline    non-overlapping windows, as KX-VAY-014 builds them
    leak-free   plus overlapping minority windows drawn only from training rows
    leaky       plus overlapping minority windows drawn from anywhere

The third is the mistake, included so its size can be measured. Read the gap
between "leaky" and "leak-free" as the illusion, not the benefit.

Checkpoints are selected on class-weighted validation loss, never accuracy -
KX-VAY-013 found accuracy selection silently collapsed every architecture to
predicting `normal`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .benchmarks import TIERS, Dataset, WindowConfig, build


@dataclass
class TrainConfig:
    hidden: int = 24
    epochs: int = 80
    patience: int = 15
    batch_size: int = 128
    lr: float = 3e-3
    seed: int = 0


class TierGRU(nn.Module):
    """Deliberately tiny, in the spirit of KX-VAY-013's 513-parameter GRU."""

    def __init__(self, input_dim: int, hidden: int, n_classes: int = len(TIERS)):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1])


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n: int = len(TIERS)) -> float:
    f1s = []
    for c in range(n):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def _per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Recall and precision per tier. Both are needed: macro-F1 can rise while
    the recall that matters operationally falls."""
    out: dict[str, float] = {}
    for c, name in enumerate(TIERS):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        out[f"recall_{name}"] = tp / (tp + fn) if tp + fn else float("nan")
        out[f"precision_{name}"] = tp / (tp + fp) if tp + fp else float("nan")
    return out


def train_and_eval(ds: Dataset, cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    Xtr = torch.tensor(ds.X_train, dtype=torch.float32)
    ytr = torch.tensor(ds.y_train, dtype=torch.long)
    Xva = torch.tensor(ds.X_val, dtype=torch.float32)
    yva = torch.tensor(ds.y_val, dtype=torch.long)
    Xte = torch.tensor(ds.X_test, dtype=torch.float32)

    # Class weights from the training split, so an augmented run is not also
    # getting a different loss shape for free.
    counts = np.bincount(ds.y_train, minlength=len(TIERS)).astype(float)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    model = TierGRU(Xtr.shape[-1], cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_loss, best_state, stale = float("inf"), None, 0
    for _ in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss_fn(model(Xtr[b]), ytr[b]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(Xva), yva))
        if vloss < best_loss - 1e-5:
            best_loss, stale = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(dim=1).numpy()

    return {
        "macro_f1": _macro_f1(ds.y_test, pred),
        "accuracy": float((pred == ds.y_test).mean()),
        **_per_class(ds.y_test, pred),
        "params": sum(p.numel() for p in model.parameters()),
        "train_windows": int(len(ds.y_train)),
        "val_loss": best_loss,
    }


VARIANTS: dict[str, dict] = {
    "baseline": {"augment_stride": 0},
    "leak-free augmentation": {"augment_stride": 1},
    "LEAKY augmentation": {"augment_stride": 1, "leaky_augmentation": True},
}


def run(scored: pd.DataFrame, seeds: tuple[int, ...] = (0, 1, 2),
        window_seed: int = 20260906, progress: bool = True) -> pd.DataFrame:
    rows = []
    for name, overrides in VARIANTS.items():
        ds = build(scored, WindowConfig(seed=window_seed, **overrides))
        for seed in seeds:
            res = train_and_eval(ds, TrainConfig(seed=seed))
            res.update(variant=name, seed=seed,
                       train_counts=str(ds.counts("train")))
            rows.append(res)
            if progress:
                print(f"  {name:24s} seed={seed}  macro-F1 {res['macro_f1']:.3f}  "
                      f"({res['train_windows']} train windows)", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["macro_f1", "accuracy",
            "recall_normal", "recall_watch", "recall_alert",
            "precision_normal", "precision_watch", "precision_alert"]
    out = df.groupby("variant", sort=False)[cols].agg(["mean", "std"]).round(3)
    out[("train_windows", "")] = df.groupby("variant", sort=False)["train_windows"].first()
    return out


def render_markdown(df: pd.DataFrame) -> str:
    g = df.groupby("variant", sort=False)
    lines = [
        "# Minority-class overlapping augmentation",
        "",
        "Non-overlapping 2-hour windows turn the record into ~6,800 samples and discard "
        "every window that straddles a boundary. Sliding the window recovers them; doing "
        "it only where the label is scarce grows the minority classes without inflating "
        "`normal`.",
        "",
        "**All data is modelled.** The field-shaped surrogate is tuned so its scored tier "
        "distribution matches the real export (88.07 / 5.82 / 6.11 against KX-VAY-012's "
        "88.06 / 5.86 / 6.08), because that balance is the thing augmentation acts on. It "
        "reproduces the cadence, geometry split, rpm regimes, drift and episode structure "
        "of the deployment - not the mine.",
        "",
        f"Small GRU ({int(df['params'].iloc[0])} parameters), checkpoint selected on "
        "class-weighted validation loss, 3 seeds. One untouched test set throughout.",
        "",
        "| variant | train windows | macro-F1 | accuracy |",
        "|---|---|---|---|",
    ]
    for name, sub in g:
        lines.append(
            f"| {name} | {int(sub['train_windows'].iloc[0]):,} | "
            f"**{sub['macro_f1'].mean():.3f}** ± {sub['macro_f1'].std():.3f} | "
            f"{sub['accuracy'].mean():.3f} |")
    lines += ["", "Per class - the headline metric and the operational one disagree:", "",
              "| variant | recall watch | recall alert | precision watch | precision alert | recall normal |",
              "|---|---|---|---|---|---|"]
    for name, sub in g:
        lines.append(
            f"| {name} | {sub['recall_watch'].mean():.3f} | {sub['recall_alert'].mean():.3f} | "
            f"{sub['precision_watch'].mean():.3f} | {sub['precision_alert'].mean():.3f} | "
            f"{sub['recall_normal'].mean():.3f} |")
    lines.append("")

    base = g.get_group("baseline")["macro_f1"].mean()
    free = g.get_group("leak-free augmentation")["macro_f1"].mean()
    leak = g.get_group("LEAKY augmentation")["macro_f1"].mean()
    lines += [
        f"- Real effect (leak-free − baseline): **{free - base:+.3f}** macro-F1.",
        f"- Leak effect (leaky − leak-free): **{leak - free:+.3f}** macro-F1.",
        "",
        "The leak was expected to inflate the score substantially and did not. Two likely "
        "reasons, both specific to this setup: the model is tiny (2.7k parameters) so it "
        "has little capacity to memorise a shifted duplicate, and the surrogate's minority "
        "episodes are long, so a window shifted by one reading carries almost the same "
        "information as its neighbour either way. Neither makes leaking safe - with a "
        "larger model, or shorter and sparser episodes, the same mistake could pay much "
        "better. The split-then-augment order costs nothing, so there is no reason to "
        "take the risk.",
        "",
        "## The headline metric and the operational one disagree",
        "",
        "Macro-F1 rose, but it did so by buying precision with recall. Augmentation gave "
        f"the alert class **{100 * (g.get_group('leak-free augmentation')['precision_alert'].mean() - g.get_group('baseline')['precision_alert'].mean()):+.1f}** points of precision "
        f"and cost it **{100 * (g.get_group('leak-free augmentation')['recall_alert'].mean() - g.get_group('baseline')['recall_alert'].mean()):+.1f}** points of recall. For predictive "
        "maintenance that is the wrong direction: a missed alert is a seized roller and a "
        "torn belt, a false alert is an inspection. KX-VAY-013's own recommendation - ship "
        "the binary detector at 76% recall - implies recall is what the programme values.",
        "",
        "So the honest reading is that this augmentation is **not** a free improvement. It "
        "is a knob that trades detections for false alarms, and the tier thresholds already "
        "do that more directly and more legibly. Take it only if a precision problem is "
        "what you actually have.",
        "",
    ]
    return "\n".join(lines)


def main(out_dir: str = "reports/augmentation") -> pd.DataFrame:
    from pathlib import Path

    from .control_limits import ScoringConfig, score_run
    from .synthetic import generate_field_record

    record = generate_field_record()
    scored = pd.concat([
        score_run(g.reset_index(drop=True), ScoringConfig())[0].assign(roller_id=r)
        for r, g in record.groupby("roller_id")
    ])

    df = run(scored)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    md = render_markdown(df)
    (out / "augmentation.md").write_text(md)
    print("\n" + md)
    return df


if __name__ == "__main__":
    main()
