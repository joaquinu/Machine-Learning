"""Window construction and splits for the anomaly-tier benchmark.

Follows KX-VAY-014: 12-reading windows (~2 h at the 626 s cadence), a 60/20/20
split stratified by target class over *whole* windows, never split across sets,
min-max normalised on training statistics only.

It adds one thing that document does not have: **minority-class overlapping
augmentation**. Non-overlapping windows turn 81,774 rows into ~6,500 samples and
throw away every window that straddles a boundary. Sliding the window recovers
them. Doing that for the whole record just inflates the majority class, so the
extra windows are drawn only where the label is scarce.

The trap this module exists to avoid
------------------------------------
Overlapping windows are near-duplicates of each other. Generate them before
splitting and the test set fills with shifted copies of training windows, and
the score goes up for a reason that has nothing to do with the model. So
augmentation happens **after** the split, and an augmented window is admitted
only when every row it covers already belongs to a training window.

``leaky_augmentation`` reproduces the wrong version deliberately, so the size of
the illusion can be measured rather than argued about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# KX-VAY-014's feature set, less rssi (cut there) and hours_since_restart
# (not represented in the surrogate).
BENCHMARK_FEATURES: tuple[str, ...] = (
    "operating_angle_deg", "rpm", "rms_per_rpm", "temp1", "temp2", "temp_delta",
    "bpfo", "bpfi", "bsf", "ftf",
)
TIERS: tuple[str, ...] = ("normal", "watch", "alert")


@dataclass
class WindowConfig:
    seq_len: int = 12
    features: tuple[str, ...] = BENCHMARK_FEATURES
    group_column: str = "roller_id"
    label_column: str = "anomaly_tier"
    # Split fractions over whole non-overlapping windows, stratified by label.
    train_frac: float = 0.60
    val_frac: float = 0.20
    seed: int = 20260906
    # Augmentation.
    augment_tiers: tuple[str, ...] = ("watch", "alert")
    augment_stride: int = 1
    leaky_augmentation: bool = False


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    stats: dict = field(default_factory=dict)

    def counts(self, split: str = "train") -> dict[str, int]:
        y = getattr(self, f"y_{split}")
        return {t: int((y == i).sum()) for i, t in enumerate(TIERS)}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the benchmark features that score_run does not already emit."""
    out = df.copy()
    if "rms_per_rpm" not in out.columns:
        out["rms_per_rpm"] = out["rms_norm"] if "rms_norm" in out.columns else (
            out["rms"] / (out["rpm"] / 600.0))
    if "temp_delta" not in out.columns and {"temp1", "temp2"}.issubset(out.columns):
        out["temp_delta"] = out["temp1"] - out["temp2"]
    # Collapse critical into alert: the benchmark is 3-way.
    out["_label"] = out["anomaly_tier"].astype(str).replace({"critical": "alert"})
    return out


def _windows(n: int, seq_len: int, stride: int) -> np.ndarray:
    if n < seq_len:
        return np.empty((0, 2), dtype=int)
    starts = np.arange(0, n - seq_len + 1, stride)
    return np.stack([starts, starts + seq_len], axis=1)


def _label_of(labels: np.ndarray, hi: int) -> int:
    """The window's label is its last reading's tier, as KX-VAY-014 validates."""
    return int(labels[hi - 1])


def build(scored: pd.DataFrame, cfg: WindowConfig | None = None) -> Dataset:
    cfg = cfg or WindowConfig()
    rng = np.random.default_rng(cfg.seed)
    df = _prepare(scored)
    missing = [c for c in cfg.features if c not in df.columns]
    if missing:
        raise KeyError(f"missing benchmark features: {missing}")

    label_index = {t: i for i, t in enumerate(TIERS)}

    base: list[tuple[str, int, int, int]] = []   # group, lo, hi, label
    per_group: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group, g in df.groupby(cfg.group_column, sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)
        values = g[list(cfg.features)].to_numpy(dtype=float)
        labels = g["_label"].map(label_index).to_numpy(dtype=int)
        per_group[group] = (values, labels)
        for lo, hi in _windows(len(g), cfg.seq_len, cfg.seq_len):
            base.append((group, int(lo), int(hi), _label_of(labels, hi)))

    # --- stratified 60/20/20 over whole non-overlapping windows --------------
    base_arr = np.array([(g, lo, hi, y) for g, lo, hi, y in base], dtype=object)
    assign = np.empty(len(base_arr), dtype=object)
    for y in range(len(TIERS)):
        idx = np.flatnonzero(np.array([b[3] for b in base]) == y)
        rng.shuffle(idx)
        n_tr = int(round(cfg.train_frac * idx.size))
        n_va = int(round(cfg.val_frac * idx.size))
        assign[idx[:n_tr]] = "train"
        assign[idx[n_tr:n_tr + n_va]] = "val"
        assign[idx[n_tr + n_va:]] = "test"

    # Rows owned by the training split, per group - the admissibility mask that
    # keeps augmentation from reaching into val/test.
    train_rows = {g: np.zeros(len(per_group[g][1]), dtype=bool) for g in per_group}
    for (g, lo, hi, _), split in zip(base, assign):
        if split == "train":
            train_rows[g][lo:hi] = True

    splits: dict[str, list[tuple[str, int, int, int]]] = {"train": [], "val": [], "test": []}
    for b, split in zip(base, assign):
        splits[split].append(b)

    # --- minority-class overlapping augmentation ----------------------------
    augment_ids = {label_index[t] for t in cfg.augment_tiers}
    if cfg.augment_stride and augment_ids:
        seen = {(g, lo) for g, lo, _, _ in splits["train"]}
        for group, (_, labels) in per_group.items():
            for lo, hi in _windows(len(labels), cfg.seq_len, cfg.augment_stride):
                y = _label_of(labels, hi)
                if y not in augment_ids or (group, int(lo)) in seen:
                    continue
                if not cfg.leaky_augmentation and not train_rows[group][lo:hi].all():
                    # Touches a row owned by val or test - would leak.
                    continue
                splits["train"].append((group, int(lo), int(hi), y))
                seen.add((group, int(lo)))

    def materialise(rows: list[tuple[str, int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
        if not rows:
            return (np.empty((0, cfg.seq_len, len(cfg.features))), np.empty(0, dtype=int))
        X = np.stack([per_group[g][0][lo:hi] for g, lo, hi, _ in rows])
        y = np.array([lab for *_, lab in rows], dtype=int)
        return X, y

    Xtr, ytr = materialise(splits["train"])
    Xva, yva = materialise(splits["val"])
    Xte, yte = materialise(splits["test"])

    # --- min-max on training statistics only --------------------------------
    lo_s = Xtr.reshape(-1, Xtr.shape[-1]).min(axis=0)
    hi_s = Xtr.reshape(-1, Xtr.shape[-1]).max(axis=0)
    span = np.where(hi_s - lo_s > 0, hi_s - lo_s, 1.0)
    norm = lambda X: (X - lo_s) / span if len(X) else X  # noqa: E731

    return Dataset(
        X_train=norm(Xtr), y_train=ytr,
        X_val=norm(Xva), y_val=yva,
        X_test=norm(Xte), y_test=yte,
        feature_names=cfg.features,
        stats={"min": lo_s, "max": hi_s,
               "base_windows": len(base),
               "augmented_windows": len(splits["train"]) - int((assign == "train").sum())},
    )
