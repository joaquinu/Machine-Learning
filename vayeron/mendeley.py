"""Loader for the Mendeley Data / KAIST ball-bearing run-to-failure dataset.

The dataset is an accelerated life test: a bearing is run at constant speed
(~1770 rpm) with vibration and bearing temperature recorded periodically until
the rig trips on its destruction criteria (bearing temperature ~85 C or
vibration ~9 m/s^2). That gives the physical ground truth the Vayeron field
record lacks.

Distributions of run-to-failure data vary in packaging, so the loader is
deliberately tolerant. It handles:

  * a directory of per-acquisition waveform files (``.csv``/``.txt``/``.mat``),
    one snapshot per file, ordered by an embedded timestamp or by name;
  * a single long waveform file, chopped into fixed-length snapshots;
  * an already-reduced feature table (one row per acquisition) - in which case
    it is passed through after column-name normalisation.

Anything the heuristics get wrong can be pinned explicitly through
:class:`LoadConfig`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .features import FeatureConfig, waveform_features

WAVEFORM_SUFFIXES = (".csv", ".txt", ".dat", ".mat", ".npy")

_VIBRATION_HINTS = ("vib", "accel", "acc", "acceleration", "ch1", "channel1", "signal", "x")
_TEMP_HINTS = ("temp", "temperature", "thermo", "ntc")
_TIME_HINTS = ("time", "timestamp", "date", "sec", "t")

_TS_PATTERNS = (
    re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})[-_ T]?(\d{2})[-_:]?(\d{2})[-_:]?(\d{2})"),
    re.compile(r"^(\d{10,13})$"),
)


@dataclass
class LoadConfig:
    """Everything the loader may need told to it rather than guessed."""

    fs: float = 25600.0                  # accelerometer sample rate, Hz
    rpm: float = 1770.0                  # constant test speed
    acquisition_interval_s: float = 600.0  # spacing between snapshots when not timestamped
    snapshot_seconds: float | None = None  # chop a long record into snapshots this long
    vibration_column: str | int | None = None
    temp_columns: Sequence[str] | None = None
    time_column: str | None = None
    has_header: bool | None = None
    max_snapshots: int | None = None
    file_glob: str = "*"
    defect_orders: dict[str, float] | None = None
    feature_overrides: dict = field(default_factory=dict)

    def feature_config(self) -> FeatureConfig:
        kwargs = dict(fs=self.fs, rpm=self.rpm, **self.feature_overrides)
        if self.defect_orders:
            kwargs["defect_orders"] = dict(self.defect_orders)
        return FeatureConfig(**kwargs)


def _natural_key(p: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)]


def timestamp_from_name(name: str) -> pd.Timestamp | None:
    stem = Path(name).stem
    m = _TS_PATTERNS[0].search(stem)
    if m:
        y, mo, d, h, mi, s = (int(g) for g in m.groups())
        try:
            return pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, second=s)
        except ValueError:
            return None
    m = _TS_PATTERNS[1].search(stem)
    if m:
        raw = int(m.group(1))
        unit = "ms" if raw > 10**11 else "s"
        try:
            return pd.Timestamp(raw, unit=unit)
        except ValueError:
            return None
    return None


def _pick_column(columns: Sequence[str], hints: Iterable[str]) -> str | None:
    lowered = {c: str(c).strip().lower() for c in columns}
    for hint in hints:
        for col, low in lowered.items():
            if low == hint:
                return col
    for hint in hints:
        for col, low in lowered.items():
            if hint in low:
                return col
    return None


def _read_table(path: Path, cfg: LoadConfig) -> pd.DataFrame:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr
        return pd.DataFrame(arr, columns=[f"col{i}" for i in range(arr.shape[1])])
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat

        mat = loadmat(path, squeeze_me=True)
        data = {k: np.asarray(v).ravel() for k, v in mat.items()
                if not k.startswith("__") and np.asarray(v).size > 1}
        if not data:
            raise ValueError(f"no array variables found in {path}")
        n = max(v.size for v in data.values())
        return pd.DataFrame({k: v for k, v in data.items() if v.size == n})

    header = 0 if cfg.has_header is not False else None
    if cfg.has_header is None:
        probe = pd.read_csv(path, nrows=1, header=None)
        looks_numeric = all(
            pd.api.types.is_numeric_dtype(pd.to_numeric(probe[c], errors="coerce"))
            and pd.to_numeric(probe[c], errors="coerce").notna().all()
            for c in probe.columns
        )
        header = None if looks_numeric else 0
    df = pd.read_csv(path, header=header)
    if header is None:
        df.columns = [f"col{i}" for i in range(df.shape[1])]
    return df


def _vibration_series(df: pd.DataFrame, cfg: LoadConfig) -> np.ndarray:
    if cfg.vibration_column is not None:
        col = df.columns[cfg.vibration_column] if isinstance(cfg.vibration_column, int) \
            else cfg.vibration_column
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    numeric = [c for c in df.columns
               if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
    named = _pick_column(numeric, _VIBRATION_HINTS)
    if named is not None:
        return pd.to_numeric(df[named], errors="coerce").to_numpy(dtype=float)
    # Fall back to the widest-swinging numeric column that is not a time ramp.
    best, best_std = None, -np.inf
    for c in numeric:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if v.size > 2 and np.all(np.diff(v[np.isfinite(v)]) > 0):
            continue  # monotonic -> a time axis
        s = float(np.nanstd(v))
        if s > best_std:
            best, best_std = c, s
    if best is None:
        raise ValueError("could not identify a vibration column; set LoadConfig.vibration_column")
    return pd.to_numeric(df[best], errors="coerce").to_numpy(dtype=float)


def _temperatures(df: pd.DataFrame, cfg: LoadConfig) -> dict[str, float]:
    if cfg.temp_columns is not None:
        cols = list(cfg.temp_columns)
    else:
        cols = [c for c in df.columns
                if any(h in str(c).strip().lower() for h in _TEMP_HINTS)]
    out: dict[str, float] = {}
    for i, c in enumerate(cols[:2], start=1):
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            out[f"temp{i}"] = float(vals.mean())
    return out


def discover_acquisitions(root: Path, cfg: LoadConfig) -> list[Path]:
    files = [p for p in sorted(root.rglob(cfg.file_glob), key=_natural_key)
             if p.is_file() and p.suffix.lower() in WAVEFORM_SUFFIXES]
    if cfg.max_snapshots:
        files = files[: cfg.max_snapshots]
    return files


def _snapshot_rows_from_long_record(values: np.ndarray, cfg: LoadConfig,
                                    start: pd.Timestamp) -> list[dict]:
    n = int(cfg.snapshot_seconds * cfg.fs)
    if n <= 0:
        raise ValueError("snapshot_seconds must be positive")
    fcfg = cfg.feature_config()
    rows = []
    count = values.size // n
    if cfg.max_snapshots:
        count = min(count, cfg.max_snapshots)
    for i in range(count):
        chunk = values[i * n : (i + 1) * n]
        row = {"timestamp": start + pd.Timedelta(seconds=i * cfg.acquisition_interval_s),
               "acquisition": i, "rpm": cfg.rpm}
        row.update(waveform_features(chunk, fcfg))
        rows.append(row)
    return rows


def build_feature_table(path: str | Path, cfg: LoadConfig | None = None,
                        progress: bool = False) -> pd.DataFrame:
    """Reduce a run-to-failure record to one Vayeron-shaped row per acquisition.

    Output columns: ``timestamp``, ``rpm``, ``rms``, ``bpfo``, ``bpfi``,
    ``bsf``, ``ftf`` (+ ``temp1``/``temp2`` and waveform diagnostics when
    available) - exactly the channels :func:`vayeron.control_limits.score_run`
    expects.
    """
    cfg = cfg or LoadConfig()
    path = Path(path)
    fcfg = cfg.feature_config()

    if path.is_file():
        df = _read_table(path, cfg)
        cols = {str(c).strip().lower(): c for c in df.columns}
        if "rms" in cols and any(k in cols for k in ("bpfo", "bpfi", "bsf", "ftf")):
            # Already a reduced feature table.
            out = df.rename(columns={v: k for k, v in cols.items()})
            if "timestamp" not in out.columns:
                tcol = _pick_column(list(out.columns), _TIME_HINTS)
                if tcol is not None:
                    out = out.rename(columns={tcol: "timestamp"})
                else:
                    out["timestamp"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(
                        np.arange(len(out)) * cfg.acquisition_interval_s, unit="s")
            if "rpm" not in out.columns:
                out["rpm"] = cfg.rpm
            return out
        values = _vibration_series(df, cfg)
        if cfg.snapshot_seconds:
            rows = _snapshot_rows_from_long_record(values, cfg, pd.Timestamp("2000-01-01"))
        else:
            row = {"timestamp": pd.Timestamp("2000-01-01"), "acquisition": 0, "rpm": cfg.rpm}
            row.update(waveform_features(values, fcfg))
            row.update(_temperatures(df, cfg))
            rows = [row]
        return pd.DataFrame(rows)

    files = discover_acquisitions(path, cfg)
    if not files:
        raise FileNotFoundError(f"no acquisition files under {path} matching {cfg.file_glob!r}")

    rows: list[dict] = []
    for i, f in enumerate(files):
        table = _read_table(f, cfg)
        values = _vibration_series(table, cfg)
        ts = timestamp_from_name(f.name)
        row = {
            "acquisition": i,
            "source_file": f.name,
            "rpm": cfg.rpm,
        }
        row["timestamp"] = ts if ts is not None else pd.NaT
        row.update(waveform_features(values, fcfg))
        row.update(_temperatures(table, cfg))
        rows.append(row)
        if progress and i % 50 == 0:
            print(f"  ... {i}/{len(files)} acquisitions", flush=True)

    out = pd.DataFrame(rows)
    if out["timestamp"].isna().any():
        # No usable timestamps in the filenames: synthesise a uniform cadence.
        out["timestamp"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(
            out["acquisition"] * cfg.acquisition_interval_s, unit="s")
    return out.sort_values("timestamp").reset_index(drop=True)


def failure_index(df: pd.DataFrame, temp_limit_c: float = 85.0,
                  rms_limit: float = 9.0) -> int:
    """Row at which the rig's destruction criteria are first met.

    The KAIST test terminates on bearing temperature ~85 C or vibration
    ~9 m/s^2; if neither is recorded the last row is taken as the failure point,
    since the record ends at seizure by construction.
    """
    hit = np.zeros(len(df), dtype=bool)
    for col in ("temp1", "temp2"):
        if col in df.columns:
            hit |= pd.to_numeric(df[col], errors="coerce").to_numpy(float) >= temp_limit_c
    if "rms" in df.columns:
        hit |= pd.to_numeric(df["rms"], errors="coerce").to_numpy(float) >= rms_limit
    idx = np.flatnonzero(hit)
    return int(idx[0]) if idx.size else int(len(df) - 1)
