"""
data_loader.py
─────────────────────────────────────────────────────────────────────────────
Centralised data-loading layer for the Grassi analysis pipeline.

All functions return plain pandas DataFrames and are safe to call from
plotting scripts, notebooks, or any other downstream code.  No processing
or side-effects are performed on import.

Processed data is read from  : data/processed/
Raw data is read from        : data/raw/training/

If processed files are missing, every public loader will call
``preprocess.run()`` automatically before loading, so callers never need to
worry about whether the pipeline has been run.

Usage
-----
    from data_loader import load_total, load_events, load_raw

    df_total  = load_total()                     # features + labels joined
    df_events = load_events()                    # paired-event deltas
    df_raw    = load_raw()                       # unprocessed parquet files

    # Optional filters / sampling
    df_total  = load_total(label="detected", sample=50_000)
    df_events = load_events(columns=["delta_t", "delta_r", "label_name_2"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(script_path: Path = Path(__file__)) -> logging.Logger:
    """Configure logging for *script_path*.

    - Console  : INFO and above, streamed to stdout.
    - File     : INFO and above, written to ``../logs/<stem>.log``
                 (one level above the script directory), overwritten each run.

    The logger is named after the script stem so log lines are always
    attributable to the file that generated them.
    """
    log_dir  = script_path.resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_path.stem}.log"

    fmt    = logging.Formatter(
        fmt="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(script_path.stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_h    = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_h.setFormatter(fmt)
    console_h = logging.StreamHandler()
    console_h.setFormatter(fmt)

    logger.addHandler(file_h)
    logger.addHandler(console_h)
    return logger


log = _setup_logging()

# ─────────────────────────────────────────────────────────────────────────────
# Paths  (mirrors preprocess.py — single source of truth kept there)
# ─────────────────────────────────────────────────────────────────────────────

_RAW_DIR       = Path("data/raw/training")
_PROCESSED_DIR = Path("data/processed")

_PROCESSED_FILES = {
    "total_features":  _PROCESSED_DIR / "total_features.parquet",
    "total_labels":    _PROCESSED_DIR / "total_labels.parquet",
    "events_features": _PROCESSED_DIR / "events_features.parquet",
    "events_labels":   _PROCESSED_DIR / "events_labels.parquet",
}

_RAW_FILES = {
    "events": _RAW_DIR / "events.parquet",
    "truth":  _RAW_DIR / "truth.parquet",
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_processed() -> None:
    """Run the preprocessing pipeline if any output file is missing."""
    missing = [p for p in _PROCESSED_FILES.values() if not p.exists()]
    if missing:
        log.warning(
            "%d processed file(s) missing — running preprocess.run() …",
            len(missing),
        )
        from preprocess import run as _preprocess_run  # local import avoids circular deps
        _preprocess_run()


def _read_parquet(
    path: Path,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Read a single parquet file, optionally projecting a column subset."""
    log.info("Reading %s%s", path, f"  cols={columns}" if columns else "")
    if columns is not None:
        return pq.read_table(path, columns=columns).to_pandas()
    return pq.read_table(path).to_pandas()


def _apply_sample(
    df: pd.DataFrame,
    sample: Optional[int],
    random_state: int = 42,
) -> pd.DataFrame:
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=random_state).reset_index(drop=True)
        log.info("  sampled → %d rows", len(df))
    return df


def _apply_label_filter(
    df: pd.DataFrame,
    label: Optional[str],
    label_col: str = "label_name",
) -> pd.DataFrame:
    if label is not None:
        before = len(df)
        df = df[df[label_col] == label].reset_index(drop=True)
        log.info("  filtered to %s=%r → %d / %d rows", label_col, label, len(df), before)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_total(
    *,
    columns:      Optional[list[str]] = None,
    label:        Optional[str]       = None,
    sample:       Optional[int]       = None,
    random_state: int                 = 42,
) -> pd.DataFrame:
    """Load the full per-event DataFrame (features + labels joined).

    This is the unpaired dataset — one row per raw detector event — with
    engineered coordinates (r, r2, phi, distance) and merged labels
    (radio / detected).

    Parameters
    ----------
    columns      : If given, only these columns are loaded from disk (faster
                   for large files when only a subset is needed).
    label        : Filter to a single ``label_name`` value, e.g. ``"radio"``
                   or ``"detected"``.  Applied after loading.
    sample       : Randomly subsample to at most this many rows.
    random_state : Seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame with columns from total_features + total_labels.
    """
    _ensure_processed()

    features = _read_parquet(_PROCESSED_FILES["total_features"], columns=columns)
    labels   = _read_parquet(_PROCESSED_FILES["total_labels"])

    df = pd.concat([features, labels], axis=1)
    log.info("load_total: %d rows, %d cols", *df.shape)

    df = _apply_label_filter(df, label, label_col="label_name")
    df = _apply_sample(df, sample, random_state)
    return df


def load_events(
    *,
    columns:      Optional[list[str]] = None,
    label:        Optional[str]       = None,
    sample:       Optional[int]       = None,
    random_state: int                 = 42,
) -> pd.DataFrame:
    """Load the paired-event DataFrame (delta features + labels joined).

    Each row represents one forward pair ``(event_i, event_k)`` where
    ``0 < t_k - t_i <= PAIR_THRESHOLD_NS``.  Columns are suffixed ``_1``
    (anchor) and ``_2`` (partner), plus ``delta_*`` for every numeric feature.

    Parameters
    ----------
    columns      : Column projection passed directly to the parquet reader.
    label        : Filter on ``label_name_2`` (the partner event's class).
    sample       : Randomly subsample to at most this many rows.
    random_state : Seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame with columns from events_features + events_labels.
    """
    _ensure_processed()

    features = _read_parquet(_PROCESSED_FILES["events_features"], columns=columns)
    labels   = _read_parquet(_PROCESSED_FILES["events_labels"])

    df = pd.concat([features, labels], axis=1)
    log.info("load_events: %d rows, %d cols", *df.shape)

    df = _apply_label_filter(df, label, label_col="label_name_2")
    df = _apply_sample(df, sample, random_state)
    return df


def load_raw(
    *,
    columns_events: Optional[list[str]] = None,
    columns_truth:  Optional[list[str]] = None,
    sample:         Optional[int]       = None,
    random_state:   int                 = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the unprocessed raw parquet files directly.

    Returns the events and truth tables as separate DataFrames, exactly as
    they appear on disk (no feature engineering, no label merging).

    Parameters
    ----------
    columns_events : Column projection for the events table.
    columns_truth  : Column projection for the truth table.
    sample         : Subsample both tables to the same *n* rows (same seed,
                     so row alignment is preserved).
    random_state   : Seed for reproducible sampling.

    Returns
    -------
    (df_events, df_truth) — two DataFrames aligned by row index.
    """
    for name, path in _RAW_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Raw file not found: {path}\n"
                "Ensure data/raw/training/ contains events.parquet and truth.parquet."
            )

    df_events = _read_parquet(_RAW_FILES["events"], columns=columns_events)
    df_truth  = _read_parquet(_RAW_FILES["truth"],  columns=columns_truth)
    log.info("load_raw: events %s, truth %s", df_events.shape, df_truth.shape)

    if sample is not None and sample < len(df_events):
        idx       = df_events.sample(n=sample, random_state=random_state).index
        df_events = df_events.loc[idx].reset_index(drop=True)
        df_truth  = df_truth.loc[idx].reset_index(drop=True)
        log.info("  sampled → %d rows (both tables)", sample)

    return df_events, df_truth


def load_total_features(
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load only the feature columns of the total dataset (no labels).

    Useful when labels are not needed and avoiding the concat saves memory.
    """
    _ensure_processed()
    df = _read_parquet(_PROCESSED_FILES["total_features"], columns=columns)
    log.info("load_total_features: %d rows, %d cols", *df.shape)
    return df


def load_total_labels() -> pd.DataFrame:
    """Load only the label columns of the total dataset (label, pair_id, label_name)."""
    _ensure_processed()
    df = _read_parquet(_PROCESSED_FILES["total_labels"])
    log.info("load_total_labels: %d rows, %d cols", *df.shape)
    return df


def load_events_features(
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load only the feature columns of the paired-event dataset (no labels)."""
    _ensure_processed()
    df = _read_parquet(_PROCESSED_FILES["events_features"], columns=columns)
    log.info("load_events_features: %d rows, %d cols", *df.shape)
    return df


def load_events_labels() -> pd.DataFrame:
    """Load only the label columns of the paired-event dataset."""
    _ensure_processed()
    df = _read_parquet(_PROCESSED_FILES["events_labels"])
    log.info("load_events_labels: %d rows, %d cols", *df.shape)
    return df


def describe(which: str = "all") -> None:
    """Print a quick summary of available processed files.

    Parameters
    ----------
    which : ``"total"``, ``"events"``, or ``"all"`` (default).
    """
    _ensure_processed()

    targets = {
        "total":  ["total_features",  "total_labels"],
        "events": ["events_features", "events_labels"],
        "all":    list(_PROCESSED_FILES.keys()),
    }.get(which, list(_PROCESSED_FILES.keys()))

    for key in targets:
        path = _PROCESSED_FILES[key]
        meta = pq.read_metadata(path)
        schema = pq.read_schema(path)
        print(
            f"\n{'─' * 60}\n"
            f"  {key:<22}  {path}\n"
            f"  rows     : {meta.num_rows:,}\n"
            f"  row groups: {meta.num_row_groups}\n"
            f"  columns  : {len(schema.names)}\n"
            f"  {schema.names}"
        )