"""
data_loader.py
─────────────────────────────────────────────────────────────────────────────
Centralised data-loading layer for the Grassi analysis pipeline.

All public functions return plain pandas DataFrames or generators of them.
No processing or side-effects are performed on import.

If processed files are missing, every loader calls ``preprocess.run()``
automatically before loading.

Loaders
-------
load_total(...)              → DataFrame   — raw per-event dataset
load_events(...)             → DataFrame   — signal pair dataset (in-window)
load_false_events(...)       → DataFrame   — raises MemoryWarning; use iterator
stream_false_events(...)     → Generator   — one row-group per iteration
stream_events(...)           → Generator   — one row-group per iteration
load_raw(...)                → (DataFrame, DataFrame)

Feature schema (events / false_events)
---------------------------------------
Individual anchor  : t_1  E_1  x_1  y_1  z_1  r_1  r2_1  phi_1  D_1
Individual partner : t_2  E_2  x_2  y_2  z_2  r_2  r2_2  phi_2  D_2
Deltas             : dt   dE   dx   dy   dz   dr   dr2   dphi   dD
Derived            : dR  (= sqrt(dx²+dy²+dz²))
Labels             : pair_id  label_name_1  label_name_2

Usage
-----
    from data_loader import load_total, load_events, stream_false_events

    # Signal pairs — fit in RAM
    df_events = load_events()

    # Accidentals — stream row-group by row-group
    for chunk in stream_false_events():
        do_something(chunk)           # chunk is freed after the loop body

    # With column projection to reduce memory further
    for chunk in stream_false_events(columns=["E_1", "E_2", "dR", "label_name_2"]):
        plot(chunk)
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(script_path: Path = Path(__file__)) -> logging.Logger:
    log_dir  = script_path.resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_path.stem}.log"
    fmt      = logging.Formatter(
        fmt="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(script_path.stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = _setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_RAW_DIR       = Path("data/raw/training")
_PROCESSED_DIR = Path("data/processed")

_PROCESSED_FILES = {
    "total_features":        _PROCESSED_DIR / "total_features.parquet",
    "total_labels":          _PROCESSED_DIR / "total_labels.parquet",
    "events_features":       _PROCESSED_DIR / "events_features.parquet",
    "events_labels":         _PROCESSED_DIR / "events_labels.parquet",
    "false_events_features": _PROCESSED_DIR / "false_events_features.parquet",
    "false_events_labels":   _PROCESSED_DIR / "false_events_labels.parquet",
}

_RAW_FILES = {
    "events": _RAW_DIR / "events.parquet",
    "truth":  _RAW_DIR / "truth.parquet",
}

# ── Known column sets for the paired datasets ─────────────────────────────────
# Used to route a user-supplied column list to the correct parquet file.
# Any column not found in either set is silently ignored.
_FEAT_COLS: frozenset[str] = frozenset([
    "t_1","E_1","x_1","y_1","z_1","r_1","r2_1","phi_1","D_1",
    "t_2","E_2","x_2","y_2","z_2","r_2","r2_2","phi_2","D_2",
    "dt","dE","dx","dy","dz","dr","dr2","dphi","dD","dR",
])
_LABEL_COLS: frozenset[str] = frozenset(["pair_id","label_name_1","label_name_2"])

# Approximate per-row size for a 28-feature + 3-label paired DataFrame.
# Used to warn when load_false_events() would exceed a sensible memory cap.
_BYTES_PER_ROW_PAIRS = (28 + 3) * 8   # 31 cols × 8 bytes ≈ 248 bytes / row
_WARN_THRESHOLD_GB   = 2.0


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
        from preprocess import run as _preprocess_run
        _preprocess_run()


def _read_parquet(
    path: Path,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    log.info("Reading %s%s", path, f"  cols={columns}" if columns else "")
    return pq.read_table(path, columns=columns).to_pandas()


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
        log.info(
            "  filtered to %s=%r → %d / %d rows",
            label_col, label, len(df), before,
        )
    return df


def _split_columns(
    columns: Optional[list[str]],
) -> tuple[Optional[list[str]], Optional[list[str]]]:
    """Split a user-supplied column list into (feat_cols, label_cols).

    Returns ``None`` for a side when no columns from that file were requested,
    which tells ``pq.read_table`` to load all columns from that file.
    If ``columns`` is ``None``, both sides are ``None`` (load everything).
    """
    if columns is None:
        return None, None
    feat_cols  = [c for c in columns if c in _FEAT_COLS]  or None
    label_cols = [c for c in columns if c in _LABEL_COLS] or None
    unknown    = [c for c in columns if c not in _FEAT_COLS and c not in _LABEL_COLS]
    if unknown:
        log.warning("Unknown columns ignored (not in features or labels): %s", unknown)
    return feat_cols, label_cols


def _join_feat_label_chunk(
    feat_chunk: pd.DataFrame,
    label_chunk: pd.DataFrame,
    label: Optional[str] = None,
    label_col: str = "label_name_2",
) -> pd.DataFrame:
    """Join a pre-projected feature chunk with its label chunk.

    Column projection is handled upstream (in ``_split_columns``) so both
    chunks already contain only the requested columns before this call.
    """
    df = pd.concat([feat_chunk, label_chunk], axis=1)
    if label is not None:
        df = df[df[label_col] == label].reset_index(drop=True)
    return df


def _stream_paired_file(
    feat_key:  str,
    label_key: str,
    columns:   Optional[list[str]] = None,
    label:     Optional[str]       = None,
    label_col: str                 = "label_name_2",
) -> Generator[pd.DataFrame, None, None]:
    """Core streaming generator: yield one joined row-group at a time.

    Both the feature and label parquet files share the same row-group
    boundaries (they were written by the same ParquetWriter in preprocess),
    so row-group i of features aligns exactly with row-group i of labels.

    Peak RAM = size of one row-group (~FLUSH_EVERY rows) rather than the
    full file.

    Parameters
    ----------
    feat_key  : Key in ``_PROCESSED_FILES`` for the features file.
    label_key : Key in ``_PROCESSED_FILES`` for the labels file.
    columns   : Optional column projection applied to each chunk.
    label     : Optional label filter applied to each chunk.
    label_col : Column to filter on.

    Yields
    ------
    pd.DataFrame — one row-group worth of joined features + labels.
    """
    _ensure_processed()

    feat_pf  = pq.ParquetFile(_PROCESSED_FILES[feat_key])
    label_pf = pq.ParquetFile(_PROCESSED_FILES[label_key])

    n_rg = feat_pf.metadata.num_row_groups
    log.info(
        "Streaming %s (%d row-groups, %s rows total) …",
        feat_key,
        n_rg,
        f"{feat_pf.metadata.num_rows:,}",
    )

    feat_cols, label_cols = _split_columns(columns)

    for rg in range(n_rg):
        feat_chunk  = feat_pf.read_row_group(rg, columns=feat_cols).to_pandas()
        label_chunk = label_pf.read_row_group(rg, columns=label_cols).to_pandas()

        chunk = _join_feat_label_chunk(
            feat_chunk, label_chunk,
            label=label, label_col=label_col,
        )

        if chunk.empty:
            continue

        log.info("  row-group %d/%d → %d rows", rg + 1, n_rg, len(chunk))
        yield chunk
        # chunk goes out of scope here; memory is released before the next
        # row-group is read.


# ─────────────────────────────────────────────────────────────────────────────
# Public API — full-load functions
# ─────────────────────────────────────────────────────────────────────────────

def load_total(
    *,
    columns:      Optional[list[str]] = None,
    label:        Optional[str]       = None,
    sample:       Optional[int]       = None,
    random_state: int                 = 42,
) -> pd.DataFrame:
    """Load the full per-event DataFrame (features + labels joined).

    One row per raw detector event with engineered coordinates and merged
    labels (radio / detected).

    Parameters
    ----------
    columns      : Column projection; loaded from disk if given.
    label        : Filter to a single ``label_name`` value.
    sample       : Subsample to at most this many rows.
    random_state : Seed for reproducible sampling.
    """
    _ensure_processed()
    features = _read_parquet(_PROCESSED_FILES["total_features"], columns=columns)
    labels   = _read_parquet(_PROCESSED_FILES["total_labels"])
    df       = pd.concat([features, labels], axis=1)
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
    """Load the full in-window paired-event DataFrame.

    Each row is one prompt-delayed pair with individual measurements for
    both events, delta features, and the derived spatial separation dR.

    Parameters
    ----------
    columns      : Column projection; loaded from disk if given.
    label        : Filter on ``label_name_2`` (partner class).
    sample       : Subsample to at most this many rows.
    random_state : Seed for reproducible sampling.
    """
    _ensure_processed()
    feat_cols, label_cols = _split_columns(columns)
    features = _read_parquet(_PROCESSED_FILES["events_features"], columns=feat_cols)
    labels   = _read_parquet(_PROCESSED_FILES["events_labels"],   columns=label_cols)
    df       = pd.concat([features, labels], axis=1)
    log.info("load_events: %d rows, %d cols", *df.shape)
    df = _apply_label_filter(df, label, label_col="label_name_2")
    df = _apply_sample(df, sample, random_state)
    return df


def load_false_events(
    *,
    columns:      Optional[list[str]] = None,
    label:        Optional[str]       = None,
    sample:       Optional[int]       = None,
    random_state: int                 = 42,
) -> pd.DataFrame:
    """Load the accidental (off-time-window) paired-event DataFrame.

    .. warning::
        This file can be very large.  A memory check is performed before
        loading: if the estimated in-memory size exceeds
        ``_WARN_THRESHOLD_GB`` GB and no ``sample`` cap is set,
        a ``ResourceWarning`` is raised and you are directed to
        ``stream_false_events()`` instead.

        For most use cases prefer ``stream_false_events()``.

    Parameters
    ----------
    columns      : Column projection applied before the size check.
    label        : Filter on ``label_name_2``.
    sample       : Hard cap on rows loaded.  Strongly recommended.
    random_state : Seed for reproducible sampling.
    """
    _ensure_processed()

    # ── Memory guard ──────────────────────────────────────────────────────────
    meta        = pq.read_metadata(_PROCESSED_FILES["false_events_features"])
    total_rows  = meta.num_rows
    n_cols      = (
        len(columns)
        if columns is not None
        else pq.read_schema(_PROCESSED_FILES["false_events_features"]).names.__len__()
    )
    est_gb = total_rows * n_cols * 8 / 1e9

    if sample is None and est_gb > _WARN_THRESHOLD_GB:
        raise ResourceWarning(
            f"Loading false_events would require ~{est_gb:.1f} GB "
            f"({total_rows:,} rows × {n_cols} cols).\n"
            f"Either pass sample=<n> to load a random subset, or use\n"
            f"  stream_false_events()  to iterate row-group by row-group."
        )

    if est_gb > _WARN_THRESHOLD_GB * 0.5:
        warnings.warn(
            f"false_events is large (~{est_gb:.1f} GB estimated). "
            "Consider stream_false_events() if memory is limited.",
            ResourceWarning,
            stacklevel=2,
        )

    feat_cols, label_cols = _split_columns(columns)
    features = _read_parquet(_PROCESSED_FILES["false_events_features"], columns=feat_cols)
    labels   = _read_parquet(_PROCESSED_FILES["false_events_labels"],   columns=label_cols)
    df       = pd.concat([features, labels], axis=1)
    log.info("load_false_events: %d rows, %d cols", *df.shape)
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

    Returns events and truth as separate DataFrames, exactly as they appear
    on disk — no feature engineering, no label merging.

    Returns
    -------
    (df_events, df_truth) — row-aligned.
    """
    for path in _RAW_FILES.values():
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


# ─────────────────────────────────────────────────────────────────────────────
# Public API — streaming iterators
# ─────────────────────────────────────────────────────────────────────────────

def stream_false_events(
    *,
    columns:   Optional[list[str]] = None,
    label:     Optional[str]       = None,
) -> Generator[pd.DataFrame, None, None]:
    """Stream the accidental (off-time-window) dataset row-group by row-group.

    Each iteration yields one row-group as a joined features + labels
    DataFrame and immediately releases it from memory before the next
    row-group is read.  Peak RAM is therefore bounded by one row-group
    (~``FLUSH_EVERY`` rows from preprocessing, default 500 000), not by
    the full file size.

    The natural class ratio of the file is preserved — no resampling is
    applied within each chunk.

    Parameters
    ----------
    columns : Optional column projection applied to every chunk.
              Use this to load only the columns you need, e.g.
              ``columns=["E_1", "E_2", "dR", "label_name_2"]`` cuts
              memory per chunk by ~85%.
    label   : Optional filter on ``label_name_2`` applied to every chunk.

    Yields
    ------
    pd.DataFrame — one row-group of joined features + labels.

    Examples
    --------
    Plot E_1 spectrum from the accidental sample without loading everything:

        for chunk in stream_false_events(columns=["E_1", "label_name_2"]):
            hist, edges = np.histogram(chunk["E_1"], bins=100, range=(0, 10))
            accumulate(hist)   # accumulate across chunks

    Collect the first 1 M rows and stop early:

        rows = []
        for chunk in stream_false_events():
            rows.append(chunk)
            if sum(len(r) for r in rows) >= 1_000_000:
                break
        df = pd.concat(rows, ignore_index=True)
    """
    yield from _stream_paired_file(
        feat_key="false_events_features",
        label_key="false_events_labels",
        columns=columns,
        label=label,
        label_col="label_name_2",
    )


def stream_events(
    *,
    columns: Optional[list[str]] = None,
    label:   Optional[str]       = None,
) -> Generator[pd.DataFrame, None, None]:
    """Stream the in-window signal pairs row-group by row-group.

    Identical interface to ``stream_false_events`` but for the signal
    dataset.  Useful when the signal pairs file is also too large to
    load at once.

    Parameters
    ----------
    columns : Optional column projection applied to every chunk.
    label   : Optional filter on ``label_name_2`` applied to every chunk.

    Yields
    ------
    pd.DataFrame — one row-group of joined features + labels.
    """
    yield from _stream_paired_file(
        feat_key="events_features",
        label_key="events_labels",
        columns=columns,
        label=label,
        label_col="label_name_2",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API — split-file loaders and describe
# ─────────────────────────────────────────────────────────────────────────────

def load_total_features(columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Load only the feature columns of the total dataset (no labels)."""
    _ensure_processed()
    df = _read_parquet(_PROCESSED_FILES["total_features"], columns=columns)
    log.info("load_total_features: %d rows, %d cols", *df.shape)
    return df


def load_total_labels() -> pd.DataFrame:
    """Load only the label columns of the total dataset."""
    _ensure_processed()
    df = _read_parquet(_PROCESSED_FILES["total_labels"])
    log.info("load_total_labels: %d rows, %d cols", *df.shape)
    return df


def load_events_features(columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Load only the feature columns of the signal pairs dataset."""
    _ensure_processed()
    df = _read_parquet(_PROCESSED_FILES["events_features"], columns=columns)
    log.info("load_events_features: %d rows, %d cols", *df.shape)
    return df


def load_events_labels() -> pd.DataFrame:
    """Load only the label columns of the signal pairs dataset."""
    _ensure_processed()
    df = _read_parquet(_PROCESSED_FILES["events_labels"])
    log.info("load_events_labels: %d rows, %d cols", *df.shape)
    return df


def describe(which: str = "all") -> None:
    """Print a concise summary of available processed files.

    Parameters
    ----------
    which : ``"total"``, ``"events"``, ``"false_events"``, or ``"all"``.
    """
    _ensure_processed()

    targets = {
        "total":        ["total_features",        "total_labels"],
        "events":       ["events_features",        "events_labels"],
        "false_events": ["false_events_features",  "false_events_labels"],
        "all":          list(_PROCESSED_FILES.keys()),
    }.get(which, list(_PROCESSED_FILES.keys()))

    for key in targets:
        path   = _PROCESSED_FILES[key]
        meta   = pq.read_metadata(path)
        schema = pq.read_schema(path)
        size_gb = path.stat().st_size / 1e9
        print(
            f"\n{'─' * 60}\n"
            f"  {key:<26}  {path}\n"
            f"  rows       : {meta.num_rows:,}\n"
            f"  row groups : {meta.num_row_groups}\n"
            f"  columns    : {len(schema.names)}\n"
            f"  disk size  : {size_gb:.2f} GB\n"
            f"  {schema.names}"
        )