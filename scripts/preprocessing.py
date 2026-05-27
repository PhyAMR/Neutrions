"""
preprocess.py
─────────────────────────────────────────────────────────────────────────────
Preprocesses raw PET/detector event data into paired-event feature sets.

Reads from  : data/raw/training/{events,truth}.parquet
Writes to   : data/processed/{total_features,total_labels,
                               events_features,events_labels}.parquet

Usage
-----
Standalone:
    python preprocess.py

From another script:
    from preprocess import run
    run()                          # uses defaults
    run(force=True)                # re-generate even if outputs exist
    run(raw_dir="my/raw/path")     # override input path
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR       = Path("data/raw/training")
PROCESSED_DIR = Path("data/processed")

RAW_FILES = {
    "events": RAW_DIR / "events.parquet",
    "truth":  RAW_DIR / "truth.parquet",
}

PROCESSED_FILES = {
    "total_features":  PROCESSED_DIR / "total_features.parquet",
    "total_labels":    PROCESSED_DIR / "total_labels.parquet",
    "events_features": PROCESSED_DIR / "events_features.parquet",
    "events_labels":   PROCESSED_DIR / "events_labels.parquet",
}

COLUMN_NAMES = [
    "time", "energy", "x", "y", "z",
    "label", "pair_id", "label_name",
    "r", "r2", "phi", "distance",
]

#: IBD_p and IBD_d are merged into a single 'detected' class before pairing.
MERGE_LABEL_NAME_MAP: dict[str, str] = {"IBD_p": "detected", "IBD_d": "detected"}
MERGE_LABEL_ID_MAP:   dict[int, int] = {1: 12, 2: 12}   # IBD_p=1, IBD_d=2 → detected=12

LABEL_COLS_TOTAL  = ["label", "pair_id", "label_name"]
PAIR_THRESHOLD_NS = 10 * 200_000   # 2 ms expressed in nanoseconds

#: Number of completed pairs accumulated in memory before flushing to disk.
#: Tune upward for fewer I/O round-trips, downward to reduce peak RAM.
FLUSH_EVERY = 500_000

#: Only delta features and label columns are kept in the events output.
#: Every other column (_1 / _2 raw values) is discarded after pairing.
EVENTS_DELTA_COLS = [
    "delta_t", "delta_energy",
    "delta_x", "delta_y", "delta_z",
    "delta_r", "delta_r2", "delta_phi", "delta_distance",
]
EVENTS_LABEL_COLS = ["label_1", "pair_id_1", "label_name_1",
                     "label_2", "pair_id_2", "label_name_2"]

#: Union used for the dropna guard — all of these must be non-null to keep a row.
REQUIRED_EVENT_COLS = EVENTS_DELTA_COLS + EVENTS_LABEL_COLS

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(script_path: Path = Path(__file__)) -> logging.Logger:
    """Configure logging for *script_path*.

    - Console  : INFO and above, streamed to stdout.
    - File     : INFO and above, written to ``../logs/<stem>.log``
                 (one level above the script directory), overwritten each run.

    The logger is named after the script stem so log lines are always
    attributable to the file that generated them even when multiple modules
    log to the same file handler.
    """
    log_dir  = script_path.resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_path.stem}.log"

    fmt     = logging.Formatter(
        fmt="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger  = logging.getLogger(script_path.stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()          # avoid duplicate handlers on re-import

    file_h  = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_h.setFormatter(fmt)

    console_h = logging.StreamHandler()
    console_h.setFormatter(fmt)

    logger.addHandler(file_h)
    logger.addHandler(console_h)
    return logger


log = _setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _processed_outputs_exist() -> bool:
    """Return True only when every expected output file is present."""
    return all(p.exists() for p in PROCESSED_FILES.values())


def _validate_raw_inputs(raw_dir: Path) -> None:
    """Raise FileNotFoundError if any raw input file is missing."""
    for name, path in RAW_FILES.items():
        full = raw_dir / path.name
        if not full.exists():
            raise FileNotFoundError(
                f"Raw input not found: {full}\n"
                "Ensure 'data/raw/training/' contains events.parquet and truth.parquet."
            )


def _load_raw(raw_dir: Path) -> pd.DataFrame:
    """Load, merge, and engineer base features from raw parquet files."""
    log.info("Loading raw data from %s …", raw_dir)

    events = pq.read_table(raw_dir / "events.parquet").to_pandas()
    truth  = pq.read_table(raw_dir / "truth.parquet").to_pandas()

    df = pd.concat([events, truth], axis=1)

    # Rename only the 8 raw columns (events: 5, truth: 3) that exist at this point
    raw_col_names = [c for c in COLUMN_NAMES if c not in ("r", "r2", "phi", "distance")]
    df.columns = raw_col_names

    # Polar / spherical coordinate features (appended as new columns)
    df["r"]        = np.hypot(df["x"], df["y"])
    df["r2"]       = df["r"] ** 2
    df["phi"]      = np.arctan2(df["y"], df["x"]) % (2 * np.pi)
    df["distance"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)

    df = df.sort_values(["pair_id", "time"]).reset_index(drop=True)

    log.info("Raw data loaded: %d rows, %d columns.", *df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Label merging
# ─────────────────────────────────────────────────────────────────────────────

def _relabel(df: pd.DataFrame) -> pd.DataFrame:
    """Merge IBD_p and IBD_d into a single 'detected' class.

    Both the human-readable column (``label_name``) and the numeric column
    (``label``) are remapped according to ``MERGE_LABEL_NAME_MAP`` and
    ``MERGE_LABEL_ID_MAP``.  ``radio`` (id=3) and ``pair_id=-1`` are unchanged.

    After this step the dataset has exactly two classes:

    +-----------+-------+
    | label_name| label |
    +===========+=======+
    | radio     |   3   |
    +-----------+-------+
    | detected  |  12   |
    +-----------+-------+
    """
    df = df.copy()
    df["label_name"] = df["label_name"].replace(MERGE_LABEL_NAME_MAP)
    df["label"]      = df["label"].replace(MERGE_LABEL_ID_MAP)

    unique_names = sorted(df["label_name"].unique())
    unique_ids   = sorted(df["label"].unique())
    log.info(
        "Labels after merging — names: %s  ids: %s",
        unique_names, unique_ids,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pair-finding logic
# ─────────────────────────────────────────────────────────────────────────────

def _find_forward_pairs(
    df:         pd.DataFrame,
    threshold:  int  = PAIR_THRESHOLD_NS,
    flush_every: int = FLUSH_EVERY,
    out_path:   Path = PROCESSED_DIR / "_pairs_tmp.parquet",
) -> Path:
    """Sliding-window forward-pair scan with streaming output.

    Algorithm
    ---------
    The DataFrame is sorted by time once.  Two integer pointers walk it:

    * ``i`` — the "anchor" event (left edge of the window).
    * ``j`` — advances rightward as long as ``t[j] - t[i] <= threshold``.

    Every index in ``[i+1, j]`` is a valid partner for ``i``.  When ``i``
    increments, ``j`` never resets — it can only stay or move right — so the
    total work is O(n + total_pairs), with no cross-join and no missed
    boundary pairs.

    Memory management
    -----------------
    Pairs are accumulated in plain Python lists and flushed to a Parquet file
    (via ``pyarrow.ParquetWriter``) every ``flush_every`` rows.  Peak RAM is
    proportional to ``flush_every``, not to the dataset size.

    Parameters
    ----------
    df          : Full event DataFrame, sorted or unsorted.
    threshold   : Maximum ``t₂ - t₁`` in nanoseconds for a valid pair.
    flush_every : Number of pairs to buffer before writing to disk.
    out_path    : Temporary parquet file that receives streaming output.

    Returns
    -------
    Path to the written parquet file.
    """
    df = df.sort_values("time").reset_index(drop=True)
    n  = len(df)

    numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_delta = [c for c in numeric_cols if c != "time"]
    all_cols      = df.columns.tolist()

    # ── Extract every column as a numpy array up front ───────────────────────
    # This avoids holding a list-of-dicts (which repeats key strings n times)
    # and avoids per-row df.iloc[] pandas overhead in the inner loop.
    # Total cost: one numpy array per column, ~same memory as the DataFrame.
    arrays: dict[str, np.ndarray] = {c: df[c].to_numpy() for c in all_cols}
    times  = arrays["time"]

    # Column names for the output table: <col>_1, <col>_2, delta_t, delta_*
    out_col_names = (
        [f"{c}_1" for c in all_cols]
        + [f"{c}_2" for c in all_cols]
        + ["delta_t"]
        + [f"delta_{c}" for c in cols_to_delta]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "Sliding-window pair scan — %d events, threshold=%d ns, flush every %d pairs …",
        n, threshold, flush_every,
    )
    t0 = time.perf_counter()

    writer:      pq.ParquetWriter | None = None
    total_pairs: int                     = 0
    j:           int                     = 0          # right pointer (never resets)

    # Per-column output buffers — one plain Python list per column.
    # Only flush_every scalars live in RAM at once per column.
    buf: dict[str, list] = {name: [] for name in out_col_names}

    def _flush() -> None:
        """Write current buffers as one Parquet row group, then clear."""
        nonlocal writer
        table = pa.table({name: pa.array(lst) for name, lst in buf.items()})
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        for lst in buf.values():
            lst.clear()

    for i in range(n):
        t_i = times[i]

        # Advance j while the next event is still within the threshold
        while j + 1 < n and times[j + 1] - t_i <= threshold:
            j += 1

        # Emit all pairs (i, k) for k in (i, j]
        for k in range(i + 1, j + 1):
            delta_t = int(times[k] - t_i)

            for c in all_cols:
                buf[f"{c}_1"].append(arrays[c][i])
                buf[f"{c}_2"].append(arrays[c][k])
            buf["delta_t"].append(delta_t)
            for c in cols_to_delta:
                buf[f"delta_{c}"].append(arrays[c][k] - arrays[c][i])

            if len(buf["delta_t"]) >= flush_every:
                _flush()
                total_pairs += flush_every
                log.info("  … flushed %d pairs (total so far: %d)", flush_every, total_pairs)

    # Final flush for any remaining pairs
    remainder = len(buf["delta_t"])
    if remainder:
        _flush()
        total_pairs += remainder

    if writer is None:
        log.warning("No valid pairs found within %d ns.", threshold)
        empty = pd.DataFrame(columns=out_col_names)
        empty.to_parquet(out_path, index=False)
    else:
        writer.close()

    elapsed = time.perf_counter() - t0
    log.info("Pairing complete: %d pairs in %.1f s → %s", total_pairs, elapsed, out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main processing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _process(raw_dir: Path) -> None:
    """Full preprocessing pipeline: load → engineer → pair → save."""
    _validate_raw_inputs(raw_dir)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Base feature engineering ──────────────────────────────────────────
    df_total = _load_raw(raw_dir)

    # ── 2. Merge IBD_p / IBD_d → 'detected' (id=12) before pairing ──────────
    df_total = _relabel(df_total)

    label_cols   = LABEL_COLS_TOTAL
    feature_cols = [c for c in df_total.columns if c not in label_cols]

    # ── 3. Forward-pair construction (streams to a tmp parquet) ──────────────
    tmp_path = _find_forward_pairs(df_total, threshold=PAIR_THRESHOLD_NS)

    # ── 4. Load pairs, drop incomplete rows, split features / labels ─────────
    log.info("Loading pairs from %s …", tmp_path)
    paired = pq.read_table(tmp_path).to_pandas()

    if paired.empty:
        log.error("Pair table is empty — aborting without writing outputs.")
        tmp_path.unlink(missing_ok=True)
        sys.exit(1)

    # Keep only the columns we actually need — drop all raw _1/_2 event values.
    df_events = (
        paired
        .dropna(subset=REQUIRED_EVENT_COLS)
        [REQUIRED_EVENT_COLS]
        .copy()
    )
    log.info("After filtering to required columns: %d pairs, %d cols.",
             len(df_events), len(df_events.columns))
    del paired   # free RAM before writing the four outputs

    # ── 5. Persist outputs ───────────────────────────────────────────────────
    outputs = {
        PROCESSED_FILES["total_features"]:  (df_total,  feature_cols),
        PROCESSED_FILES["total_labels"]:    (df_total,  label_cols),
        PROCESSED_FILES["events_features"]: (df_events, EVENTS_DELTA_COLS),
        PROCESSED_FILES["events_labels"]:   (df_events, EVENTS_LABEL_COLS),
    }

    for path, (df, cols) in outputs.items():
        df[cols].to_parquet(path, index=False)
        log.info("Saved %-45s  (%d rows, %d cols)", str(path), len(df), len(cols))

    tmp_path.unlink(missing_ok=True)
    log.info("Preprocessing complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(
    *,
    raw_dir: Optional[str | Path] = None,
    force:   bool = False,
) -> None:
    """
    Entry point for both standalone and programmatic use.

    Parameters
    ----------
    raw_dir : Override the default raw-data directory (RAW_DIR).
    force   : When True, regenerate outputs even if they already exist.
    """
    resolved_raw = Path(raw_dir) if raw_dir else RAW_DIR

    if not force and _processed_outputs_exist():
        log.info(
            "All processed files already exist in '%s'. "
            "Pass force=True (or --force on the CLI) to regenerate.",
            PROCESSED_DIR,
        )
        return

    if force:
        log.info("Force flag set — regenerating processed data.")

    _process(resolved_raw)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> tuple[Path, bool]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess raw detector event data into paired-event features.",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(RAW_DIR),
        help=f"Path to raw input directory (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate outputs even if they already exist.",
    )
    args = parser.parse_args()
    return Path(args.raw_dir), args.force


if __name__ == "__main__":
    raw_dir, force = _parse_args()
    run(raw_dir=raw_dir, force=force)