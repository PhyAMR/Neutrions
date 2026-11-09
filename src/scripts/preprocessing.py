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

# Still not created files but rutes to store the processed data.
PROCESSED_FILES = {
    "total_features":         PROCESSED_DIR / "total_features.parquet",
    "total_labels":           PROCESSED_DIR / "total_labels.parquet",
    "events_features":        PROCESSED_DIR / "events_features.parquet",
    "events_labels":          PROCESSED_DIR / "events_labels.parquet",
    "false_events_features":  PROCESSED_DIR / "false_events_features.parquet",
    "false_events_labels":    PROCESSED_DIR / "false_events_labels.parquet",
    "parameters":    PROCESSED_DIR / "parameters.npz",
}

#: Required columns expected in the input events.parquet file.
#: The r, r2, phi, and distance columns are computed later in _load_raw()
# from the x, y, and z coordinates.
EVENT_COLS_TOTAL = [
    "time", "energy", "x", "y", "z",
]

LABEL_COLS_TOTAL  = ["pair_id", "label", "label_name"]
PAIR_THRESHOLD_NS = 10 * 200_000   # 2 ms expressed in nanoseconds
DETECTOR_RADIUS_M = 1.5 # Radius of the instrument in meters 
DETECTOR_HALF_HEIGHT_M = 1.55 #height of the instrument in meters 
#: Number of completed pairs accumulated in memory before flushing to disk.
FLUSH_EVERY = 500_000

# ── Physics-notation column mapping ──────────────────────────────────────────
# Raw column name → physics label used in output parquet files.
# Applied once in _rename_pair_columns() after the pair scan.
_RAW_TO_PHYS: dict[str, str] = {
    "time":     "t",
    "energy":   "E",
    "x":        "x",
    "y":        "y",
    "z":        "z",
    "r":        "r",
    "r2":       "r2",
    "phi":      "phi",
    "distance": "D",
}

# Individual per-event columns kept in the output (both _1 and _2 variants).
# Named t_1, E_1, x_1 … and t_2, E_2, x_2 …
_INDIV_BASE = list(_RAW_TO_PHYS.values())   # ['t','E','x','y','z','r','r2','phi','D']

EVENTS_INDIV_COLS = (
    [f"{p}_1" for p in _INDIV_BASE] +
    [f"{p}_2" for p in _INDIV_BASE] + 
    ["dwall_1", "dwall_2"] # Wall distance for each event in the pair.
)

# Delta columns: Δt, ΔE, Δx, Δy, Δz, Δr, Δr2, Δphi, ΔD + derived ΔR
EVENTS_DELTA_COLS = [f"d{p}" for p in _INDIV_BASE] + ["dR"]
# dR = sqrt(dx²+dy²+dz²) — 3D spatial separation, key IBD discriminant
# Note that Δr is the difference in horizontal distance from the detector center, 
# while ΔR is the true 3D separation.

# Label columns kept in the output: We store all the labels that indicate the nature of
# the event
EVENTS_LABEL_COLS = ["event_id_1", "event_id_2", "pair_id_1", "pair_id_2", "label_1", "label_2", "label_name_1", "label_name_2", "is_ibd", "false_ibd", "pair_type"]

# Temporal-window context and event-count metadata saved for each pair.
# These columns do not define the target label or filter rows; they are
# available for analysis and could be using for training if selected in training.yaml.
# - k_window: index of the time window containing the pair (0 for on-time pairs).
# - dt_shifted: pair time difference relative to the start of that window.
# - n_before: events within one threshold interval before the anchor event.
# - n_on: events in the immediate on-time window after the anchor event.
# - n_gap: events in the on-time and earlier shifted windows before this pair's window.
# - n_in_window: events in the same time window as the pair.
# - is_dc: flag set when n_before == 0, n_gap == 0, and n_in_window == 1.
EVENTS_COUNTS_COLS = ["k_window", "dt_shifted", "n_before", "n_on", "n_gap",
                     "n_in_window", "is_dc"]


EVENTS_FEATURE_COLS = EVENTS_INDIV_COLS + EVENTS_DELTA_COLS + EVENTS_COUNTS_COLS

# All columns that must be non-null to keep a paired row
REQUIRED_EVENT_COLS = EVENTS_LABEL_COLS + EVENTS_FEATURE_COLS 

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
    """Function to check the raw data before using it

    Raises
    ------ 
    FileNotFoundError if files are missing 
    ValueError if columns are missing or their name mismatch
    
    """
    required = {"events": EVENT_COLS_TOTAL, "truth": LABEL_COLS_TOTAL}

    for name, path in RAW_FILES.items():
        full = raw_dir / path.name
        if not full.exists():
            raise FileNotFoundError(
                f"Raw input not found: {full}\n"
                f"Ensure {raw_dir} contains {path.name}"
            )
        missing = [x for x in required[name] if x not in pq.read_schema(full).names]
        found = [x for x in pq.read_schema(full).names if x not in required[name]]
        if missing: 
            raise ValueError("The following columns mismatch with the expected "
                f"structure of {full}: found: {found} \n missing: {missing}")
       


def _load_raw(raw_dir: Path) -> pd.DataFrame:
    """Load, merge, and engineer base features from raw parquet files.


    Returns
    ------- 
    df: pd.DataFrame
        A combinned DataFrame with the events and truth values for the training
    Raises
    ------ 
    FileNotFoundError if files are missing 
    ValueError if columns are missing or their name mismatch
    """
    
    log.info("Loading raw data from %s …", raw_dir)

    events = pq.read_table(raw_dir / "events.parquet").to_pandas()
    truth  = pq.read_table(raw_dir / "truth.parquet").to_pandas()

    df = pd.concat([events, truth], axis=1)
    df['event_id'] = df.index

    # Polar / spherical coordinate features (appended as new columns)
    df["r"]        = np.hypot(df["x"], df["y"])
    df["r2"]       = df["r"] ** 2
    df["phi"]      = np.arctan2(df["y"], df["x"]) % (2 * np.pi)
    df["distance"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)

    df = df.sort_values(["time", "label"], kind="stable").reset_index(drop=True)

    log.info("Raw data loaded: %d rows, %d columns.", *df.shape)
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
    df = df.sort_values(["time","label"],kind='stable').reset_index(drop=True)
    n  = len(df)

    numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_delta = [c for c in numeric_cols if c not in ["time", "label", "pair_id", "event_id"]]
    all_cols      = df.columns.tolist()

    # ── Extract every column as a numpy array up front ───────────────────────
    # This avoids holding a list-of-dicts (which repeats key strings n times)
    # and avoids per-row df.iloc[] pandas overhead in the inner loop.
    # Total cost: one numpy array per column, ~same memory as the DataFrame.
    arrays: dict[str, np.ndarray] = {c: df[c].to_numpy() for c in all_cols}
    times  = arrays["time"]

    # Column names for the raw tmp output: <col>_1, <col>_2, delta_t, delta_*
    # These are internal names — physics renaming happens in _project_pairs_chunked.
    out_col_names = (
        [f"{c}_1" for c in all_cols]
        + [f"{c}_2" for c in all_cols]
        + ["delta_t"]
        + [f"delta_{c}" for c in cols_to_delta]
        + ["n_in_window","n_before"]
        + ["k_window","dt_shifted","n_gap","n_on"] # Columns so forward and of window functions produce the same structure
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
    b:           int                     = 0 
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
        # Move b to the first event at most T before t_i (older events are skipped) 
        while b<i and t_i -times[b] > threshold:
            b += 1  
                    
        # Advance j while the next event is still within the threshold
        while j + 1 < n and times[j + 1] - t_i <= threshold:
            j += 1
        n_before = i-b 
        n_in_window = j-i 
        
        n_on = n_in_window

        # Emit all pairs (i, k) for k in (i, j]
        for k in range(i + 1, j + 1):
            delta_t = int(times[k] - t_i)

            k_window = 0 
            
            dt_shifted = delta_t 

            n_gap = 0 

            
            for c in all_cols:
                buf[f"{c}_1"].append(arrays[c][i])
                buf[f"{c}_2"].append(arrays[c][k])
            buf["delta_t"].append(delta_t)
            buf["n_in_window"].append(n_in_window)
            buf["n_before"].append(n_before)
            buf["k_window"].append(k_window)
            buf["dt_shifted"].append(dt_shifted)
            buf["n_gap"].append(n_gap)
            buf["n_on"].append(n_on)
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
# False-pair finding logic
# ─────────────────────────────────────────────────────────────────────────────

def _find_false_pairs(
    df:          pd.DataFrame,
    threshold:   int  = PAIR_THRESHOLD_NS,
    flush_every: int  = FLUSH_EVERY,
    out_path:    Path = PROCESSED_DIR / "_false_pairs_tmp.parquet",
    n_off:       int = 10
) -> Path:
    """Sliding-window false-pair scan — events guaranteed to be uncorrelated.

    For each anchor event ``i``, valid partners are events whose time separation
    from ``i`` is strictly **beyond** the coincidence threshold, making physical
    correlation impossible.

   
    The output schema is identical to ``_find_forward_pairs`` so both datasets
    can be fed to the same plotting and modelling code.

    Parameters
    ----------
    df          : Full event DataFrame (sorted or unsorted).
    threshold   : Coincidence window in nanoseconds (same as forward pairs).
    flush_every : Pairs buffered before each disk flush.
    out_path    : Temporary parquet file for streaming output.
    n_off       : Number of threshold units so the windows are from (t_i + k*T, t_i + (k+1)*T] k = 1, .., n_off
    K_window    : Gives as the time window where the pairs are found so we can compute the
    counting error later
    dt_shifted  : Gives the time difference with respect to the last window to compute
    the counting error

    Returns
    -------
    Path to the written parquet file.
    """

    df = df.sort_values(["time","label"],kind='stable').reset_index(drop=True)
    n  = len(df)

    numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_delta = [c for c in numeric_cols if c not in ["time","label","pair_id", "event_id"]]
    all_cols      = df.columns.tolist()

    arrays: dict[str, np.ndarray] = {c: df[c].to_numpy() for c in all_cols}
    times  = arrays["time"]

    out_col_names = (
        [f"{c}_1" for c in all_cols]
        + [f"{c}_2" for c in all_cols]
        + ["delta_t"]
        + [f"delta_{c}" for c in cols_to_delta]
        + ["k_window", "dt_shifted", "n_before", "n_gap", "n_on", "n_in_window"]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "False-pair scan n_off=%d — %d events, threshold=%d ns, flush every %d pairs …",
         n_off, n, threshold, flush_every,
    )
    t0 = time.perf_counter()

    writer:      pq.ParquetWriter | None = None
    total_pairs: int                     = 0

    buf: dict[str, list] = {name: [] for name in out_col_names}

    def _flush() -> None:
        nonlocal writer
        table = pa.table({name: pa.array(lst) for name, lst in buf.items()})
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        for lst in buf.values():
            lst.clear()

    # Two pointers that track the valid partner window for each anchor i:
    #   j_lo  — first index where t_k - t_i  > threshold       (window start)
     #   j_hi  — last index where t_k - t_i <= (n_off + 1) * threshold  (end of the last window)
    j_lo: int = 0
    b: int = 0
    for i in range(n):
        t_i      = times[i]
       # lo_bound = t_i + threshold          # exclusive lower bound
        hi_bound = t_i + (n_off+1) * threshold      # inclusive upper bound 

        while b<i and t_i -times[b] > threshold:
            b += 1  
                
        # Advance j_lo to the first index strictly beyond the threshold.
        # j_lo never resets — it can only move right.
        while j_lo < n and times[j_lo] - t_i <= threshold:
            j_lo += 1

        if j_lo >= n:
            break   # no events beyond threshold remain for any future i

        if times[j_lo] > hi_bound:
            continue # No events for this off-window window
        j_hi = j_lo
        while j_hi + 1 < n and times[j_hi + 1] <= hi_bound:
            j_hi += 1

        
        partner_range = range(j_lo, j_hi + 1)
        n_before = i - b
        n_on = j_lo -i - 1
        counts = [0]*(n_off + 1)
        for k in partner_range:
            delta_t = int(times[k] - t_i)
             
            k_window = int(np.ceil(delta_t/threshold)) -1 
            counts[k_window] += 1
 
        for k in partner_range:
            delta_t = int(times[k] - t_i)
             
            k_window = int(np.ceil(delta_t/threshold)) -1 
            
            dt_shifted = delta_t - k_window * threshold 

            n_gap = n_on + sum(counts[:k_window])

            n_in_window = counts[k_window]

            for c in all_cols:
                buf[f"{c}_1"].append(arrays[c][i])
                buf[f"{c}_2"].append(arrays[c][k])
            buf["delta_t"].append(delta_t)
            buf["k_window"].append(k_window)
            buf["dt_shifted"].append(dt_shifted)
            buf["n_before"].append(n_before)
            buf["n_on"].append(n_on)
            buf["n_gap"].append(n_gap)
            buf["n_in_window"].append(n_in_window)
            for c in cols_to_delta:
                buf[f"delta_{c}"].append(arrays[c][k] - arrays[c][i])

            if len(buf["delta_t"]) >= flush_every:
                _flush()
                total_pairs += flush_every
                log.info("  … flushed %d pairs (total so far: %d)", flush_every, total_pairs)

    remainder = len(buf["delta_t"])
    if remainder:
        _flush()
        total_pairs += remainder

    if writer is None:
        log.warning("No false pairs found beyond %d ns.", threshold)
        empty = pd.DataFrame(columns=out_col_names)
        empty.to_parquet(out_path, index=False)
    else:
        writer.close()

    elapsed = time.perf_counter() - t0
    log.info(
        "False-pair scan complete: %d pairs in %.1f s → %s",
        total_pairs, elapsed, out_path,
    )
    return out_path

def _event_distance_to_wall(r: pd.Series, z: pd.Series, radius: float= DETECTOR_RADIUS_M, half_height: float=DETECTOR_HALF_HEIGHT_M) -> pd.Series : 
    """This helper function recieves the horizontal and vertical position of the event and computes the minimum distance to
    the walls of the instrument

    Parameters
    ----------
     r: pd.Series
        Horizontal distance where the event occurs, in m 
    z: pd.Series 
        Vertical position of the event, in m 
    radius: float 
        Radius of the instrument, in m. Default DETECTOR_RADIUS_M 
    half_height: float
        Half height of the instrument, in m. Default DETECTOR_HALF_HEIGHT_M
        

    Returns
    -------
    dist_wall: pd.Series
        Distance to the closest wall at the time of the event 
    """

    dist_wall = np.minimum(radius-r,half_height-np.abs(z)
                    )
    return pd.Series(dist_wall)  

# ─────────────────────────────────────────────────────────────────────────────
# Chunked projection helper
# ─────────────────────────────────────────────────────────────────────────────

def _rename_pair_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    """Rename raw tmp pair columns to physics notation and compute derived cols.

    Input columns (from _find_forward_pairs / _find_false_pairs):
        <raw>_1, <raw>_2  — individual measurements for anchor and partner
        delta_t, delta_<raw>  — time difference and per-quantity deltas
        label_1, pair_id_1, label_name_1, label_2, pair_id_2, label_name_2

    Output columns:
        Individual : t_1, E_1, x_1, y_1, z_1, r_1, r2_1, phi_1, D_1  (anchor)
                     t_2, E_2, x_2, y_2, z_2, r_2, r2_2, phi_2, D_2  (partner)
        Deltas     : dt, dE, dx, dy, dz, dr, dr2, dphi, dD
        Derived    : dR = sqrt(dx²+dy²+dz²)   [3D spatial separation], dwall_1, dwall_2
        Labels     : pair_id_1, pair_id_2, label_name_1, label_name_2, is_ibd, pair_type, label_1, label_2 
    """
    rename: dict[str, str] = {}

    # Individual columns: time_1 → t_1, energy_1 → E_1, distance_1 → D_1, …
    for raw, phys in _RAW_TO_PHYS.items():
        rename[f"{raw}_1"] = f"{phys}_1"
        rename[f"{raw}_2"] = f"{phys}_2"

    # Delta columns: delta_time → dt, delta_energy → dE, …
    rename["delta_t"] = "dt"
    for raw, phys in _RAW_TO_PHYS.items():
        if raw != "time":
            rename[f"delta_{raw}"] = f"d{phys}"

    chunk = chunk.rename(columns=rename)

    # Derived column: 3D spatial separation ΔR = sqrt(Δx²+Δy²+Δz²)
    chunk["dR"] = np.sqrt(chunk["dx"]**2 + chunk["dy"]**2 + chunk["dz"]**2)

    # Distance to the nearest detector wall (side or cap), in m
    chunk["dwall_1"] = _event_distance_to_wall(chunk["r_1"], chunk["z_1"])
    chunk["dwall_2"] = _event_distance_to_wall(chunk["r_2"], chunk["z_2"])
    # Filter to check whether is a true event or not
    is_ibd = (chunk["pair_id_1"] == chunk["pair_id_2"]) & (chunk["pair_id_1"] != -1 ) & (chunk["label_1"] == 1) & (chunk["label_2"] == 2)
    chunk["is_ibd"] = is_ibd.astype(int)
    
    false_ibd =  (chunk["is_ibd"] == 1 ) & (chunk["k_window"] > 0)
    chunk["false_ibd"] = false_ibd.astype(int)
    
    chunk.loc[(chunk["k_window"] > 0) , "is_ibd"] = 0
    # Filter to know which type of pair is, true (ibd), accidental (acc_ibd) or false (radio_radio) 
    pair_type = np.select([(chunk["is_ibd"] == 0) &
        ((chunk["label_1"] != 3) | (chunk["label_2"] !=3)), (chunk["label_1"] == 3) &( chunk["label_2"] == 3) ], ["acc_ibd", "radio_radio"], default="ibd")
    chunk["pair_type"] = pair_type


    is_dc = (chunk["n_before"] == 0) & (chunk["n_gap"]==0) & (chunk["n_in_window"]
                                                              == 1)
    chunk["is_dc"] = is_dc.astype(int)

    return chunk


def _project_pairs_chunked(
    tmp_path:   Path,
    feat_path:  Path,
    label_path: Path,
) -> int:
    """Rename, derive columns, filter and split a raw pairs parquet file.

    Reads the tmp parquet file one row-group at a time, applies physics-notation
    renaming via ``_rename_pair_columns``, drops rows with any NaN in
    ``REQUIRED_EVENT_COLS``, and streams the results into two separate
    ``ParquetWriter`` files (features and labels).

    Output schema
    -------------
    features : EVENTS_FEATURE_COLS

    labels   : EVENTS_LABEL_COLS

    Parameters
    ----------
    tmp_path   : Path to the raw pairs parquet written by a find_*_pairs call.
    feat_path  : Destination for the features parquet.
    label_path : Destination for the labels parquet.

    Returns
    -------
    Total number of rows written across all row groups.
    """
    pf           = pq.ParquetFile(tmp_path)
    feat_writer:  pq.ParquetWriter | None = None
    label_writer: pq.ParquetWriter | None = None
    total_rows = 0

    for rg_idx in range(pf.metadata.num_row_groups):
        raw_chunk: pd.DataFrame = pf.read_row_group(rg_idx).to_pandas()

        # Rename to physics notation and compute dR
        chunk = _rename_pair_columns(raw_chunk)

        # Drop rows missing any required column
        chunk = chunk.dropna(subset=REQUIRED_EVENT_COLS)

        if chunk.empty:
            continue

        feat_chunk  = pa.Table.from_pandas(
            chunk[EVENTS_FEATURE_COLS], preserve_index=False
        )
        label_chunk = pa.Table.from_pandas(
            chunk[EVENTS_LABEL_COLS], preserve_index=False
        )

        if feat_writer is None:
            feat_writer  = pq.ParquetWriter(feat_path,  feat_chunk.schema)
            label_writer = pq.ParquetWriter(label_path, label_chunk.schema)

        feat_writer.write_table(feat_chunk)
        label_writer.write_table(label_chunk)
        total_rows += len(chunk)

        log.info(
            "  row-group %d/%d → %d rows kept (running total: %d)",
            rg_idx + 1, pf.metadata.num_row_groups, len(chunk), total_rows,
        )

    if feat_writer is not None:
        feat_writer.close()
        label_writer.close()
    else:
        # No valid rows at all — write empty files so downstream never hits a
        # missing-file error.
        empty_feat  = pa.table({c: pa.array([], type=(pa.int64() if c in EVENTS_COUNTS_COLS else pa.float64())) for c in EVENTS_FEATURE_COLS })
        empty_label = pa.table({c: pa.array([], type=(pa.string() if "label_name" in c or c == "pair_type" else pa.int64()) )  for c in EVENTS_LABEL_COLS})
        pq.write_table(empty_feat,  feat_path)
        pq.write_table(empty_label, label_path)

    return total_rows


def _compute_parameters(events_times: np.ndarray | pd.Series) -> dict[str, float | int]:
    """This functioncomputes R and the total time  

    Parameters
    ----------
    events_times : np.ndarray or pd.Series 
        Array or pd.Series containing the time in nanoseconds (no sorting required)

    Returns
    -------
    dict[str, int | float]: 
        Dict containg the name and the values of the computed parameters: 
        R_s [ns-1]: Rate of the events
        time_total [ns]: Total time of the experiment (t_max - t_min)
        N_total [# of events]: Counts all the events without knowing the type of event
    Raises 
    ------ 

    ValueError
        If N is < 2 or max == min 
    """
    N = len(events_times)
    if N < 2:
        raise ValueError(f"Need at least 2 events got {N=}", )

    t_min = events_times.min() 
    t_max = events_times.max()
    if t_min < t_max:
        time_total = t_max - t_min
    else:
        raise ValueError(f"{t_max=} and {t_min=}, negative or zero total time")

    R_s = N/time_total

    log.info(f"Parameters computed corretly: {R_s=}, {time_total=}, {N=}")

    return {"N_total": N, "time_total": time_total, "R_s": R_s}

# ─────────────────────────────────────────────────────────────────────────────
# Main processing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _process(raw_dir: Path) -> None:
    """Full preprocessing pipeline: load → engineer → pair → save."""
    _validate_raw_inputs(raw_dir)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Base feature engineering ──────────────────────────────────────────
    df_total = _load_raw(raw_dir)


    label_cols   = LABEL_COLS_TOTAL
    feature_cols = [c for c in df_total.columns if c not in label_cols]

    # ── 3. Save total features / labels (these fit in RAM, written directly) ──
    df_total[feature_cols].to_parquet(PROCESSED_FILES["total_features"], index=False)
    log.info("Saved %-45s  (%d rows, %d cols)",
             str(PROCESSED_FILES["total_features"]), len(df_total), len(feature_cols))

    df_total[label_cols].to_parquet(PROCESSED_FILES["total_labels"], index=False)
    log.info("Saved %-45s  (%d rows, %d cols)",
             str(PROCESSED_FILES["total_labels"]), len(df_total), len(label_cols))
    # Compute parameters and save them into processed files

    params = _compute_parameters(df_total["time"])
    np.savez(PROCESSED_FILES["parameters"], **params)
    log.info("Saved %-45s  %s",
             str(PROCESSED_FILES["parameters"]), params)


    # ── 4. Forward-pair construction (streams to a tmp parquet) ──────────────
    tmp_path = _find_forward_pairs(df_total, threshold=PAIR_THRESHOLD_NS)

    if not tmp_path.exists() or pq.read_metadata(tmp_path).num_rows == 0:
        log.error("Pair table is empty — aborting.")
        tmp_path.unlink(missing_ok=True)
        sys.exit(1)

    # ── 5. Project real pairs row-group by row-group (never loads full table) ─
    log.info("Projecting real pairs from %s …", tmp_path)
    n_events = _project_pairs_chunked(
        tmp_path,
        PROCESSED_FILES["events_features"],
        PROCESSED_FILES["events_labels"],
    )
    log.info("Real pairs written: %d rows.", n_events)
    tmp_path.unlink(missing_ok=True)

    # ── 6. False-pair construction (double-window complement) ─────────────────
    tmp_false_path = _find_false_pairs(
        df_total,
        threshold=PAIR_THRESHOLD_NS,
    )

    # ── 7. Project false pairs row-group by row-group ─────────────────────────
    log.info("Projecting false pairs from %s …", tmp_false_path)
    n_false = _project_pairs_chunked(
        tmp_false_path,
        PROCESSED_FILES["false_events_features"],
        PROCESSED_FILES["false_events_labels"],
    )
    log.info("False pairs written: %d rows.", n_false)
    tmp_false_path.unlink(missing_ok=True)

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
