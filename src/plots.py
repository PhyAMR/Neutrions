"""
plot.py
─────────────────────────────────────────────────────────────────────────────
Plotting utilities for PET/detector event data visualisation.

All public functions accept a DataFrame and write a PDF (or return a
patchworklib object when save=False).  No top-level I/O or side-effects
are performed on import.

Typical usage
-------------
From another script:

    from plot import plot_joint_threshold_split, plot_all_combos
    from preprocess import run as preprocess

    preprocess()                        # ensure processed data exists
    plot_all_combos(df_total, df_events)
"""

from __future__ import annotations

import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patchworklib as pw
import seaborn as sns

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

    fmt    = logging.Formatter(
        fmt="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(script_path.stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()          # avoid duplicate handlers on re-import

    file_h    = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_h.setFormatter(fmt)

    console_h = logging.StreamHandler()
    console_h.setFormatter(fmt)

    logger.addHandler(file_h)
    logger.addHandler(console_h)
    return logger


log = _setup_logging()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_ROOT = Path("outputs/plots")

#: Default mapping from DataFrame column names to axis labels.
LABEL_DICT: dict[str, str] = {
    "r":              r"$r$ m",
    "r2":             r"$r^2$ m$^2$",
    "phi":            r"$\phi$",
    "theta":          r"$\theta$",
    "label_name":     r"Particle Type",
    "event_id":       r"Event ID",
    "energy":         r"$E$",
    "delta_t":        r"$\Delta t$ ns",
    "delta_r":        r"$\Delta r$ m",
    "delta_phi":      r"$\Delta \phi$",
    "delta_energy":   r"$\Delta E$",
    "delta_distance": r"$\Delta D$ m",
    "distance":       r"$D$ m",
    "pair_id":        r"Pair ID",
    "label":          r"Label",
}

_COLORS  = ["#3498db", "#e74c3c", "#2ecc71"]
_MARKERS = ["o", "s", "^"]

# Feature sets for the two standard combo loops
TOTAL_FEATURE_COLS  = ["r", "r2", "energy", "phi", "distance"]
EVENTS_FEATURE_COLS = [
    "delta_t", "delta_energy",
    "delta_x", "delta_y", "delta_z",
    "delta_r", "delta_r2", "delta_phi", "delta_distance",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _output_path(row_col: str, x_col: str, y_col: str) -> Path:
    """Return the canonical PDF output path for a given plot configuration."""
    return (
        OUTPUT_ROOT
        / row_col
        / "threshold_split"
        / x_col
        / f"joint_threshold_{x_col}_{y_col}.pdf"
    )


def _build_row_conditions(
    row_col: Optional[str],
    threshold_row: Optional[float],
) -> list[tuple]:
    """Return (filter_fn, title_fragment) pairs for row splitting."""
    if threshold_row is not None and row_col is not None:
        label = LABEL_DICT.get(row_col, row_col)
        return [
            (lambda d, c=row_col, t=threshold_row: d[c] <= t,
             rf"{label} $\leq$ {threshold_row}"),
            (lambda d, c=row_col, t=threshold_row: d[c] > t,
             rf"{label} $>$ {threshold_row}"),
        ]
    return [(lambda d: pd.Series(True, index=d.index), "All Data")]


def _build_col_conditions(
    col_col: Optional[str],
    threshold_col: Optional[list],
) -> list[tuple]:
    """Return (filter_fn, title_fragment) pairs for column splitting."""
    if col_col is not None and threshold_col is not None:
        label = LABEL_DICT.get(col_col, col_col)
        return [
            (
                lambda d, c=col_col, lo=threshold_col[i], hi=threshold_col[i + 1]:
                    (d[c] >= lo) & (d[c] < hi),
                rf"{threshold_col[i]} $\leq$ {label} < {threshold_col[i + 1]}",
            )
            for i in range(len(threshold_col) - 1)
        ]
    return [(lambda d: pd.Series(True, index=d.index), "")]


def _build_brick(
    dfs: list[pd.DataFrame],
    labels: list[str],
    x_col: str,
    y_col: str,
    needed_cols: list[str],
    check_row,
    check_col,
    cond_title: str,
    x_min: float,
    x_lim: float,
    y_min: float,
    y_lim: float,
    sample_size: int,
    quantile: float,
) -> pw.Brick:
    """Render one JointGrid cell and return it as a patchworklib Brick."""
    plt.clf()
    g = sns.JointGrid(height=15)

    for i, dfi in enumerate(dfs):
        # Data is already sampled and NaN-free from plot_joint_threshold_split.
        # Apply row/col conditions as boolean masks — no copy until we must.
        mask_r  = check_row(dfi)
        mask_c  = check_col(dfi)
        subset  = dfi[mask_r & mask_c]

        if subset.empty:
            log.warning("No data for '%s' in condition '%s'.", labels[i], cond_title)
            continue

        # Clip to global + local quantile limits for the scatter overlay only.
        local_cut_x = subset[x_col].quantile(quantile)
        local_cut_y = subset[y_col].quantile(quantile)
        scat_mask   = (
            (subset[x_col] <= local_cut_x) & (subset[y_col] <= local_cut_y) &
            (subset[x_col] <= x_lim)       & (subset[y_col] <= y_lim)
        )
        subset_scat = subset[scat_mask]

        sns.scatterplot(
            data=subset_scat, x=x_col, y=y_col, ax=g.ax_joint,
            color=_COLORS[i], marker=_MARKERS[i], alpha=0.4, s=15, label=labels[i],
        )
        sns.kdeplot(
            data=subset_scat, x=x_col, y=y_col, ax=g.ax_joint,
            color=_COLORS[i], levels=[0.1, 0.25, 0.5, 0.75, 0.9], alpha=0.8,
        )
        sns.histplot(
            data=subset, x=x_col, ax=g.ax_marg_x,
            color=_COLORS[i], element="step", fill=False,
            linewidth=2, binrange=(x_min, x_lim), kde=True,
        )
        sns.histplot(
            data=subset, y=y_col, ax=g.ax_marg_y,
            color=_COLORS[i], element="step", fill=False,
            linewidth=2, binrange=(y_min, y_lim), kde=True,
        )

    g.ax_joint.set_xlabel(LABEL_DICT.get(x_col, x_col))
    g.ax_joint.set_ylabel(LABEL_DICT.get(y_col, y_col))
    g.ax_joint.legend(title="Origin Cluster", loc="upper right")

    brick = pw.load_seaborngrid(g)
    brick.case.set_title(cond_title, fontsize=10)
    plt.close(g.fig)
    return brick


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def plot_joint_threshold_split(
    df: pd.DataFrame,
    labels: list[str],
    x_col: str,
    y_col: str,
    row_col: str,
    *,
    threshold_row: Optional[float] = None,
    cat: str = "label_name",
    sample_size: int = 200_000,
    col_col: Optional[str] = None,
    threshold_col: Optional[list] = None,
    quantile: float = 0.9,
    save: bool = True,
) -> Optional[pw.Brick]:
    """Create a grid of joint plots split by a row threshold and optional column bins.

    Builds JointGrid scatter + KDE plots for each unique value in `df[cat]`,
    split into row groups (≤ / > threshold_row) and optional column bins defined
    by threshold_col.  Cells are tiled with patchworklib and saved to PDF.

    Parameters
    ----------
    df            : Input DataFrame.
    labels        : Display label for each unique category value (same order as
                    df[cat].unique()).
    x_col, y_col  : Columns for the joint plot axes.
    row_col       : Column used for row-based splitting.
    threshold_row : Split value for row_col.  None → single "All Data" row.
    cat           : Column whose unique values define separate overlay series.
    sample_size   : Max points per series per cell (randomly sub-sampled).
    col_col       : Optional column for additional column-bin splitting.
    threshold_col : Bin edges for col_col (e.g. [0, 10, 20] → two bins).
    quantile      : Upper quantile used to clip axis range.
    save          : Write PDF when True; return Brick when False.

    Returns
    -------
    None if save=True, otherwise the assembled patchworklib Brick.
    """
    pw.overwrite_axisgrid()

    needed_cols = list({x_col, y_col, cat, row_col} - {None})
    if col_col:
        needed_cols.append(col_col)

    # ── Project to only the columns we need, drop NaNs, then sample once. ────
    # No .copy() — we never mutate df_clean.  Sampling here rather than inside
    # _build_brick means we hold at most sample_size rows per category in RAM
    # for the entire function, instead of one full-size copy per category.
    df_clean = df[needed_cols].dropna()
    if len(df_clean) > sample_size * df_clean[cat].nunique():
        df_clean = df_clean.groupby(cat, group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), sample_size), random_state=42)
        )

    unique_cats = df_clean[cat].unique()

    # Views, not copies — boolean indexing returns a view when no copy is forced.
    dfs = [df_clean[df_clean[cat] == c] for c in unique_cats]

    # Global axis limits computed from the already-sampled views (cheap).
    x_lim = df_clean[x_col].quantile(quantile)
    y_lim = df_clean[y_col].quantile(quantile)
    x_min = df_clean[x_col].min()
    y_min = df_clean[y_col].min()
    del df_clean   # release the sampled frame; dfs hold the per-cat views

    conditions_row = _build_row_conditions(row_col, threshold_row)
    conditions_col = _build_col_conditions(col_col, threshold_col)

    row_bricks: list[pw.Brick] = []

    for check_row, label_row in conditions_row:
        col_bricks: list[pw.Brick] = []
        for check_col, label_col in conditions_col:
            title = f"{label_row}  {label_col}".strip()
            brick = _build_brick(
                dfs, labels, x_col, y_col, needed_cols,
                check_row, check_col, title,
                x_min, x_lim, y_min, y_lim,
                sample_size, quantile,
            )
            col_bricks.append(brick)

        row_bricks.append(pw.stack(col_bricks, operator="|"))

    full_grid = pw.stack(row_bricks, operator="/")

    if save:
        out = _output_path(row_col, x_col, y_col)
        out.parent.mkdir(parents=True, exist_ok=True)
        full_grid.savefig(fname=str(out), quick=True)
        log.info("Saved → %s", out)
        return None

    return full_grid


def plot_all_combos(
    df: pd.DataFrame,
    *,
    events_sample_size: int        = 10_000,
    quantile:           float      = 0.99,
    combos:             list[str]  = EVENTS_FEATURE_COLS,
    label_col:          str        = "label_name_2",
    row_col:            str | None = "pair_id",
    col_col:            str | None = "label",
    threshold_row:      float | None = None,
    threshold_col:      float | None = None,
    save:               bool       = True,
) -> None:
    """Plot all pairwise feature combinations for a single DataFrame.

    Iterates over every unique pair of columns in ``combos``, skipping any
    combination that has already been plotted in this call.  Errors are caught
    and logged so that one failing combination never aborts the full run.

    Parameters
    ----------
    df                 : DataFrame to plot (total or events).
    events_sample_size : Max scatter points per series per cell.
    quantile           : Upper quantile used to clip axis ranges.
    combos             : Columns to form pairwise combinations from.
                         Defaults to ``EVENTS_FEATURE_COLS``.
    label_col          : Column whose unique values define overlay series
                         (e.g. ``"label_name"`` for total, ``"label_name_2"``
                         for events).
    row_col            : Column for optional row-based threshold splitting.
    col_col            : Column for optional column-based bin splitting.
    threshold_row      : Split value for ``row_col`` (None → "All Data").
    threshold_col      : Bin edges for ``col_col`` (None → no column split).
    save               : Write PDFs when True; return None when False.
    """
    plotted: set[tuple[str, str]] = set()

    for x_col, y_col in combinations(combos, 2):
        key = (x_col, y_col)
        if key in plotted:
            continue
        try:
            pw.overwrite_axisgrid()
            plot_joint_threshold_split(
                df,
                labels=list(df[label_col].unique()),
                x_col=x_col,
                y_col=y_col,
                row_col=row_col,
                threshold_row=threshold_row,
                cat=label_col,
                sample_size=events_sample_size,
                col_col=col_col,
                threshold_col=threshold_col,
                quantile=quantile,
                save=save,
            )
            plotted.add(key)
        except Exception:
            log.exception("Failed to plot %s vs %s.", x_col, y_col)