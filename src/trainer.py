"""
trainer.py
─────────────────────────────────────────────────────────────────────────────
Training pipeline orchestrator for binary IBD classification.

Workflow
--------
1. Load processed events data and encode the binary target.
2. Stratified four-way split: train / val / calibration / test.
3. Run Optuna hyperparameter search — each trial calls model.fit() on
   train, scores on val using PR-AUC (or any metric in config).
4. Refit the best model on train+val combined.
5. Calibrate probabilities on the calibration split.
6. Select the optimal decision threshold on the calibration split.
7. Evaluate the full pipeline on the held-out test split.
8. Persist the model, calibrator, metrics, and Optuna study.

BaseModel contract
------------------
Every model must subclass ``BaseModel`` and implement:

    fit(X_train, y_train, X_val, y_val) → self
    predict_proba(X) → np.ndarray of shape (n,)  # positive-class probabilities

The trainer never inspects model internals — it only calls these two methods
plus ``save(path)``.

Usage
-----
Standalone:
    python trainer.py --model cnn --config config/training.yaml

Programmatic:
    from trainer import run
    from models.cnn_model import CNNModel
    run(model_cls=CNNModel, config_path="config/training.yaml")
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np
import optuna
#import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.preprocessing import LabelEncoder

from evaluation import (
    evaluate,
    get_calibrator,
    optimal_cost_threshold,
    optimal_f1_threshold,
    optimal_gmean_threshold,
)
from models.base_model import BaseModel

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

# Suppress Optuna's own verbose output — our logger handles progress.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config/training.yaml") -> dict:
    """Load and return the YAML training configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    log.info("Config loaded from %s", config_path)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and splitting
# ─────────────────────────────────────────────────────────────────────────────

def _load_data(
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray] | None]:
    """Load features and encoded binary labels from processed parquet files.

    Returns
    -------
    (X, y, feature_names, metadata)
    X : float64 array of shape (n_samples, n_features)
    y : int array of shape (n_samples,) — 1 = detected, 0 = radio
    feature_names : list of column names in X
    """
    feat_path  = Path(cfg["data"]["features_path"])
    label_path = Path(cfg["data"]["labels_path"])

    log.info("Loading features from %s …", feat_path)
    X_df = pq.read_table(feat_path).to_pandas()

    log.info("Loading labels from %s …", label_path)
    y_df = pq.read_table(label_path).to_pandas()

    if len(X_df) != len(y_df):
        raise ValueError(
            f"Features and labels row counts differ: {len(X_df)} != {len(y_df)}."
        )

    split_strategy = cfg["data"].get("split_strategy", "stratified_random")
    metadata = None
    if split_strategy == "chronological_time_blocks":
        metadata_columns = [
            "event_id_1", "event_id_2",
            "time_block_id_1", "time_block_id_2",
        ]
        missing = [column for column in metadata_columns if column not in y_df.columns]
        if missing:
            raise ValueError(
                "Grouped splitting requires endpoint metadata in the labels file: "
                + ", ".join(missing)
                + ". Regenerate processed data with the updated preprocessing script."
            )
        if "t_1" not in X_df.columns:
            raise ValueError("Grouped splitting requires the anchor timestamp t_1.")
        metadata = {column: y_df[column].to_numpy() for column in metadata_columns}
        metadata["t_1"] = X_df["t_1"].to_numpy()

    # Optional column projection
    include = cfg["features"].get("include") or []
    exclude = cfg["features"].get("exclude") or []
    if include:
        X_df = X_df[[c for c in include if c in X_df.columns]]
    if exclude:
        X_df = X_df[[c for c in X_df.columns if c not in exclude]]

    feature_names = X_df.columns.tolist()

    # Encode binary target: positive_label → 1, everything else → 0
    target_col     = cfg["data"]["target_col"]
    positive_label = cfg["data"]["positive_label"]
    y_raw          = y_df[target_col]
    y              = (y_raw == positive_label).astype(int).to_numpy()

    X = X_df.to_numpy(dtype=np.float64)

    log.info(
        "Data loaded: %d samples, %d features | "
        "positives: %d (%.1f%%) negatives: %d (%.1f%%)",
        len(X), len(feature_names),
        y.sum(), 100 * y.mean(),
        (y == 0).sum(), 100 * (1 - y.mean()),
    )
    return X, y, feature_names, metadata


def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Stratified four-way split: train / val / calibration / test.

    Stratification preserves the class ratio in every split — critical when
    the dataset is imbalanced (Thölke et al., 2023).

    Returns
    -------
    Dict with keys "train", "val", "calibration", "test", each mapping to
    a (X_split, y_split) tuple.
    """
    seed    = cfg["data"]["random_seed"]
    ratios  = cfg["data"]["splits"]

    # Step 1: carve out the test set
    test_size  = ratios["test"]
    sss_test   = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_idx, test_idx = next(sss_test.split(X, y))

    X_rest, y_rest = X[rest_idx], y[rest_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Step 2: carve out calibration from the remainder
    cal_size_of_rest = ratios["calibration"] / (1 - test_size)
    sss_cal = StratifiedShuffleSplit(
        n_splits=1, test_size=cal_size_of_rest, random_state=seed
    )
    tv_idx, cal_idx = next(sss_cal.split(X_rest, y_rest))

    X_tv,  y_tv  = X_rest[tv_idx],  y_rest[tv_idx]
    X_cal, y_cal = X_rest[cal_idx], y_rest[cal_idx]

    # Step 3: carve out val from train+val
    val_size_of_tv = ratios["val"] / (ratios["train"] + ratios["val"])
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size_of_tv, random_state=seed
    )
    train_idx, val_idx = next(sss_val.split(X_tv, y_tv))

    splits = {
        "train":       (X_tv[train_idx],  y_tv[train_idx]),
        "val":         (X_tv[val_idx],    y_tv[val_idx]),
        "calibration": (X_cal,            y_cal),
        "test":        (X_test,           y_test),
    }

    for name, (Xs, ys) in splits.items():
        log.info(
            "Split %-13s : %7d rows | pos=%.1f%%",
            name, len(Xs), 100 * ys.mean(),
        )
    return splits


def _chronological_time_block_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata: dict[str, np.ndarray],
    cfg: dict,
    time_origin: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split chronologically by whole blocks; exclude cross-split pairs and guards."""
    data_cfg = cfg["data"]
    ratios = data_cfg["splits"]
    split_names = ["train", "val", "calibration", "test"]
    fractions = np.asarray([ratios[name] for name in split_names], dtype=float)

    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Grouped split fractions must be positive and sum to 1.0.")

    block_duration_ns = int(data_cfg["block_duration_ns"])
    boundary_guard_ns = int(data_cfg.get("boundary_guard_ns", 0))
    if block_duration_ns <= 0 or boundary_guard_ns < 0:
        raise ValueError("block_duration_ns must be positive and boundary_guard_ns non-negative.")

    block_1 = metadata["time_block_id_1"].astype(np.int64, copy=False)
    block_2 = metadata["time_block_id_2"].astype(np.int64, copy=False)
    occupied_blocks = np.unique(np.concatenate([block_1, block_2]))
    if len(occupied_blocks) < 20:
        raise ValueError(
            f"Only {len(occupied_blocks)} occupied time blocks; at least 20 are "
            "required for the configured four-way chronological split."
        )

    cut_indices = np.floor(np.cumsum(fractions[:-1]) * len(occupied_blocks)).astype(int)
    block_counts = np.diff(np.concatenate(([0], cut_indices, [len(occupied_blocks)])))
    if np.any(block_counts == 0):
        raise ValueError("The number of occupied blocks is too small for these split ratios.")

    cut_blocks = occupied_blocks[cut_indices]
    split_by_block_1 = np.searchsorted(cut_blocks, block_1, side="right")
    split_by_block_2 = np.searchsorted(cut_blocks, block_2, side="right")
    same_split = split_by_block_1 == split_by_block_2

    cut_times = time_origin + cut_blocks * block_duration_ns
    anchor_times = metadata["t_1"]
    safe_context = np.ones(len(y), dtype=bool)
    for boundary in cut_times:
        safe_context &= (
            (anchor_times < boundary - boundary_guard_ns)
            | (anchor_times >= boundary + boundary_guard_ns)
        )

    keep = same_split & safe_context
    row_indices = {
        name: np.flatnonzero(keep & (split_by_block_1 == split_number))
        for split_number, name in enumerate(split_names)
    }

    seen_event_ids: dict[str, np.ndarray] = {}
    for name, indices in row_indices.items():
        if len(indices) == 0 or np.unique(y[indices]).size < 2:
            raise ValueError(
                f"Grouped split {name!r} has {len(indices)} rows or only one class. "
                "Adjust block_duration_ns or split boundaries."
            )
        event_ids = np.unique(np.concatenate((
            metadata["event_id_1"][indices],
            metadata["event_id_2"][indices],
        )))
        for other_name, other_ids in seen_event_ids.items():
            if np.intersect1d(event_ids, other_ids, assume_unique=True).size:
                raise RuntimeError(
                    f"Raw event IDs overlap between {other_name!r} and {name!r} splits."
                )
        seen_event_ids[name] = event_ids

        log.info(
            "Split %-13s : %7d rows | pos=%.1f%% | blocks=%d",
            name, len(indices), 100 * y[indices].mean(), block_counts[split_names.index(name)],
        )

    cross_split_count = int((~same_split).sum())
    guard_count = int((same_split & ~safe_context).sum())
    excluded_true_pairs = int(y[~keep].sum())
    log.info(
        "Grouped split excluded %d cross-split pairs and %d guard-region pairs "
        "(%d positive IBD pairs total).",
        cross_split_count, guard_count, excluded_true_pairs,
    )

    return {
        name: (X[indices], y[indices])
        for name, indices in row_indices.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optuna HPO
# ─────────────────────────────────────────────────────────────────────────────

def _run_hpo(
    model_cls: Type[BaseModel],
    splits:    dict[str, tuple[np.ndarray, np.ndarray]],
    cfg:       dict,
) -> tuple[dict[str, Any], optuna.Study]:
    """Run Optuna hyperparameter search.

    Each trial:
      1. Samples params via model_cls.suggest_params(trial).
      2. Instantiates and fits the model on train.
      3. Scores on val using the configured metric.

    Returns
    -------
    (best_params, study)
    """
    hpo_cfg    = cfg["optuna"]
    metric     = hpo_cfg["metric"]
    X_tr, y_tr = splits["train"]
    X_vl, y_vl = splits["val"]

    def objective(trial: optuna.Trial) -> float:
        params = model_cls.suggest_params(trial)
        model  = model_cls(params)
        model.fit(X_tr, y_tr, X_vl, y_vl)
        y_prob = model.predict_proba(X_vl)
        scores = evaluate(y_vl, y_prob, threshold=0.5)
        return scores[metric]

    sampler = optuna.samplers.TPESampler(seed=cfg["data"]["random_seed"])
    pruner  = (
        optuna.pruners.MedianPruner()
        if hpo_cfg.get("pruning", True)
        else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(
        direction=hpo_cfg["direction"],
        sampler=sampler,
        pruner=pruner,
        study_name=hpo_cfg.get("study_name"),
        storage=hpo_cfg.get("storage"),
        load_if_exists=True,
    )

    log.info(
        "Starting HPO: %d trials, metric=%s, direction=%s …",
        hpo_cfg["n_trials"], metric, hpo_cfg["direction"],
    )
    study.optimize(
        objective,
        n_trials=hpo_cfg["n_trials"],
        timeout=hpo_cfg.get("timeout"),
        n_jobs=hpo_cfg.get("n_jobs", 1),
        show_progress_bar=False,
    )

    log.info(
        "HPO complete. Best %s=%.4f | params: %s",
        metric, study.best_value, study.best_params,
    )
    return study.best_params, study


# ─────────────────────────────────────────────────────────────────────────────
# Threshold selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cfg:    dict,
) -> float:
    """Select the operating threshold on the calibration split."""
    t_cfg    = cfg["threshold"]
    strategy = t_cfg["strategy"]
    n_steps  = t_cfg.get("n_steps", 500)

    if strategy == "cost":
        threshold, best = optimal_cost_threshold(
            y_true, y_prob,
            fp_cost=t_cfg["fp_cost"],
            fn_cost=t_cfg["fn_cost"],
            n_steps=n_steps,
        )
        log.info("Threshold (cost strategy): %.4f  cost=%.1f", threshold, best)

    elif strategy == "f1":
        threshold, best = optimal_f1_threshold(y_true, y_prob, n_steps)
        log.info("Threshold (F1 strategy): %.4f  F1=%.4f", threshold, best)

    elif strategy == "gmean":
        threshold, best = optimal_gmean_threshold(y_true, y_prob, n_steps)
        log.info("Threshold (G-mean strategy): %.4f  G-mean=%.4f", threshold, best)

    else:
        raise ValueError(
            f"Unknown threshold strategy {strategy!r}. "
            "Choose from: cost | f1 | gmean"
        )
    return threshold


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_run_dir(model_name: str, cfg: dict) -> Path:
    """Create and return a timestamped output directory for this run."""
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["outputs"]["root"]) / model_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run directory: %s", run_dir)
    return run_dir


def _save_run(
    run_dir:   Path,
    model:     BaseModel,
    calibrator,
    threshold: float,
    metrics:   dict,
    study:     optuna.Study | None,
    cfg:       dict,
) -> None:
    """Persist all run artefacts to *run_dir*."""
    if cfg["outputs"].get("save_model", True):
        model.save(run_dir / "model")

    if cfg["outputs"].get("save_metrics", True):
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "threshold": threshold,
                    "test_metrics": metrics,
                    "best_params": study.best_params if study else {},
                },
                f,
                indent=2,
            )
        log.info("Metrics saved → %s/metrics.json", run_dir)

    if cfg["outputs"].get("save_study", True) and study is not None:
        study_records = [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
            if t.value is not None
        ]
        with open(run_dir / "optuna_study.json", "w") as f:
            json.dump(study_records, f, indent=2)
        log.info("Optuna study saved → %s/optuna_study.json", run_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(
    model_cls:   Type[BaseModel],
    *,
    config_path: str | Path     = "config/training.yaml",
    best_params: Optional[dict] = None,
    skip_hpo:    bool           = False,
) -> dict[str, Any]:
    """Execute the full training pipeline.

    Parameters
    ----------
    model_cls   : A BaseModel subclass (not an instance).
    config_path : Path to the YAML config file.
    best_params : If provided, skip HPO and use these params directly.
    skip_hpo    : If True, call model_cls.suggest_params with no trial
                  (uses defaults) and skip the Optuna search.

    Returns
    -------
    Dict containing:
        "model"      — fitted BaseModel instance (post-refit)
        "calibrator" — fitted calibrator
        "threshold"  — selected operating threshold
        "metrics"    — test-set evaluation dict
        "study"      — Optuna study (or None if HPO was skipped)
        "run_dir"    — Path to the output directory
    """
    t_start = time.perf_counter()
    cfg     = load_config(config_path)
    seed    = cfg["data"]["random_seed"]
    np.random.seed(seed)

    model_name = model_cls.__name__

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y, feature_names, split_metadata = _load_data(cfg)

    # ── 2. Dataset split ──────────────────────────────────────────────────────
    split_strategy = cfg["data"].get("split_strategy", "stratified_random")
    if split_strategy == "stratified_random":
        splits = _stratified_split(X, y, cfg)
    elif split_strategy == "chronological_time_blocks":
        if split_metadata is None:
            raise ValueError("Chronological grouped splitting requires endpoint metadata.")
        parameters_path = Path(cfg["data"]["parameters_path"])
        if not parameters_path.exists():
            raise FileNotFoundError(
                f"Preprocessing parameters not found: {parameters_path}. "
                "Regenerate processed data with the current preprocessing script."
            )
        with np.load(parameters_path) as parameters:
            time_origin = int(parameters["time_origin"])
        splits = _chronological_time_block_split(
            X, y, split_metadata, cfg, time_origin=time_origin
        )
    else:
        raise ValueError(
            f"Unknown split strategy {split_strategy!r}. "
            "Choose 'stratified_random' or 'chronological_time_blocks'."
        )

    X_tr, y_tr = splits["train"]
    X_vl, y_vl = splits["val"]
    X_ca, y_ca = splits["calibration"]
    X_te, y_te = splits["test"]

    # ── 3. Hyperparameter optimisation ────────────────────────────────────────
    study = None
    if best_params is not None:
        log.info("Using provided best_params — skipping HPO.")
        params = best_params
    elif skip_hpo:
        log.info("HPO skipped — using model defaults.")
        params = {}
    else:
        params, study = _run_hpo(model_cls, splits, cfg)

    # ── 4. Refit best model on train + val combined ───────────────────────────
    log.info("Refitting best model on train+val …")
    X_tv = np.concatenate([X_tr, X_vl], axis=0)
    y_tv = np.concatenate([y_tr, y_vl], axis=0)

    model = model_cls(params)
    model.fit(X_tv, y_tv, X_vl, y_vl)   # val passed for early-stopping hooks

    # ── 5. Calibrate on calibration split ─────────────────────────────────────
    cal_cfg    = cfg["calibration"]
    cal_method = (
        cal_cfg["method"]
        if len(y_ca) >= cal_cfg.get("min_isotonic_samples", 1000)
        else "platt"
    )
    log.info("Calibrating probabilities with %s …", cal_method)
    calibrator = get_calibrator(cal_method)
    raw_cal    = model.predict_proba(X_ca)
    calibrator.fit(raw_cal, y_ca)
    cal_probs  = calibrator.predict(raw_cal)

    # ── 6. Select operating threshold on calibration split ────────────────────
    threshold = _select_threshold(y_ca, cal_probs, cfg)

    # ── 7. Final evaluation on test split ─────────────────────────────────────
    raw_test  = model.predict_proba(X_te)
    cal_test  = calibrator.predict(raw_test)
    metrics   = evaluate(y_te, cal_test, threshold=threshold)

    elapsed = time.perf_counter() - t_start
    log.info(
        "Test results — PR-AUC: %.4f  ROC-AUC: %.4f  "
        "F1: %.4f  MCC: %.4f  G-mean: %.4f  (threshold=%.3f)",
        metrics["pr_auc"],  metrics["roc_auc"],
        metrics["f1"],      metrics["mcc"],
        metrics["g_mean"],  metrics["threshold"],
    )
    log.info("Total pipeline time: %.1f s", elapsed)

    # ── 8. Persist artefacts ──────────────────────────────────────────────────
    run_dir = _make_run_dir(model_name, cfg)
    _save_run(run_dir, model, calibrator, threshold, metrics, study, cfg)

    return {
        "model":      model,
        "calibrator": calibrator,
        "threshold":  threshold,
        "metrics":    metrics,
        "study":      study,
        "run_dir":    run_dir,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and evaluate a binary IBD classifier."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model class name to import (e.g. 'CNNModel').",
    )
    parser.add_argument(
        "--module",
        default=None,
        help="Python module containing the model class (e.g. 'models.cnn_model'). "
             "Defaults to 'models.<model_lower>'.",
    )
    parser.add_argument(
        "--config",
        default="training.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--skip-hpo",
        action="store_true",
        help="Skip Optuna search and use model defaults.",
    )
    args = parser.parse_args()

    import importlib
    module_name = args.module or f"models.{args.model.lower()}"
    module      = importlib.import_module(module_name)
    model_cls   = getattr(module, args.model)

    run(model_cls=model_cls, config_path=args.config, skip_hpo=args.skip_hpo)