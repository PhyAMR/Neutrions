"""
test_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Smoke test for the full training pipeline.

Generates a tiny synthetic dataset that matches the real feature schema,
writes it to a temp directory, then runs trainer.run() with Optuna reduced
to 2 trials and all epoch/patience counts cut to the bone.

Run from the project root:
    python test_pipeline.py

A passing run prints a metrics dict and exits with code 0.
Any exception propagates and exits with a non-zero code.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Add the project root to sys.path so 'src.models...' and 'src.trainer' can be found
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.models.MLP_model import MLPModel, FEATURE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_data(
    n_samples: int = 500,
    positive_frac: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features_df, labels_df) with the real column schema."""
    rng        = np.random.default_rng(seed)
    n_pos      = int(n_samples * positive_frac)
    n_neg      = n_samples - n_pos

    # Positive class: detected pairs — give them slightly different means
    # so the model has something to learn.
    X_pos = rng.normal(loc=1.0, scale=1.0, size=(n_pos, len(FEATURE_NAMES)))
    X_neg = rng.normal(loc=0.0, scale=1.0, size=(n_neg, len(FEATURE_NAMES)))
    X     = np.vstack([X_pos, X_neg]).astype(np.float32)

    label_name_2 = ["detected"] * n_pos + ["radio"] * n_neg
    label_2      = [12]         * n_pos + [3]       * n_neg
    pair_id_2    = list(range(n_pos)) + [-1] * n_neg
    label_name_1 = label_name_2   # anchor same class for simplicity
    label_1      = label_2
    pair_id_1    = pair_id_2

    # Shuffle
    idx = rng.permutation(n_samples)
    X   = X[idx]
    label_name_2 = [label_name_2[i] for i in idx]
    label_2      = [label_2[i]      for i in idx]
    pair_id_2    = [pair_id_2[i]    for i in idx]
    label_name_1 = [label_name_1[i] for i in idx]
    label_1      = [label_1[i]      for i in idx]
    pair_id_1    = [pair_id_1[i]    for i in idx]

    features_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    labels_df   = pd.DataFrame({
        "label_name_1": label_name_1,
        "label_1":      label_1,
        "pair_id_1":    pair_id_1,
        "label_name_2": label_name_2,
        "label_2":      label_2,
        "pair_id_2":    pair_id_2,
    })
    return features_df, labels_df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def _make_test_config(data_dir: Path, output_dir: Path) -> dict:
    """Return a minimal config dict — 2 Optuna trials, 3 epochs max."""
    return {
        "data": {
            "features_path": str(data_dir / "events_features.parquet"),
            "labels_path":   str(data_dir / "events_labels.parquet"),
            "target_col":    "label_name_2",
            "positive_label":"detected",
            "splits": {
                "train":       0.60,
                "val":         0.15,
                "calibration": 0.10,
                "test":        0.15,
            },
            "random_seed": 42,
        },
        "features": {
            "include": FEATURE_NAMES,
            "exclude": [],
        },
        "optuna": {
            "metric":      "pr_auc",
            "direction":   "maximize",
            "n_trials":    2,          # ← bare minimum
            "timeout":     None,
            "n_jobs":      1,
            "pruning":     False,      # off — too few steps to prune
            "storage":     None,
            "study_name":  "smoke_test",
        },
        "calibration": {
            "method":               "platt",   # platt safer for tiny cal set
            "min_isotonic_samples": 999_999,   # force platt regardless of size
        },
        "threshold": {
            "strategy": "f1",
            "fp_cost":  1.0,
            "fn_cost":  10.0,
            "n_steps":  50,            # ← coarse sweep is fine for a test
        },
        "outputs": {
            "root":         str(output_dir),
            "save_model":   True,
            "save_metrics": True,
            "save_study":   True,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ibd_smoke_"))
    data_dir   = tmp / "data"
    output_dir = tmp / "outputs"
    cfg_path   = tmp / "training.yaml"
    data_dir.mkdir()

    print("─" * 60)
    print("IBD pipeline smoke test")
    print(f"Temp dir : {tmp}")
    print("─" * 60)

    try:
        # ── 1. Synthetic data ─────────────────────────────────────────────────
        print("\n[1/5] Generating synthetic data (500 samples, 20% positive) …")
        features_df, labels_df = _make_synthetic_data(n_samples=500)
        _write_parquet(features_df, data_dir / "events_features.parquet")
        _write_parquet(labels_df,   data_dir / "events_labels.parquet")
        print(f"      Features : {features_df.shape}  {features_df.columns.tolist()}")
        print(f"      Labels   : {labels_df.shape}")
        print(f"      Class balance: {labels_df['label_name_2'].value_counts().to_dict()}")

        # ── 2. Write minimal config ───────────────────────────────────────────
        print("\n[2/5] Writing test config …")
        cfg = _make_test_config(data_dir, output_dir)
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        # ── 3. Patch MLPModel to use minimal epochs/patience ─────────────────
        # Override suggest_params so every Optuna trial uses 1 layer + 3 epochs.
        print("\n[3/5] Patching MLPModel for minimal training …")
        import optuna as _optuna

        original_suggest = MLPModel.suggest_params

        @classmethod  # type: ignore[misc]
        def _fast_suggest(cls, trial: _optuna.Trial) -> dict:
            return {
                "hidden_units":  [32],
                "activation":    "relu",
                "dropout_rate":  0.0,
                "use_bn":        False,
                "learning_rate": 1e-3,
                "batch_size":    64,
                "l2":            0.0,
                "patience":      2,    # ← stop after 2 non-improving epochs
                "epochs":        3,    # ← hard cap at 3 epochs
            }

        MLPModel.suggest_params = _fast_suggest

        # ── 4. Run the pipeline ───────────────────────────────────────────────
        print("\n[4/5] Running trainer.run() …")
        from src.trainer import run
        results = run(MLPModel, config_path=cfg_path)

        # Restore original suggest_params
        MLPModel.suggest_params = original_suggest

        # ── 5. Report ─────────────────────────────────────────────────────────
        print("\n[5/5] Results")
        print("─" * 60)
        m = results["metrics"]
        print(f"  PR-AUC      : {m['pr_auc']:.4f}")
        print(f"  ROC-AUC     : {m['roc_auc']:.4f}")
        print(f"  F1          : {m['f1']:.4f}")
        print(f"  MCC         : {m['mcc']:.4f}")
        print(f"  G-mean      : {m['g_mean']:.4f}")
        print(f"  Threshold   : {m['threshold']:.4f}")
        print(f"  TP/FP/TN/FN : {m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}")
        print(f"\n  Run artefacts → {results['run_dir']}")
        print("─" * 60)
        print("\n✓  Smoke test passed.\n")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()