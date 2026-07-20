"""
models/mlp_model.py
─────────────────────────────────────────────────────────────────────────────
Multi-Layer Perceptron for binary IBD / radio classification.

Input
-----
9 scalar tabular features per paired event:
    delta_t, delta_energy, delta_x, delta_y, delta_z,
    delta_r, delta_r2, delta_phi, delta_distance

Architecture
------------
Input(9) → [Dense → BN → Activation → Dropout] × n_layers → Dense(1, sigmoid)

The number of layers, units per layer, activation, dropout rate, and
learning rate are all exposed to Optuna via ``suggest_params``.

Default architecture (no HPO)
------------------------------
    Dense(256, relu) → BN → Dropout(0.3)
    Dense(128, relu) → BN → Dropout(0.3)
    Dense(64,  relu) → BN → Dropout(0.3)
    Dense(1,   sigmoid)

Usage
-----
    from models.mlp_model import MLPModel
    from trainer import run

    results = run(MLPModel, config_path="config/training.yaml")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import optuna

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers

from models.base_model import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: Full feature set: 9 individual anchor cols + 9 individual partner cols
#: + 9 delta cols + 1 derived spatial separation = 28 features total.
#: Order must match the column order in events_features.parquet.
FEATURE_NAMES: list[str] = [
    # Individual anchor measurements
    "t_1", "E_1", "x_1", "y_1", "z_1", "r_1", "r2_1", "phi_1", "D_1",
    # Individual partner measurements
    "t_2", "E_2", "x_2", "y_2", "z_2", "r_2", "r2_2", "phi_2", "D_2",
    # Delta features
    "dt", "dE", "dx", "dy", "dz", "dr", "dr2", "dphi", "dD",
    # Derived spatial separation
    "dR",
]
N_FEATURES: int = len(FEATURE_NAMES)   # 28


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MLPModel(BaseModel):
    """Fully-connected MLP binary classifier.

    Parameters (via ``params`` dict)
    ---------------------------------
    hidden_units   : List of ints defining units per hidden layer.
                     Default [256, 128, 64].
    activation     : Hidden-layer activation function.  Default ``"relu"``.
    dropout_rate   : Dropout probability after each hidden layer.  Default 0.3.
    use_bn         : Whether to apply BatchNormalization after each hidden
                     layer.  Default True.
    l2             : L2 weight-decay on all Dense kernels.  Default 0.0.
    learning_rate  : Adam learning rate.  Default 1e-3.
    batch_size     : Mini-batch size.  Default 256.
    epochs         : Maximum training epochs.  Default 100.
    patience       : EarlyStopping patience on val PR-AUC.  Default 10.
    class_weight   : Dict {0: w0, 1: w1} or None.  Managed per model —
                     the trainer never injects this.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params or {})
        self._keras_model: tf.keras.Model | None = None
        self._build()

    # ── Architecture ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        hidden_units = self.params.get("hidden_units", [256, 128, 64])
        activation   = self.params.get("activation",   "relu")
        dropout_rate = self.params.get("dropout_rate",  0.3)
        use_bn       = self.params.get("use_bn",         True)
        l2           = self.params.get("l2",             0.0)
        lr           = self.params.get("learning_rate",  1e-3)
        reg          = regularizers.l2(l2) if l2 > 0 else None

        inp = layers.Input(shape=(N_FEATURES,), name="input")
        x   = inp

        for i, units in enumerate(hidden_units):
            x = layers.Dense(
                units,
                kernel_regularizer=reg,
                name=f"dense_{i}",
            )(x)
            if use_bn:
                x = layers.BatchNormalization(name=f"bn_{i}")(x)
            x = layers.Activation(activation, name=f"act_{i}")(x)
            x = layers.Dropout(dropout_rate, name=f"drop_{i}")(x)

        out = layers.Dense(1, activation="sigmoid", name="output")(x)

        self._keras_model = models.Model(inp, out, name="IBD_MLP")
        self._keras_model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.AUC(curve="PR",  name="pr_auc"),
                tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            ],
        )

    # ── BaseModel interface ───────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> "MLPModel":
        """Train the MLP.

        Parameters
        ----------
        X_train, y_train : Training features (n, 9) and binary labels.
        X_val,   y_val   : Validation features and labels for early stopping.

        Returns
        -------
        self
        """
        self._validate_input(X_train)
        self._validate_input(X_val)

        cb = [
            callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=self.params.get("patience", 10),
                restore_best_weights=True,
                verbose=0,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc",
                mode="max",
                factor=0.5,
                patience=max(3, self.params.get("patience", 10) // 3),
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        self._keras_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.params.get("epochs", 100),
            batch_size=self.params.get("batch_size", 256),
            class_weight=self.params.get("class_weight", None),
            callbacks=cb,
            verbose=0,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return 1-D array of positive-class probabilities."""
        self._validate_input(X)
        return self._keras_model.predict(X, verbose=0).ravel()

    def save(self, path: Path) -> None:
        """Save the full Keras model to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._keras_model.save(path / "mlp_model.keras")

    @classmethod
    def load(cls, path: Path) -> "MLPModel":
        """Reload a saved MLPModel from *path*."""
        path = Path(path)
        obj  = cls.__new__(cls)
        obj.params       = {}
        obj._keras_model = tf.keras.models.load_model(path / "mlp_model.keras")
        return obj

    # ── Optuna search space ───────────────────────────────────────────────────

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
        """Hyperparameter search space.

        Tuned parameters
        ----------------
        n_layers       : Number of hidden layers — 1, 2, or 3.
        units_l{i}     : Units in layer i — log-uniform in [32, 512].
        activation     : relu | elu | tanh.
        dropout_rate   : uniform in [0.0, 0.5].
        use_bn         : Whether to use BatchNormalization.
        learning_rate  : log-uniform in [1e-5, 1e-2].
        batch_size     : 64 | 128 | 256 | 512.
        l2             : log-uniform in [1e-6, 1e-2] or 0 (disabled).
        """
        n_layers = trial.suggest_int("n_layers", 1, 3)
        units    = [
            trial.suggest_int(f"units_l{i}", 32, 512, log=True)
            for i in range(n_layers)
        ]
        use_l2 = trial.suggest_categorical("use_l2", [True, False])

        return {
            "hidden_units":  units,
            "activation":    trial.suggest_categorical("activation", ["relu", "elu", "tanh"]),
            "dropout_rate":  trial.suggest_float("dropout_rate", 0.0, 0.5),
            "use_bn":        trial.suggest_categorical("use_bn", [True, False]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "l2":            trial.suggest_float("l2", 1e-6, 1e-2, log=True) if use_l2 else 0.0,
            "patience":      10,
            "epochs":        100,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_input(X: np.ndarray) -> None:
        """Raise if X does not have the expected number of features."""
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(
                f"MLPModel expects input of shape (n, {N_FEATURES}) — {N_FEATURES} features. "
                f"Got {X.shape}. Feature order: {FEATURE_NAMES}"
            )