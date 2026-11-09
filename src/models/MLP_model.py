"""
models/mlp_model.py
─────────────────────────────────────────────────────────────────────────────
Multi-Layer Perceptron for binary IBD / radio classification.

Input
-----
The input width is inferred from the configured feature matrix at the first
fit, allowing the trainer's selected feature list to define the model input.

Architecture
------------
Input(n_features) → [Dense → BN → Activation → Dropout] times n_layers
                  → Dense(1, sigmoid)

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

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import optuna

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

callbacks = tf.keras.callbacks
layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
regularizers = tf.keras.regularizers

from models.base_model import BaseModel


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
    input_normalization : Whether to standardize each input feature using
                          training-set statistics. Default True.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params or {})
        self._keras_model: tf.keras.Model | None = None
        self._input_features: int | None = None

    # ── Architecture ──────────────────────────────────────────────────────────

    def _build(self, input_features: int, X_train: np.ndarray) -> None:
        hidden_units = self.params.get("hidden_units", [256, 128, 64])
        activation   = self.params.get("activation",   "relu")
        dropout_rate = self.params.get("dropout_rate",  0.3)
        use_bn       = self.params.get("use_bn",         True)
        l2           = self.params.get("l2",             0.0)
        lr           = self.params.get("learning_rate",  1e-3)
        normalize    = self.params.get("input_normalization", True)
        reg          = regularizers.l2(l2) if l2 > 0 else None

        inp = layers.Input(shape=(input_features,), name="input")
        if normalize:
            normalizer = layers.Normalization(axis=-1, name="feature_normalization")
            normalizer.adapt(np.asarray(X_train, dtype=np.float32))
            x = normalizer(inp)
        else:
            x = inp

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
        X_train, y_train : Training feature matrix and binary labels.
        X_val,   y_val   : Validation features and labels for early stopping.

        Returns
        -------
        self
        """
        self._validate_input(X_train, name="X_train", check_width=False)
        self._validate_input(X_val, name="X_val", check_width=False)
        if X_train.shape[1] == 0:
            raise ValueError("X_train must contain at least one feature.")
        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError(
                f"Training and validation feature widths differ: "
                f"{X_train.shape[1]} != {X_val.shape[1]}."
            )
        y_train = self._validate_labels(y_train, len(X_train), "y_train")
        y_val = self._validate_labels(y_val, len(X_val), "y_val")
        self._validate_training_params()

        if self._keras_model is None:
            input_features = X_train.shape[1]
            self._input_features = input_features
            self._build(input_features, X_train)
        self._validate_input(X_train)
        self._validate_input(X_val)

        patience = self.params.get("patience", 10)
        cb = [
            callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=patience,
                restore_best_weights=True,
                verbose=0,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc",
                mode="max",
                factor=0.5,
                patience=self._learning_rate_patience(patience),
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
        if self._keras_model is None:
            raise RuntimeError("MLPModel must be fitted before calling predict_proba().")
        self._validate_input(X)
        return self._keras_model.predict(X, verbose=0).ravel()

    def save(self, path: Path) -> None:
        """Save the Keras model and constructor parameters to *path*."""
        if self._keras_model is None:
            raise RuntimeError("MLPModel must be fitted before it can be saved.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._keras_model.save(path / "mlp_model.keras")
        with open(path / "mlp_config.json", "w", encoding="utf-8") as config_file:
            json.dump(self.params, config_file, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MLPModel":
        """Reload a saved MLPModel from *path*."""
        path = Path(path)
        config_path = path / "mlp_config.json"
        params = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as config_file:
                params = json.load(config_file)
        if isinstance(params.get("class_weight"), dict):
            params["class_weight"] = {
                int(label): weight
                for label, weight in params["class_weight"].items()
            }

        obj = cls(params)
        obj._keras_model = tf.keras.models.load_model(path / "mlp_model.keras")
        obj._input_features = int(obj._keras_model.input_shape[-1])
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
    def _learning_rate_patience(early_stopping_patience: int) -> int:
        return max(1, early_stopping_patience // 3)

    def _validate_input(
        self,
        X: np.ndarray,
        *,
        name: str = "X",
        check_width: bool = True,
    ) -> None:
        """Validate a feature matrix's dimensions, values, and fitted width."""
        if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0:
            shape = getattr(X, "shape", None)
            raise ValueError(f"{name} must be a non-empty 2-D numpy array; got {shape}.")
        if not np.issubdtype(X.dtype, np.number) or not np.isrealobj(X):
            raise ValueError(f"{name} must contain numeric values; got {X.dtype}.")
        if not np.isfinite(X).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
        if (
            check_width
            and self._input_features is not None
            and X.shape[1] != self._input_features
        ):
            expected = self._input_features
            raise ValueError(
                f"{name} expects {expected} features from the fitted model; "
                f"got shape {X.shape}."
            )

    @staticmethod
    def _validate_labels(
        labels: np.ndarray,
        expected_rows: int,
        name: str,
    ) -> np.ndarray:
        """Validate binary labels and return a one-dimensional int array."""
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != expected_rows:
            raise ValueError(
                f"{name} must have shape ({expected_rows},); got {labels.shape}."
            )
        if labels.dtype != np.bool_ and (
            not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all()
        ):
            raise ValueError(f"{name} must contain finite numeric binary labels.")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{name} must contain only binary labels 0 and 1.")
        if np.unique(labels).size != 2:
            raise ValueError(f"{name} must contain examples from both classes 0 and 1.")
        return labels.astype(np.int32, copy=False)

    def _validate_training_params(self) -> None:
        """Fail early for invalid MLP training hyperparameters."""
        hidden_units = self.params.get("hidden_units", [256, 128, 64])
        if not isinstance(hidden_units, (list, tuple)) or not hidden_units:
            raise ValueError("hidden_units must be a non-empty list of positive integers.")
        if any(
            not isinstance(units, (int, np.integer))
            or isinstance(units, bool)
            or units <= 0
            for units in hidden_units
        ):
            raise ValueError("hidden_units must be a non-empty list of positive integers.")

        activation = self.params.get("activation", "relu")
        if not isinstance(activation, str) or activation not in {"relu", "elu", "tanh"}:
            raise ValueError("activation must be one of: relu, elu, tanh.")

        dropout_rate = self.params.get("dropout_rate", 0.3)
        if (
            not isinstance(dropout_rate, (int, float, np.number))
            or isinstance(dropout_rate, bool)
            or not np.isfinite(dropout_rate)
            or not 0 <= dropout_rate < 1
        ):
            raise ValueError("dropout_rate must be finite and in the interval [0, 1).")

        learning_rate = self.params.get("learning_rate", 1e-3)
        if (
            not isinstance(learning_rate, (int, float, np.number))
            or isinstance(learning_rate, bool)
            or not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive.")

        l2 = self.params.get("l2", 0.0)
        if (
            not isinstance(l2, (int, float, np.number))
            or isinstance(l2, bool)
            or not np.isfinite(l2)
            or l2 < 0
        ):
            raise ValueError("l2 must be finite and non-negative.")

        for name, default in (("epochs", 100), ("batch_size", 256), ("patience", 10)):
            value = self.params.get(name, default)
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")

        for name in ("use_bn", "input_normalization"):
            if name in self.params and not isinstance(self.params[name], bool):
                raise ValueError(f"{name} must be a boolean.")

        class_weight = self.params.get("class_weight")
        if class_weight is not None:
            if (
                not isinstance(class_weight, dict)
                or set(class_weight) != {0, 1}
                or any(
                    not isinstance(weight, (int, float, np.number))
                    or isinstance(weight, bool)
                    or not np.isfinite(weight)
                    or weight < 0
                    for weight in class_weight.values()
                )
            ):
                raise ValueError(
                    "class_weight must map both integer labels 0 and 1 "
                    "to finite non-negative weights."
                )
