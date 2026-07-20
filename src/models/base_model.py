"""
models/base_model.py
─────────────────────────────────────────────────────────────────────────────
Abstract base class for all classifiers in the IBD pipeline.

Every model (CNN, XGBoost, GraphNN, …) must subclass ``BaseModel`` and
implement the three abstract methods below.  The trainer and evaluation
pipeline only call these methods — they never inspect model internals.

Contract
--------
fit(X_train, y_train, X_val, y_val) → self
    Train the model.  Validation data is provided for early stopping or
    any internal validation logic the model needs.  Must return ``self``
    to allow method chaining.

predict_proba(X) → np.ndarray of shape (n,)
    Return positive-class probabilities in [0, 1].  The trainer feeds
    these directly to the calibrator and threshold selector.

save(path) → None
    Persist the fitted model to *path* in whatever format the model
    requires.  The trainer creates the directory; the model writes into it.

Optional override
-----------------
suggest_params(trial) → dict
    Define the Optuna hyperparameter search space for this model.
    The default implementation returns an empty dict (no HPO, use
    whatever defaults the constructor sets).  Override in each subclass
    to expose model-specific knobs to the trainer.

Usage
-----
    from models.base_model import BaseModel

    class MyModel(BaseModel):

        def fit(self, X_train, y_train, X_val, y_val):
            ...
            return self

        def predict_proba(self, X):
            ...
            return probabilities   # 1-D np.ndarray

        def save(self, path):
            ...

        @classmethod
        def suggest_params(cls, trial):
            return {
                "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import optuna


class BaseModel(ABC):
    """Abstract base class for all pipeline classifiers.

    Parameters
    ----------
    params : Dict of hyperparameters passed by the trainer (from Optuna or
             from ``best_params`` / ``skip_hpo`` in ``trainer.run()``).
             Each subclass documents which keys it reads.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = params or {}

    # ── Required ──────────────────────────────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> "BaseModel":
        """Train the model.

        Parameters
        ----------
        X_train : Feature matrix, shape (n_train, n_features).
        y_train : Binary labels, shape (n_train,).  1 = positive, 0 = negative.
        X_val   : Validation features, shape (n_val, n_features).
        y_val   : Validation labels, shape (n_val,).

        Returns
        -------
        self — enables ``model = MyModel(params).fit(X_tr, y_tr, X_vl, y_vl)``.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities.

        Parameters
        ----------
        X : Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        1-D array of shape (n_samples,) with values in [0, 1].
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted model to *path*.

        Parameters
        ----------
        path : Directory path.  May or may not exist — implementations
               should call ``path.mkdir(parents=True, exist_ok=True)``
               before writing.
        """

    # ── Optional override ─────────────────────────────────────────────────────

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
        """Define the Optuna hyperparameter search space for this model.

        Called once per trial by the trainer.  Override in each subclass
        to expose model-specific hyperparameters to Optuna.

        The default implementation returns an empty dict, which causes the
        trainer to instantiate the model with no params (using whatever
        defaults the constructor defines).

        Parameters
        ----------
        trial : Active Optuna trial — use ``trial.suggest_*`` methods to
                sample values.

        Returns
        -------
        Dict mapping hyperparameter name → sampled value.  This dict is
        passed verbatim to ``__init__`` as the ``params`` argument.

        Example
        -------
        ::

            @classmethod
            def suggest_params(cls, trial):
                return {
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-5, 1e-2, log=True
                    ),
                    "dropout_rate": trial.suggest_float(
                        "dropout_rate", 0.0, 0.5
                    ),
                    "batch_size": trial.suggest_categorical(
                        "batch_size", [16, 32, 64]
                    ),
                }
        """
        return {}

    # ── Convenience ───────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return hard binary predictions at *threshold*.

        This is a convenience wrapper around ``predict_proba`` — the trainer
        never calls it directly, but it is useful for quick evaluation.

        Parameters
        ----------
        X         : Feature matrix, shape (n_samples, n_features).
        threshold : Decision boundary.  Default 0.5.

        Returns
        -------
        1-D integer array of shape (n_samples,) with values in {0, 1}.
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params})"