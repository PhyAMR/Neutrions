"""
evaluation/calibration.py
─────────────────────────────────────────────────────────────────────────────
Probability calibration wrappers.

Both calibrators expose a minimal fit / predict interface so the trainer
can treat them interchangeably.  Calibration should always be fitted on a
held-out calibration split (never on train or test data) and evaluated on
the test split at the real-world class prevalence.

References
----------
Tian et al. (2020) — posterior re-calibration for imbalanced datasets.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ─────────────────────────────────────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────────────────────────────────────

class _BaseCalibrator:
    """Minimal interface shared by all calibrators."""

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "_BaseCalibrator":
        raise NotImplementedError

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_predict(
        self, probabilities: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        return self.fit(probabilities, labels).predict(probabilities)


# ─────────────────────────────────────────────────────────────────────────────
# Isotonic regression calibrator (non-parametric, recommended for large n)
# ─────────────────────────────────────────────────────────────────────────────

class IsotonicCalibrator(_BaseCalibrator):
    """Isotonic regression calibration.

    Non-parametric and monotone — works well when the model is systematically
    over- or under-confident.  Requires enough calibration samples to avoid
    overfitting (≥ 1 000 recommended).

    Parameters
    ----------
    out_of_bounds : How to handle predictions outside [0, 1].
                    ``"clip"`` is safest for deployment.
    """

    def __init__(self, out_of_bounds: str = "clip") -> None:
        self._model = IsotonicRegression(out_of_bounds=out_of_bounds)

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> "IsotonicCalibrator":
        self._model.fit(probabilities.ravel(), labels.ravel())
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return self._model.transform(probabilities.ravel())


# ─────────────────────────────────────────────────────────────────────────────
# Platt scaling calibrator (parametric sigmoid, recommended for small n)
# ─────────────────────────────────────────────────────────────────────────────

class PlattCalibrator(_BaseCalibrator):
    """Platt scaling (logistic regression on raw model scores).

    A single sigmoid is fitted on top of the uncalibrated probabilities.
    Suitable when the calibration set is small (< 1 000 samples) because it
    has only two free parameters and is less prone to overfitting than
    isotonic regression.

    Parameters
    ----------
    C : Inverse regularisation strength passed to LogisticRegression.
    """

    def __init__(self, C: float = 1.0) -> None:
        self._model = LogisticRegression(C=C, solver="lbfgs", max_iter=1_000)

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> "PlattCalibrator":
        self._model.fit(probabilities.ravel().reshape(-1, 1), labels.ravel())
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(
            probabilities.ravel().reshape(-1, 1)
        )[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_calibrator(method: str = "isotonic", **kwargs) -> _BaseCalibrator:
    """Return a calibrator by name.

    Parameters
    ----------
    method : ``"isotonic"`` or ``"platt"``.
    kwargs : Forwarded to the calibrator constructor.

    Returns
    -------
    An unfitted calibrator instance.
    """
    _registry = {
        "isotonic": IsotonicCalibrator,
        "platt":    PlattCalibrator,
    }
    if method not in _registry:
        raise ValueError(
            f"Unknown calibration method {method!r}. "
            f"Choose from: {list(_registry)}"
        )
    return _registry[method](**kwargs)