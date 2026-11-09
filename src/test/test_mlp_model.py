from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.MLP_model import MLPModel


class MLPModelTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.X_train = rng.normal(size=(32, 4)).astype(np.float32)
        self.y_train = np.tile([0, 1], 16)
        self.X_val = rng.normal(loc=100.0, size=(12, 4)).astype(np.float32)
        self.y_val = np.tile([0, 1], 6)
        self.params = {
            "hidden_units": [8],
            "epochs": 1,
            "batch_size": 8,
            "patience": 2,
            "use_bn": False,
        }

    def test_feature_normalization_uses_training_data_only(self) -> None:
        model = MLPModel(self.params).fit(
            self.X_train, self.y_train, self.X_val, self.y_val
        )
        normalizer = model._keras_model.get_layer("feature_normalization")
        np.testing.assert_allclose(
            normalizer.mean.numpy().reshape(-1),
            self.X_train.mean(axis=0),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_fit_rejects_mismatched_feature_widths(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature widths differ"):
            MLPModel(self.params).fit(
                self.X_train,
                self.y_train,
                self.X_val[:, :3],
                self.y_val,
            )

    def test_fit_rejects_non_finite_features_and_invalid_labels(self) -> None:
        bad_features = self.X_train.copy()
        bad_features[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            MLPModel(self.params).fit(
                bad_features, self.y_train, self.X_val, self.y_val
            )

        with self.assertRaisesRegex(ValueError, "only binary labels"):
            MLPModel(self.params).fit(
                self.X_train,
                np.where(self.y_train == 1, 2, 0),
                self.X_val,
                self.y_val,
            )

    def test_fit_rejects_invalid_hyperparameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "dropout_rate"):
            MLPModel({**self.params, "dropout_rate": 1.0}).fit(
                self.X_train, self.y_train, self.X_val, self.y_val
            )

    def test_learning_rate_patience_scales_below_early_stopping(self) -> None:
        self.assertEqual(MLPModel._learning_rate_patience(2), 1)
        self.assertEqual(MLPModel._learning_rate_patience(10), 3)

    def test_save_and_load_preserve_params_and_predictions(self) -> None:
        model = MLPModel(self.params).fit(
            self.X_train, self.y_train, self.X_val, self.y_val
        )
        expected = model.predict_proba(self.X_val)
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save(Path(temp_dir))
            restored = MLPModel.load(Path(temp_dir))
            actual = restored.predict_proba(self.X_val)

        self.assertEqual(restored.params, self.params)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
