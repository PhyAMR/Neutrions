from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trainer import _chronological_time_block_split, _load_data


class GroupedSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.block_duration_ns = 1_000
        self.boundary_guard_ns = 100
        self.n_blocks = 20
        self.features = []
        self.labels = []

        for block_id in range(self.n_blocks):
            block_time = block_id * self.block_duration_ns + 500
            base_event_id = block_id * 6
            event_pairs = [
                (base_event_id, base_event_id + 1),
                (base_event_id, base_event_id + 2),
                (base_event_id + 3, base_event_id + 4),
                (base_event_id + 3, base_event_id + 5),
            ]
            for pair_index, (event_id_1, event_id_2) in enumerate(event_pairs):
                self.features.append({
                    "E_1": float(pair_index),
                    "t_1": block_time,
                })
                self.labels.append({
                    "is_ibd": pair_index % 2,
                    "event_id_1": event_id_1,
                    "event_id_2": event_id_2,
                    "time_block_id_1": block_id,
                    "time_block_id_2": block_id,
                })

        # Place one valid pair exactly on a split boundary; the guard should remove it.
        self.features.append({"E_1": 99.0, "t_1": 12 * self.block_duration_ns})
        self.labels.append({
            "is_ibd": 1,
            "event_id_1": self.n_blocks * 6,
            "event_id_2": self.n_blocks * 6 + 1,
            "time_block_id_1": 12,
            "time_block_id_2": 12,
        })

        self.features_df = pd.DataFrame(self.features)
        self.labels_df = pd.DataFrame(self.labels)
        self.config = {
            "data": {
                "target_col": "is_ibd",
                "positive_label": 1,
                "split_strategy": "chronological_time_blocks",
                "block_duration_ns": self.block_duration_ns,
                "boundary_guard_ns": self.boundary_guard_ns,
                "splits": {
                    "train": 0.60,
                    "val": 0.15,
                    "calibration": 0.10,
                    "test": 0.15,
                },
            },
            "features": {"include": ["E_1"], "exclude": []},
        }

    def test_load_data_keeps_split_metadata_out_of_model_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            features_path = temp_path / "features.parquet"
            labels_path = temp_path / "labels.parquet"
            pq.write_table(pa.Table.from_pandas(self.features_df, preserve_index=False), features_path)
            pq.write_table(pa.Table.from_pandas(self.labels_df, preserve_index=False), labels_path)

            config = {
                **self.config,
                "data": {
                    **self.config["data"],
                    "features_path": str(features_path),
                    "labels_path": str(labels_path),
                },
            }
            X, y, feature_names, metadata = _load_data(config)

        self.assertEqual(feature_names, ["E_1"])
        self.assertEqual(X.shape, (len(self.labels_df), 1))
        self.assertEqual(len(y), len(self.labels_df))
        self.assertIsNotNone(metadata)
        self.assertIn("event_id_1", metadata)
        self.assertIn("time_block_id_2", metadata)

    def test_time_block_split_keeps_event_ids_isolated_and_applies_guard(self) -> None:
        X = np.column_stack((np.arange(len(self.labels_df)), np.ones(len(self.labels_df))))
        y = self.labels_df["is_ibd"].to_numpy()
        metadata = {
            **{column: self.labels_df[column].to_numpy() for column in (
                "event_id_1", "event_id_2", "time_block_id_1", "time_block_id_2"
            )},
            "t_1": self.features_df["t_1"].to_numpy(),
        }

        splits = _chronological_time_block_split(X, y, metadata, self.config, time_origin=0)
        split_event_ids: dict[str, set[int]] = {}

        for name, (X_split, y_split) in splits.items():
            self.assertGreater(len(X_split), 0)
            self.assertEqual(set(np.unique(y_split)), {0, 1})
            row_indices = X_split[:, 0].astype(int)
            split_event_ids[name] = set(np.concatenate((
                metadata["event_id_1"][row_indices],
                metadata["event_id_2"][row_indices],
            )).tolist())

        for first_name, first_ids in split_event_ids.items():
            for second_name, second_ids in split_event_ids.items():
                if first_name < second_name:
                    self.assertTrue(first_ids.isdisjoint(second_ids))

        self.assertTrue(all(
            80 not in X_split[:, 0].astype(int)
            for X_split, _ in splits.values()
        ))


if __name__ == "__main__":
    unittest.main()
