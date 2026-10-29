import json
import tempfile
import unittest
from pathlib import Path

from src.Predict.model_selection import record_candidate, selected_model_path


class TestModelSelection(unittest.TestCase):
    def test_only_lower_loss_candidate_is_promoted_for_same_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            first = directory / "first_ML.json"
            second = directory / "second_ML.json"
            third = directory / "third_ML.json"
            for path in (first, second, third):
                path.write_text("{}")
                path.with_name(f"{path.stem}_features.json").write_text("{}")

            promoted, _ = record_candidate("ML", first, "dataset", 0.60, 0.62, directory)
            self.assertTrue(promoted)
            promoted, incumbent = record_candidate("ML", second, "dataset", 0.70, 0.63, directory)
            self.assertFalse(promoted)
            self.assertEqual(incumbent["filename"], first.name)
            promoted, _ = record_candidate("ML", third, "dataset", 0.61, 0.61, directory)
            self.assertTrue(promoted)
            self.assertEqual(selected_model_path("ML", directory), third)
            history = json.loads((directory / "model_candidates.json").read_text())
            self.assertEqual([item["promoted"] for item in history], [True, False, True])

    def test_audited_candidates_with_different_protocols_do_not_replace_each_other(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            first = directory / "first_ML.json"
            second = directory / "second_ML.json"
            first.write_text("{}")
            second.write_text("{}")
            first_report = {"dataset": {"sha256": "a"}, "final_split": {"final_start": "2025-10-01"}, "calibration": "sigmoid"}
            second_report = {"dataset": {"sha256": "b"}, "final_split": {"final_start": "2025-10-01"}, "calibration": "sigmoid"}
            promoted, _ = record_candidate("ML", first, "dataset", 0.6, 0.62, directory, first_report)
            self.assertTrue(promoted)
            promoted, _ = record_candidate("ML", second, "dataset", 0.7, 0.5, directory, second_report)
            self.assertFalse(promoted)
            self.assertEqual(selected_model_path("ML", directory), first)
