import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.Utils import ModelFeatures
from src.Predict.model_selection import record_candidate, selected_model_path


class TestTeamModelFeatures(unittest.TestCase):
    def test_training_removes_all_player_feature_groups(self):
        frame = pd.DataFrame([{'PTS': 110, 'PTS.1': 105, **{prefix + 'TEST': 1
                               for prefix in ModelFeatures.PLAYER_PREFIXES}}])
        self.assertEqual(list(ModelFeatures.team_only_frame(frame)), ['PTS', 'PTS.1'])

    def test_team_manifest_preserves_order_and_rejects_player_models(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.json'
            ModelFeatures.save_feature_columns(model, ['PTS.1', 'PTS'])
            frame = pd.DataFrame([{'PTS': 110, 'PTS.1': 105}])
            columns = ModelFeatures.load_feature_columns(model)
            self.assertEqual(ModelFeatures.align_frame_to_columns(frame, columns).values.tolist(), [[105, 110]])
            ModelFeatures.feature_sidecar_path(model).write_text(json.dumps(
                {'version': 'player_games_v2', 'columns': ['H_PLR_TEST']}))
            with self.assertRaisesRegex(ValueError, 'team-only'):
                ModelFeatures.load_feature_columns(model)

    def test_restored_model_with_unknown_loss_can_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'selected_models.json').write_text(json.dumps(
                {'ML': {'filename': 'old_ML.json', 'dataset': 'team', 'test_log_loss': None}}))
            candidate = root / 'new_ML.json'
            candidate.write_text('{}')
            promoted, _ = record_candidate('ML', candidate, 'team', .6, .65, root)
            self.assertTrue(promoted)
            self.assertEqual(selected_model_path('ML', root), candidate)
