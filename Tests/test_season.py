import unittest
from datetime import date

from src.Utils.season import live_team_stats_url, runtime_season, scores_url


class TestSeasonHelpers(unittest.TestCase):
    def setUp(self):
        self.config = {
            "data_url": "https://example.test/?from={0}/{1}/{2}&to={3}&season={4}",
            "get-data": {
                "2025-26": {"start_date": "2025-10-21", "end_date": "2026-06-21"},
                "2026-27": {"start_date": "2026-10-20", "end_date": "2027-06-30"},
            },
        }

    def test_runtime_season_uses_active_or_latest_completed_season(self):
        self.assertEqual(runtime_season(self.config, date(2026, 9, 18)), "2025-26")
        self.assertEqual(runtime_season(self.config, date(2026, 10, 20)), "2026-27")

    def test_urls_follow_selected_season(self):
        self.assertEqual(scores_url("2026-27"), scores_url("2025-26"))
        self.assertIn('/liveData/scoreboard/', scores_url('2026-27'))
        url = live_team_stats_url(self.config, "2026-27", date(2026, 11, 2))
        self.assertIn("season=2026-27", url)
        self.assertIn("to=2026", url)
