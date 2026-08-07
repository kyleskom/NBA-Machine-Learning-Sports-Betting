"""
Fetch NBA moneylines and totals from Lumify (https://lumify.ai).

Drop-in alternative to SbrOddsProvider when you have a Lumify API key. Returns the
same shape expected by main.py / create_todays_games_from_odds:

  {
    "Boston Celtics:LA Lakers": {
      "under_over_odds": 224.5,
      "Boston Celtics": {"money_line_odds": -150},
      "LA Lakers": {"money_line_odds": 130},
    },
    ...
  }

Setup:
  1. Free instant key (no signup): https://lumify.ai/docs/ai
  2. export LUMIFY_API_KEY=lmfy-...
  3. python3 main.py -xgb -odds=lumify
     or python3 main.py -xgb -odds=lumify:fanduel
"""

from __future__ import annotations

import os
from typing import Any

import requests

BASE_URL = os.environ.get("LUMIFY_BASE_URL", "https://lumify.ai").rstrip("/")

# Map SBR-style sportsbook keys to Lumify bookmaker slugs when possible.
BOOK_ALIASES = {
    "fanduel": "fanduel",
    "draftkings": "draftkings",
    "betmgm": "betmgm",
    "caesars": "caesars",
    "pointsbet": "pointsbet",
    "wynn": "wynn",
    "bet_rivers_ny": "betrivers",
    "pinnacle": "pinnacle",
    "lumify": "fanduel",
}


def _normalize(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


class LumifyOddsProvider:
    """NBA odds from Lumify, shaped like SbrOddsProvider.get_odds()."""

    def __init__(self, sportsbook: str = "fanduel", api_key: str | None = None):
        self.sportsbook = BOOK_ALIASES.get(sportsbook, sportsbook)
        self.api_key = (api_key or os.environ.get("LUMIFY_API_KEY") or "").strip()
        if not self.api_key:
            raise SystemExit(
                "Missing LUMIFY_API_KEY. Get a free instant key at "
                "https://lumify.ai/docs/ai and export LUMIFY_API_KEY=lmfy-..."
            )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "kyleskom-nba-ml-betting/lumify",
            }
        )

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = self._session.get(f"{BASE_URL}{path}", params=params or {}, timeout=60)
        if resp.status_code == 401:
            raise SystemExit("Lumify auth failed (401). Check LUMIFY_API_KEY.")
        resp.raise_for_status()
        return resp.json()

    def _list_event_ids(self) -> list[int]:
        ids: list[int] = []
        for status in ("scheduled", "inprogress"):
            after_id = None
            while True:
                params: dict[str, Any] = {"sport": "nba", "status": status, "limit": 50}
                if after_id is not None:
                    params["after_id"] = after_id
                payload = self._get("/v1/events", params)
                for ev in payload.get("events") or []:
                    if ev.get("id") is not None:
                        ids.append(int(ev["id"]))
                after_id = payload.get("next_after_id")
                if not payload.get("events") or after_id is None:
                    break
        return ids

    @staticmethod
    def _participants(event: dict) -> tuple[str, str]:
        home = away = ""
        for p in event.get("participants") or []:
            role = (p.get("role") or "").lower()
            name = (p.get("team") or {}).get("name") or p.get("name") or ""
            if role == "home":
                home = name
            elif role == "away":
                away = name
        # Match SBR provider's Clippers normalization used elsewhere in this repo.
        home = home.replace("Los Angeles Clippers", "LA Clippers")
        away = away.replace("Los Angeles Clippers", "LA Clippers")
        return home, away

    def _extract_prices(
        self, odds_payload: dict, home: str, away: str
    ) -> tuple[Any, Any, Any]:
        """Return (home_ml, away_ml, total_line) for the configured sportsbook."""
        home_n, away_n = _normalize(home), _normalize(away)
        home_ml = away_ml = total = None
        for book in odds_payload.get("bookmakers") or []:
            if (book.get("bookmaker") or "").lower() != self.sportsbook.lower():
                continue
            for market in book.get("markets") or []:
                key = (market.get("key") or market.get("label") or "").lower()
                for outcome in market.get("outcomes") or []:
                    label = outcome.get("outcome") or ""
                    price = outcome.get("price")
                    point = outcome.get("point")
                    label_n = _normalize(label)
                    if key in ("h2h", "moneyline"):
                        if home_n and (label_n == home_n or home_n in label_n or label_n in home_n):
                            home_ml = price
                        elif away_n and (label_n == away_n or away_n in label_n or label_n in away_n):
                            away_ml = price
                    elif key in ("totals", "total") and label_n.startswith("over") and point is not None:
                        total = point
            break
        return home_ml, away_ml, total

    def get_odds(self) -> dict:
        """Return today's NBA odds keyed by 'home:away', SBR-compatible."""
        dict_res: dict = {}
        for event_id in self._list_event_ids():
            try:
                event = self._get(
                    f"/v1/events/{event_id}",
                    {"include_odds": "true", "bookmaker": self.sportsbook},
                )
            except requests.HTTPError as exc:
                print(f"warn: Lumify event {event_id} failed: {exc}")
                continue

            home, away = self._participants(event)
            if not home or not away:
                continue
            home_ml, away_ml, total = self._extract_prices(event.get("odds") or {}, home, away)
            dict_res[f"{home}:{away}"] = {
                "under_over_odds": total,
                home: {"money_line_odds": home_ml},
                away: {"money_line_odds": away_ml},
            }
        return dict_res
