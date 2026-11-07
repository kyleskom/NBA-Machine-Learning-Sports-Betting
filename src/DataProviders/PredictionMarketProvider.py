"""Odds from regulated US prediction markets (public, no API key).

Kalshi        : https://api.elections.kalshi.com/trade-api/v2
                KXNBAGAME (moneyline), KXNBATOTAL (over/under ladder)
Polymarket US : https://gateway.polymarket.us
                aec-nba-<away>-<home>-<YYYY-MM-DD> (moneyline)

Contract prices are converted to American odds *after taker fees*, so the
existing Expected_Value / Kelly code works unchanged. The returned dict has the
same shape as SbrOddsProvider.get_odds(), plus a 'market' block with raw prices.
"""
import math
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
PMUS_API = "https://gateway.polymarket.us"
ET = ZoneInfo("America/New_York")
KALSHI_FEE_COEF = 0.07      # quadratic taker fee: 0.07 * p * (1 - p) per contract
PMUS_FEE_COEF = 0.0695      # Polymarket US taker fee (market objects expose feeCoefficient)
HEADERS = {"User-Agent": "nba-ml-betting/1.0", "Accept": "application/json"}

_TEAMS = {
    "Atlanta Hawks": ["ATL"],
    "Boston Celtics": ["BOS"],
    "Brooklyn Nets": ["BKN", "BRK"],
    "Charlotte Hornets": ["CHA", "CHO"],
    "Chicago Bulls": ["CHI"],
    "Cleveland Cavaliers": ["CLE"],
    "Dallas Mavericks": ["DAL"],
    "Denver Nuggets": ["DEN"],
    "Detroit Pistons": ["DET"],
    "Golden State Warriors": ["GS", "GSW"],
    "Houston Rockets": ["HOU"],
    "Indiana Pacers": ["IND"],
    "LA Clippers": ["LAC"],
    "Los Angeles Lakers": ["LAL"],
    "Memphis Grizzlies": ["MEM"],
    "Miami Heat": ["MIA"],
    "Milwaukee Bucks": ["MIL"],
    "Minnesota Timberwolves": ["MIN"],
    "New Orleans Pelicans": ["NO", "NOP"],
    "New York Knicks": ["NY", "NYK"],
    "Oklahoma City Thunder": ["OKC"],
    "Orlando Magic": ["ORL"],
    "Philadelphia 76ers": ["PHI"],
    "Phoenix Suns": ["PHO", "PHX"],
    "Portland Trail Blazers": ["POR"],
    "Sacramento Kings": ["SAC"],
    "San Antonio Spurs": ["SA", "SAS"],
    "Toronto Raptors": ["TOR"],
    "Utah Jazz": ["UTA", "UTAH"],
    "Washington Wizards": ["WAS", "WSH"],
}
ABBR_TO_TEAM = {a: name for name, abbrs in _TEAMS.items() for a in abbrs}


def team_from_abbr(abbr):
    return ABBR_TO_TEAM.get((abbr or "").upper())


def _get(url, params=None, retries=3):
    for attempt in range(retries):
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code == 429:
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if 0 < v < 1 else None


def taker_fee(price, coef):
    return coef * price * (1 - price)


def price_to_american(price, coef=0.0):
    """Cost of a $1 contract (plus taker fee) -> American odds."""
    if price is None:
        return None
    cost = price + taker_fee(price, coef)
    if cost >= 1:
        return None
    dec = 1.0 / cost
    if dec >= 2:
        return int(round((dec - 1) * 100))
    return int(round(-100 / (dec - 1)))


def today_et():
    return datetime.now(ET).date()


# --------------------------------------------------------------------- Kalshi
def _kalshi_date_code(d):
    return d.strftime("%y%b%d").upper()            # 2026-10-26 -> 26OCT26


def _implied_total(markets):
    """Interpolate the total where P(over) crosses 50% on Kalshi's strike ladder."""
    pts = []
    for m in markets:
        strike = m.get("floor_strike")
        bid, ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
        if strike is None or bid is None or ask is None:
            continue
        pts.append((float(strike), (bid + ask) / 2))
    pts.sort()
    for (s1, p1), (s2, p2) in zip(pts, pts[1:]):
        if p1 >= 0.5 >= p2 and p1 != p2:
            line = s1 + (p1 - 0.5) * (s2 - s1) / (p1 - p2)
            return round(line * 2) / 2               # nearest half point
    return None


def kalshi_games(date=None):
    date = date or today_et()
    code = _kalshi_date_code(date)
    games, cursor = {}, None
    while True:
        params = {"series_ticker": "KXNBAGAME", "status": "open",
                  "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = _get(f"{KALSHI_API}/events", params)
        for ev in data.get("events", []):
            ticker = ev["event_ticker"]                       # KXNBAGAME-26OCT26MINCLE
            m = re.match(r"KXNBAGAME-(\d{2}[A-Z]{3}\d{2})([A-Z]+)$", ticker)
            if not m or m.group(1) != code:
                continue
            pair = m.group(2)
            sides = {}
            for mk in ev.get("markets", []):
                abbr = mk["ticker"].rsplit("-", 1)[-1]
                sides[abbr] = mk
            away_abbr = next((a for a in sides if pair.startswith(a)), None)
            home_abbr = next((a for a in sides if a != away_abbr), None)
            home, away = team_from_abbr(home_abbr), team_from_abbr(away_abbr)
            if not home or not away:
                continue
            games[f"{home}:{away}"] = {
                "event": ticker,
                "home_ask": _f(sides[home_abbr].get("yes_ask_dollars")),
                "away_ask": _f(sides[away_abbr].get("yes_ask_dollars")),
                "home_bid": _f(sides[home_abbr].get("yes_bid_dollars")),
                "away_bid": _f(sides[away_abbr].get("yes_bid_dollars")),
                "total_event": f"KXNBATOTAL-{m.group(1)}{pair}",
            }
        cursor = data.get("cursor")
        if not cursor:
            break
    for g in games.values():
        try:
            tot = _get(f"{KALSHI_API}/markets", {"event_ticker": g["total_event"], "limit": 200})
            g["total"] = _implied_total(tot.get("markets", []))
        except Exception:
            g["total"] = None
    return games


# -------------------------------------------------------------- Polymarket US
def polymarket_us_games(date=None):
    date = date or today_et()
    suffix = date.isoformat()
    games, offset = {}, 0
    while True:
        data = _get(f"{PMUS_API}/v1/events",
                    {"seriesId": 4, "closed": "false", "limit": 100, "offset": offset})
        events = data.get("events", [])
        for ev in events:
            if not (ev.get("slug", "").startswith("nba-") and ev["slug"].endswith(suffix)):
                continue
            for mk in ev.get("markets", []):
                if mk.get("sportsMarketType") != "basketball_team_full_game_winner":
                    continue
                home = away = None
                long_is_home = None
                for side in mk.get("marketSides", []):
                    t = side.get("team") or {}
                    name = team_from_abbr(t.get("abbreviation"))
                    if t.get("ordering") == "home":
                        home, long_is_home = name, bool(side.get("long"))
                    elif t.get("ordering") == "away":
                        away = name
                if not home or not away:
                    continue
                games[f"{home}:{away}"] = {
                    "slug": mk["slug"],
                    "long_is_home": long_is_home,
                    "fee_coef": float(mk.get("feeCoefficient") or PMUS_FEE_COEF),
                }
        if len(events) < 100:
            break
        offset += 100
    for g in games.values():
        bbo = _get(f"{PMUS_API}/v1/markets/{g['slug']}/bbo").get("marketData", {})
        long_ask = _f((bbo.get("bestAsk") or {}).get("value"))
        long_bid = _f((bbo.get("bestBid") or {}).get("value"))
        short_ask = round(1 - long_bid, 4) if long_bid else None   # buy the other side
        short_bid = round(1 - long_ask, 4) if long_ask else None
        if g["long_is_home"]:
            g.update(home_ask=long_ask, away_ask=short_ask, home_bid=long_bid, away_bid=short_bid)
        else:
            g.update(home_ask=short_ask, away_ask=long_ask, home_bid=short_bid, away_bid=long_bid)
        time.sleep(0.25)
    return games


# --------------------------------------------------------------- provider API
class PredictionMarketOddsProvider:
    """venue: 'kalshi' or 'polymarket'. Totals always come from Kalshi's ladder."""

    def __init__(self, venue="kalshi", date=None):
        self.venue = venue
        self.date = date or today_et()

    def get_odds(self):
        kalshi = {}
        try:
            kalshi = kalshi_games(self.date)
        except Exception as e:
            print(f"Kalshi fetch failed: {e}")
        if self.venue == "kalshi":
            games, coef = kalshi, KALSHI_FEE_COEF
        else:
            games = polymarket_us_games(self.date)
            coef = None
        out = {}
        for key, g in games.items():
            home, away = key.split(":")
            c = g.get("fee_coef", coef)
            out[key] = {
                "under_over_odds": (kalshi.get(key) or {}).get("total"),
                home: {"money_line_odds": price_to_american(g.get("home_ask"), c)},
                away: {"money_line_odds": price_to_american(g.get("away_ask"), c)},
                "market": {**g, "venue": self.venue, "fee_coef": c},
            }
        return out


def compare_markets(games, home_probs, date=None, label="Model"):
    """Print model win prob vs Kalshi and Polymarket US asks (after fees) per side.

    games: [[home, away], ...]; home_probs: model P(home wins) per game.
    Edge = model probability - all-in cost of a $1 contract on that side.
    """
    date = date or today_et()
    try:
        k = kalshi_games(date)
    except Exception as e:
        print(f"Kalshi fetch failed: {e}"); k = {}
    try:
        p = polymarket_us_games(date)
    except Exception as e:
        print(f"Polymarket US fetch failed: {e}"); p = {}

    def cost(price, coef):
        return None if price is None else price + taker_fee(price, coef)

    def fmt(x):
        return " n/a " if x is None else f"{x * 100:5.1f}"

    print(f"----------- {label} vs prediction markets ({date}) -----------")
    print("Prices are all-in cost per $1 contract (ask + taker fee), in cents.")
    print(f"{'Team':<24}{'Model%':>7}{'Kalshi':>8}{'PolyUS':>8}{'Best edge':>11}  Venue")
    for (home, away), ph in zip(games, home_probs):
        key = f"{home}:{away}"
        kg, pg = k.get(key, {}), p.get(key, {})
        for team, side, prob in ((home, "home", ph), (away, "away", 1 - ph)):
            kc = cost(kg.get(f"{side}_ask"), KALSHI_FEE_COEF)
            pc = cost(pg.get(f"{side}_ask"), pg.get("fee_coef", PMUS_FEE_COEF))
            options = [(c, v) for c, v in ((kc, "Kalshi"), (pc, "PolyUS")) if c is not None]
            best = min(options) if options else (None, "")
            edge = None if best[0] is None else prob - best[0]
            edge_s = "   n/a" if edge is None else f"{edge * 100:+6.1f}pp"
            flag = "  <-- +EV" if edge is not None and edge > 0 else ""
            print(f"{team:<24}{prob * 100:7.1f}{fmt(kc):>8}{fmt(pc):>8}{edge_s:>11}  {best[1]}{flag}")
        tot = kg.get("total")
        print(f"{'':<24}Kalshi implied total: {tot if tot is not None else 'n/a'}")
    print("-" * 66)
