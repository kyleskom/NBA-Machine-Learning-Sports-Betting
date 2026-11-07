import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

import pandas as pd
import tensorflow as tf
from colorama import Fore, Style

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.DataProviders.PredictionMarketProvider import (
    PredictionMarketOddsProvider,
    compare_markets,
    today_et,
)
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.snapshots import days_rest
from src.Utils.features import XGB_ML_FEATURES, select_features
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import (
    create_todays_games_from_odds,
    get_json_data,
    to_data_frame,
    get_todays_games_json,
    create_todays_games,
)

DATA_URL_TEMPLATE = "https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo={date_to}&Division=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={season}&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision="
PREDICTION_MARKETS = {"kalshi", "polymarket"}


def season_start_year(d):
    """NBA season starts in October: Oct 2026 -> 2026 (season 2026-27)."""
    return d.year if d.month >= 9 else d.year - 1


def season_string(d):
    y = season_start_year(d)
    return f"{y}-{(y + 1) % 100:02d}"


def todays_games_url(d):
    return ("https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/"
            f"{season_start_year(d)}/scores/00_todays_scores.json")


def schedule_path(d):
    """Use Data/nba-<year>-UTC.csv; download it from fixturedownload.com if missing."""
    year = season_start_year(d)
    path = Path("Data") / f"nba-{year}-UTC.csv"
    if not path.exists():
        url = f"https://fixturedownload.com/download/nba-{year}-UTC.csv"
        print(f"Downloading {year}-{(year + 1) % 100:02d} schedule from {url}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def create_todays_games_data(games, df, odds, schedule_df, game_date):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []

    for game in games:
        home_team, away_team = game
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        if odds:
            game_key = f"{home_team}:{away_team}"
            game_odds = odds[game_key]
            todays_games_uo.append(game_odds['under_over_odds'])
            home_team_odds.append(game_odds[home_team]['money_line_odds'])
            away_team_odds.append(game_odds[away_team]['money_line_odds'])
        else:
            todays_games_uo.append(input(home_team + ' vs ' + away_team + ': '))
            home_team_odds.append(input(home_team + ' odds: '))
            away_team_odds.append(input(away_team + ' odds: '))

        # calculate days rest for both teams
        home_games = schedule_df[
            (schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)
        ]
        away_games = schedule_df[
            (schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)
        ]
        home_days_off = days_rest(home_games['Date'], game_date)
        away_days_off = days_rest(away_games['Date'], game_date)
        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        stats = pd.concat([
            home_team_series,
            away_team_series.rename(index={col: f"{col}.1" for col in df.columns}),
        ])
        stats['Days-Rest-Home'] = home_days_off
        stats['Days-Rest-Away'] = away_days_off
        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    frame_ml = select_features(games_data_frame, XGB_ML_FEATURES)
    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def load_schedule(d):
    return pd.read_csv(schedule_path(d), parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')


def resolve_games(odds, sportsbook, allow_today_fallback=True):
    if odds:
        games = create_todays_games_from_odds(odds)
        if len(games) == 0:
            print("No games found.")
            return None, None
        game_key = f"{games[0][0]}:{games[0][1]}"
        if game_key not in odds:
            print(game_key)
            print(
                Fore.RED,
                "--------------Games list not up to date for todays games!!! Scraping disabled until list is updated.--------------",
            )
            print(Style.RESET_ALL)
            return games, None
        print(f"------------------{sportsbook} odds data------------------")
        for game_key in odds.keys():
            home_team, away_team = game_key.split(":")
            print(
                f"{away_team} ({odds[game_key][away_team]['money_line_odds']}) @ "
                f"{home_team} ({odds[game_key][home_team]['money_line_odds']})"
            )
        return games, odds

    if not allow_today_fallback:
        print("No date-specific odds games found; refusing to fall back to today's schedule for -date.")
        return None, None
    games_json = get_todays_games_json(todays_games_url(datetime.today()))
    return create_todays_games(games_json), None


def run_models(data, normalized_data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args):
    probs = {}
    if args.xgb:
        print("---------------XGBoost Model Predictions---------------")
        probs["XGBoost"] = XGBoost_Runner.xgb_runner(
            data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc
        )
        print("-------------------------------------------------------")
    if args.nn:
        print("------------Neural Network Model Predictions-----------")
        probs["Neural Net"] = NN_Runner.nn_runner(
            normalized_data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc
        )
        print("-------------------------------------------------------")
    return probs


def print_devices():
    gpus = tf.config.list_physical_devices("GPU")
    from src.Utils.gpu import xgb_device
    print(f"TensorFlow GPUs: {[g.name for g in gpus] or 'none (CPU)'} | XGBoost device: {xgb_device()}")


def main(args):
    print_devices()
    game_date = date.fromisoformat(args.date) if args.date else today_et()
    if args.markets and not args.odds:
        args.odds = "kalshi"
    if game_date != today_et() and args.odds not in PREDICTION_MARKETS:
        print("A non-today -date requires -odds kalshi or -odds polymarket; "
              "sportsbook odds are current-only, not date-specific.")
        return
    odds = None
    if args.odds in PREDICTION_MARKETS:
        odds = PredictionMarketOddsProvider(venue=args.odds, date=game_date).get_odds()
        missing = [k for k, v in (odds or {}).items() if v["under_over_odds"] is None]
        if missing:
            print(f"No Kalshi total for {len(missing)} game(s); O/U prediction will be blank for those.")
    elif args.odds:
        odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
    if args.date and not odds:
        print(f"No date-specific odds returned for {game_date}; stopping without today's schedule.")
        return
    games, odds = resolve_games(odds, args.odds, allow_today_fallback=not bool(args.date))
    if games is None:
        return

    stats_json = get_json_data(DATA_URL_TEMPLATE.format(
        season=season_string(game_date),
        date_to=(game_date - timedelta(days=1)).strftime("%m%%2F%d%%2F%Y"),
    ))
    df = to_data_frame(stats_json)
    if df.empty:
        print(Fore.RED + "Could not load team stats from stats.nba.com (blocked or timed out)." + Style.RESET_ALL)
        return
    schedule_df = load_schedule(game_date)
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = create_todays_games_data(
        games, df, odds, schedule_df, game_date
    )

    if args.A:
        args.xgb = True
        args.nn = True

    normalized_data = tf.keras.utils.normalize(data, axis=1) if args.nn else None
    probs = run_models(
        data,
        normalized_data,
        todays_games_uo,
        frame_ml,
        games,
        home_team_odds,
        away_team_odds,
        args,
    )

    if args.markets or args.odds in PREDICTION_MARKETS:
        for label, arr in probs.items():
            home_probs = [float(row[1]) for row in arr]
            compare_markets(games, home_probs, date=game_date, label=label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Odds source: kalshi, polymarket (Polymarket US), or a sportsbook (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny)')
    parser.add_argument('-markets', action='store_true', help='Compare model win % against Kalshi and Polymarket US prices')
    parser.add_argument('-date', help='Game date YYYY-MM-DD in US Eastern time (prediction markets only; default today)')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    args = parser.parse_args()
    main(args)
