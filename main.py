import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, urlencode, urlunsplit
from src.Players.identity import team_id
from src.Players.store import connect as connect_players
from src.Players.reports import refresh_reports

import pandas as pd
import tensorflow as tf
import toml
from colorama import Fore, Style

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils import PlayerContext
from src.Utils.tools import (
    create_todays_games_from_odds,
    get_json_data,
    to_data_frame,
    get_todays_games_json,
    create_todays_games,
)

TODAYS_GAMES_URL = "https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2025/scores/00_todays_scores.json"
DATA_URL = "https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2025-26&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision="
SCHEDULE_PATH = "Data/nba-2025-UTC.csv"
CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"


def create_todays_games_data(games, df, odds, schedule_df, today, config=None, include_player_features=True):
    match_data = []
    player_rows = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []
    player_feature_cols = []
    player_defaults = {}
    player_lookup = {"exact": {}, "history": {}}

    if config is None:
        config = toml.load(CONFIG_PATH)
    if include_player_features:
        player_feature_cols, player_defaults, player_lookup = PlayerContext.load_player_context(config)
    current = pd.Timestamp(today)
    current = current.tz_localize("America/New_York") if current.tzinfo is None else current
    today_str = current.tz_convert("America/New_York").date().isoformat()

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
        previous_home_games = home_games.loc[
            home_games['Date'].dt.tz_convert('America/New_York').dt.date < current.tz_convert('America/New_York').date()
        ].sort_values('Date', ascending=False).head(1)['Date']
        previous_away_games = away_games.loc[
            away_games['Date'].dt.tz_convert('America/New_York').dt.date < current.tz_convert('America/New_York').date()
        ].sort_values('Date', ascending=False).head(1)['Date']
        if len(previous_home_games) > 0:
            last_home_date = previous_home_games.iloc[0]
            home_days_off = timedelta(days=(current.tz_convert("America/New_York").date() - last_home_date.tz_convert("America/New_York").date()).days)
        else:
            home_days_off = timedelta(days=7)
        if len(previous_away_games) > 0:
            last_away_date = previous_away_games.iloc[0]
            away_days_off = timedelta(days=(current.tz_convert("America/New_York").date() - last_away_date.tz_convert("America/New_York").date()).days)
        else:
            away_days_off = timedelta(days=7)
        home_team_series = df.loc[df.TEAM_ID.astype(int) == team_id(home_team)].squeeze(axis=0)
        away_team_series = df.loc[df.TEAM_ID.astype(int) == team_id(away_team)].squeeze(axis=0)
        if not isinstance(home_team_series, pd.Series) or not isinstance(away_team_series, pd.Series):
            raise ValueError("Missing or duplicate live team statistics")
        stats = pd.concat([
            home_team_series,
            away_team_series.rename(index={col: f"{col}.1" for col in df.columns.values}),
        ])
        stats['Days-Rest-Home'] = home_days_off.days
        stats['Days-Rest-Away'] = away_days_off.days
        match_data.append(stats)
        if include_player_features:
            player_rows.append(
                PlayerContext.build_player_feature_row(
                    player_lookup,
                    today_str,
                    home_team,
                    away_team,
                    player_feature_cols,
                    player_defaults,
                    allow_asof=True,
                    as_of=current,
                )
            )

    if not match_data:
        return None, [], pd.DataFrame(), [], []

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T
    if include_player_features:
        player_frame = pd.DataFrame(player_rows or [dict(player_defaults) for _ in range(len(games_data_frame.index))])
        games_data_frame = pd.concat(
            [games_data_frame.reset_index(drop=True), player_frame.reset_index(drop=True)],
            axis=1,
        )

    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME', 'TEAM_ID.1', 'TEAM_NAME.1', 'Date', 'Date.1', 'index', 'index.1'], errors="ignore")
    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def load_schedule():
    frame = pd.read_csv(Path(__file__).resolve().parent / SCHEDULE_PATH)
    frame['Date'] = pd.to_datetime(frame['Date'], format='%d/%m/%Y %H:%M', utc=True)
    return frame


def resolve_games(odds, sportsbook):
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

    games_json = get_todays_games_json(TODAYS_GAMES_URL)
    return create_todays_games(games_json), None


def run_models(data, normalized_data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args):
    if args.xgb:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(
            data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc
        )
        print("-------------------------------------------------------")
    if args.nn:
        print("------------Neural Network Model Predictions-----------")
        NN_Runner.nn_runner(
            normalized_data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc
        )
        print("-------------------------------------------------------")


def main(args):
    odds = None
    if args.odds:
        odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
    games, odds = resolve_games(odds, args.odds)
    if games is None:
        return

    today = datetime.now(ZoneInfo("America/New_York"))
    config = toml.load(CONFIG_PATH)
    with connect_players() as con:
        print("Player reports:", refresh_reports(con, config, today.date().isoformat()))
    # Match training: exclude all current-day team results.
    parts = urlsplit(DATA_URL)
    query = parse_qs(parts.query, keep_blank_values=True)
    query["DateTo"] = [(today.date()-timedelta(days=1)).strftime("%m/%d/%Y")]
    stats_json = get_json_data(urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), parts.fragment)))
    df = to_data_frame(stats_json)
    schedule_df = load_schedule()
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = create_todays_games_data(
        games, df, odds, schedule_df, today, config=config
    )
    if data is not None:
        coverage = frame_ml[["H_PLR_GAME_COVERAGE", "A_PLR_GAME_COVERAGE"]].min().min()
        if coverage < 1:
            raise ValueError("Player histories missing/stale for this slate. Refresh player game logs before prediction.")
        eligibility = config.get('bet-eligibility', {})
        if eligibility.get('require_complete_availability', True):
            required = ["H_INJ_COVERAGE", "A_INJ_COVERAGE", "H_PLR_ROSTER_CONFIRMED", "A_PLR_ROSTER_CONFIRMED"]
            if eligibility.get('require_confirmed_lineups', True):
                required += ["H_LINEUP_COVERAGE", "A_LINEUP_COVERAGE"]
            unavailable = [column for column in required if frame_ml[column].min() < 1]
            if unavailable:
                print("No bet: live player availability is incomplete (" + ", ".join(unavailable) + ").")
                return
    if data is None:
        print("No valid games found to predict.")
        return

    if args.A:
        args.xgb = True
        args.nn = True

    normalized_data = tf.keras.utils.normalize(data, axis=1) if args.nn else None
    run_models(
        data,
        normalized_data,
        todays_games_uo,
        frame_ml,
        games,
        home_team_odds,
        away_team_odds,
        args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from. (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    args = parser.parse_args()
    main(args)
