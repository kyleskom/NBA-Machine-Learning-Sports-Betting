import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, urlencode, urlunsplit
from src.Players.identity import team_id

import pandas as pd
import tensorflow as tf
import toml
from colorama import Fore, Style

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import NN_Runner, XGBoost_Runner
from src.Predict.service import market_reasons, response as prediction_response, xgboost_board
from src.Storage.normalized import connect as connect_canonical, load_games
from src.Utils.Dictionaries import team_index_current
from src.Utils.season import live_team_stats_url, runtime_season, schedule_path, scores_url
from src.Utils.tools import (
    create_todays_games_from_odds,
    get_json_data,
    to_data_frame,
    get_todays_games_json,
    create_todays_games,
)

CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"


def create_todays_games_data(games, df, odds, schedule_df, today, config=None):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []
    current = pd.Timestamp(today)
    current = current.tz_localize("America/New_York") if current.tzinfo is None else current

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

    if not match_data:
        return None, [], pd.DataFrame(), [], []

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME', 'TEAM_ID.1', 'TEAM_NAME.1', 'Date', 'Date.1', 'index', 'index.1'], errors="ignore")
    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def load_schedule(season):
    """Load the saved season schedule, falling back to canonical completed games."""
    path = schedule_path(season)
    if path.exists():
        frame = pd.read_csv(path)
        frame['Date'] = pd.to_datetime(frame['Date'], format='%d/%m/%Y %H:%M', utc=True)
        return frame
    with connect_canonical() as con:
        games = load_games(con, season)
    if games.empty:
        return pd.DataFrame(columns=['Home Team', 'Away Team', 'Date'])
    return pd.DataFrame({
        'Home Team': games['Home'],
        'Away Team': games['Away'],
        'Date': pd.to_datetime(games['Date'], utc=True),
    })


def resolve_games(odds, sportsbook, verbose=True, season=None):
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
        if verbose:
            print(f"------------------{sportsbook} odds data------------------")
            for game_key in odds.keys():
                home_team, away_team = game_key.split(":")
                print(
                    f"{away_team} ({odds[game_key][away_team]['money_line_odds']}) @ "
                    f"{home_team} ({odds[game_key][home_team]['money_line_odds']})"
                )
        return games, odds

    if season is None:
        season = runtime_season(toml.load(CONFIG_PATH))
    games_json = get_todays_games_json(scores_url(season))
    games = create_todays_games(games_json)
    if not games:
        print('No NBA games available for today.')
        return None, None
    return games, None


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


def collect_prediction_inputs(config, season, today, games, odds):
    """Build team-stat prediction inputs without player reports or player history."""
    parts = urlsplit(live_team_stats_url(config, season, today.date() - timedelta(days=1)))
    query = parse_qs(parts.query, keep_blank_values=True)
    query['DateTo'] = [(today.date() - timedelta(days=1)).strftime('%m/%d/%Y')]
    team_stats = to_data_frame(get_json_data(urlunsplit((parts.scheme, parts.netloc, parts.path,
                                                         urlencode(query, doseq=True), parts.fragment))))
    inputs = create_todays_games_data(games, team_stats, odds, load_schedule(season), today, config=config)
    return inputs


def predict_board(sportsbook='fanduel', kelly=False):
    """Build a team-stat XGBoost board with READY/PASS status for each game."""
    config = toml.load(CONFIG_PATH)
    today = datetime.now(ZoneInfo("America/New_York"))
    season = runtime_season(config, today)
    odds = SbrOddsProvider(sportsbook=sportsbook).get_odds()
    games, odds = resolve_games(odds, sportsbook, verbose=False, season=season)
    if games is None or odds is None:
        return prediction_response([], sportsbook)
    inputs = collect_prediction_inputs(config, season, today, games, odds)
    data, totals, frame, home_odds, away_odds = inputs
    if data is None:
        return prediction_response([], sportsbook)
    board = xgboost_board(data, totals, frame, games, home_odds, away_odds, config, kelly)
    return prediction_response(board, sportsbook)


def print_board(result):
    ready = []
    for game in result['games']:
        if game['status'] == 'PASS':
            print(f"PASS {game['away_team']} @ {game['home_team']}: {'; '.join(game['reasons'])}")
        else:
            ready.append(game)
    if ready:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.print_predictions(ready)


def main(args):
    if args.A:
        args.xgb = True
        args.nn = True
    if args.xgb and not args.nn:
        print_board(predict_board(args.odds or 'fanduel', args.kc))
        return
    config = toml.load(CONFIG_PATH)
    today = datetime.now(ZoneInfo("America/New_York"))
    season = runtime_season(config, today)
    odds = None
    if args.odds:
        odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
    games, odds = resolve_games(odds, args.odds, season=season)
    if games is None:
        return

    inputs = collect_prediction_inputs(config, season, today, games, odds)
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = inputs
    if data is None:
        print("No valid games found to predict.")
        return

    eligible = []
    for index, (home, away) in enumerate(games[:len(frame_ml.index)]):
        reasons = market_reasons(todays_games_uo[index], home_team_odds[index], away_team_odds[index])
        if reasons:
            print(f"PASS {away} @ {home}: {'; '.join(reasons)}")
        else:
            eligible.append(index)
    if not eligible:
        return
    data = data[eligible]
    frame_ml = frame_ml.iloc[eligible].reset_index(drop=True)
    games = [games[index] for index in eligible]
    todays_games_uo = [todays_games_uo[index] for index in eligible]
    home_team_odds = [home_team_odds[index] for index in eligible]
    away_team_odds = [away_team_odds[index] for index in eligible]

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
