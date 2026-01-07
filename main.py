import argparse
from datetime import datetime, timedelta

import pandas as pd
import tensorflow as tf
from colorama import Fore, Style

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
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


def create_todays_games_data(games, df, odds, schedule_df, today):
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
        previous_home_games = home_games.loc[
            home_games['Date'] <= today
        ].sort_values('Date', ascending=False).head(1)['Date']
        previous_away_games = away_games.loc[
            away_games['Date'] <= today
        ].sort_values('Date', ascending=False).head(1)['Date']
        if len(previous_home_games) > 0:
            last_home_date = previous_home_games.iloc[0]
            home_days_off = timedelta(days=1) + today - last_home_date
        else:
            home_days_off = timedelta(days=7)
        if len(previous_away_games) > 0:
            last_away_date = previous_away_games.iloc[0]
            away_days_off = timedelta(days=1) + today - last_away_date
        else:
            away_days_off = timedelta(days=7)
        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_days_off.days
        stats['Days-Rest-Away'] = away_days_off.days
        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def load_schedule():
    return pd.read_csv(SCHEDULE_PATH, parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')


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

    stats_json = get_json_data(DATA_URL)
    df = to_data_frame(stats_json)
    schedule_df = load_schedule()
    today = datetime.today()
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = create_todays_games_data(
        games, df, odds, schedule_df, today
    )

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
