"""
Главный скрипт для получения предсказаний player props.
Использует обученную модель для анализа сегодняшних линий.
"""

import argparse
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init

BASE_DIR = Path(__file__).resolve().parents[1]
PLAYER_STATS_DB = BASE_DIR / "Data" / "PlayerStats.sqlite"
PLAYER_PROPS_DB = BASE_DIR / "Data" / "PlayerProps.sqlite"
MODEL_DIR = BASE_DIR / "Models" / "PlayerProps_Models"

init()

# Колонки для фич (в том же порядке, что и при обучении)
FEATURE_COLUMNS = [
    "line",
    "over_odds",
    "under_odds",
    "avg_points_L5",
    "avg_points_L10",
    "avg_minutes",
    "points_std",
    "consistency",
    "trend",
    "avg_fg_pct",
    "avg_fg3_pct",
    "avg_ft_pct",
    "max_points_L10",
    "min_points_L10",
    "avg_rebounds",
    "avg_assists",
    "home_games_pct",
    "days_rest",
]


def find_best_model():
    """Найти лучшую обученную модель"""
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Директория моделей не найдена: {MODEL_DIR}")
    
    models = list(MODEL_DIR.glob("PlayerProps_*.json"))
    
    if not models:
        raise FileNotFoundError(f"Модели не найдены в {MODEL_DIR}")
    
    # Находим модель с наибольшей точностью
    pattern = re.compile(r"PlayerProps_(\d+\.\d+)%")
    
    best_model = None
    best_accuracy = 0
    
    for model_path in models:
        match = pattern.search(model_path.name)
        if match:
            accuracy = float(match.group(1))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_path
    
    if best_model is None:
        best_model = models[0]
    
    print(f"Используется модель: {best_model.name}")
    
    return best_model


def load_model_and_calibrator():
    """Загрузить модель и калибратор"""
    model_path = find_best_model()
    
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    
    # Пытаемся загрузить калибратор
    calib_path = MODEL_DIR / f"{model_path.stem}_calibration.pkl"
    calibrator = None
    
    if calib_path.exists():
        try:
            calibrator = joblib.load(calib_path)
            print("✓ Калибратор загружен")
        except Exception as e:
            print(f"⚠ Не удалось загрузить калибратор: {e}")
    
    return booster, calibrator


def get_player_features(player_name, game_date=None, lookback_games=10):
    """Получить фичи для игрока"""
    if game_date is None:
        game_date = datetime.now().date()
    
    # Находим player_id по имени
    with sqlite3.connect(PLAYER_STATS_DB) as con:
        cursor = con.execute(
            "SELECT player_id FROM players WHERE player_name LIKE ? LIMIT 1",
            [f"%{player_name}%"],
        )
        result = cursor.fetchone()
        
        if result is None:
            return None
        
        player_id = result[0]
    
    # Получаем последние игры
    with sqlite3.connect(PLAYER_STATS_DB) as con:
        query = """
            SELECT 
                game_date,
                points,
                minutes_played,
                rebounds,
                assists,
                home_away,
                fg_pct,
                fg3_pct,
                ft_pct
            FROM game_stats
            WHERE player_id = ?
            AND game_date < ?
            ORDER BY game_date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, con, params=[player_id, game_date, lookback_games])
    
    if df.empty or len(df) < 3:
        return None
    
    # Рассчитываем фичи
    features = {
        "avg_points_L5": df["points"].head(5).mean(),
        "avg_points_L10": df["points"].mean(),
        "avg_minutes": df["minutes_played"].mean(),
        "points_std": df["points"].std(),
        "consistency": 1 / (df["points"].std() + 1),
        "trend": df["points"].head(3).mean() - df["points"].tail(3).mean(),
        "avg_fg_pct": df["fg_pct"].mean(),
        "avg_fg3_pct": df["fg3_pct"].mean(),
        "avg_ft_pct": df["ft_pct"].mean(),
        "max_points_L10": df["points"].max(),
        "min_points_L10": df["points"].min(),
        "avg_rebounds": df["rebounds"].mean(),
        "avg_assists": df["assists"].mean(),
        "home_games_pct": (df["home_away"] == "HOME").sum() / len(df),
    }
    
    # Days rest
    if len(df) > 0:
        last_game = pd.to_datetime(df["game_date"].iloc[0])
        current_game = pd.to_datetime(game_date)
        features["days_rest"] = (current_game - last_game).days
    else:
        features["days_rest"] = 7
    
    return features


def get_todays_props():
    """Получить сегодняшние props"""
    today = datetime.now().date()
    
    with sqlite3.connect(PLAYER_PROPS_DB) as con:
        query = """
            SELECT 
                player_name,
                AVG(line) as line,
                AVG(over_odds) as over_odds,
                AVG(under_odds) as under_odds,
                COUNT(*) as sources_count
            FROM player_props
            WHERE DATE(game_date) = ?
            GROUP BY player_name
            HAVING sources_count >= 1
            ORDER BY player_name
        """
        df = pd.read_sql_query(query, con, params=[today])
    
    return df


def calculate_ev(prob, odds):
    """Рассчитать Expected Value"""
    if odds is None or pd.isna(odds):
        return 0
    
    odds = int(odds)
    
    if odds > 0:
        payout = odds
    else:
        payout = (100 / abs(odds)) * 100
    
    return round((prob * payout) - ((1 - prob) * 100), 2)


def predict_props(model, calibrator, props_df):
    """Сделать предсказания для списка props"""
    results = []
    
    for idx, row in props_df.iterrows():
        player_name = row["player_name"]
        line = row["line"]
        over_odds = row["over_odds"]
        under_odds = row["under_odds"]
        
        # Получаем фичи игрока
        features = get_player_features(player_name)
        
        if features is None:
            print(f"⚠ Нет данных для {player_name}")
            continue
        
        # Собираем все фичи в правильном порядке
        feature_vector = [
            line,
            over_odds if over_odds else -110,
            under_odds if under_odds else -110,
        ]
        
        for col in FEATURE_COLUMNS[3:]:  # Пропускаем line, over_odds, under_odds
            feature_vector.append(features.get(col, 0))
        
        # Предсказание
        X = np.array([feature_vector])
        
        if calibrator:
            probs = calibrator.predict_proba(X)[0]
            prob_over = probs[1]
        else:
            prob_over = model.predict(xgb.DMatrix(X))[0]
        
        prob_under = 1 - prob_over
        
        # Expected Value
        ev_over = calculate_ev(prob_over, over_odds)
        ev_under = calculate_ev(prob_under, under_odds)
        
        # Рекомендация
        if ev_over > 5:
            recommendation = "OVER"
            confidence = prob_over
            edge = ev_over
        elif ev_under > 5:
            recommendation = "UNDER"
            confidence = prob_under
            edge = ev_under
        else:
            recommendation = "PASS"
            confidence = max(prob_over, prob_under)
            edge = max(ev_over, ev_under)
        
        results.append(
            {
                "player_name": player_name,
                "line": line,
                "prob_over": prob_over * 100,
                "prob_under": prob_under * 100,
                "ev_over": ev_over,
                "ev_under": ev_under,
                "recommendation": recommendation,
                "confidence": confidence * 100,
                "edge": edge,
                "avg_L10": features["avg_points_L10"],
                "trend": features["trend"],
            }
        )
    
    return pd.DataFrame(results)


def print_predictions(results_df):
    """Красиво вывести предсказания"""
    if results_df.empty:
        print("Нет предсказаний")
        return
    
    # Сортируем по edge
    results_df = results_df.sort_values("edge", ascending=False)
    
    print("\n" + "=" * 100)
    print(f"{'PLAYER':<25} {'LINE':>6} {'PRED':>6} {'AVG':>6} {'TREND':>6} {'REC':>6} {'CONF':>6} {'EDGE':>6}")
    print("=" * 100)
    
    for _, row in results_df.iterrows():
        player = row["player_name"][:24]
        line = row["line"]
        
        # Цвета для рекомендации
        if row["recommendation"] == "OVER":
            rec_color = Fore.GREEN
            rec = "OVER"
        elif row["recommendation"] == "UNDER":
            rec_color = Fore.BLUE
            rec = "UNDER"
        else:
            rec_color = Fore.YELLOW
            rec = "PASS"
        
        # Цвет для edge
        edge_color = Fore.GREEN if row["edge"] > 10 else Fore.WHITE
        
        # Предсказанное значение
        predicted = (
            row["avg_L10"] + row["trend"]
        )  # Упрощённое предсказание для отображения
        
        print(
            f"{player:<25} "
            f"{line:>6.1f} "
            f"{predicted:>6.1f} "
            f"{row['avg_L10']:>6.1f} "
            f"{row['trend']:>+6.1f} "
            f"{rec_color}{rec:>6}{Style.RESET_ALL} "
            f"{row['confidence']:>6.1f}% "
            f"{edge_color}{row['edge']:>+6.1f}{Style.RESET_ALL}"
        )
    
    print("=" * 100)
    
    # Статистика
    total = len(results_df)
    overs = (results_df["recommendation"] == "OVER").sum()
    unders = (results_df["recommendation"] == "UNDER").sum()
    passes = (results_df["recommendation"] == "PASS").sum()
    
    print(f"\nВсего: {total} | Over: {overs} | Under: {unders} | Pass: {passes}")
    
    # Топ рекомендации
    top_picks = results_df[results_df["edge"] > 5].head(5)
    
    if not top_picks.empty:
        print(f"\n{Fore.CYAN}=== ТОП-5 СТАВОК ==={Style.RESET_ALL}")
        for i, row in enumerate(top_picks.itertuples(), 1):
            print(
                f"{i}. {row.player_name}: {row.recommendation} {row.line:.1f} "
                f"(edge: {row.edge:+.1f}, confidence: {row.confidence:.1f}%)"
            )


def main():
    parser = argparse.ArgumentParser(description="Предсказания player props")
    parser.add_argument("--player", type=str, help="Предсказание для одного игрока")
    parser.add_argument(
        "--min-edge", type=float, default=0, help="Минимальный edge для отображения"
    )
    
    args = parser.parse_args()
    
    # Загружаем модель
    print("Загрузка модели...")
    model, calibrator = load_model_and_calibrator()
    
    # Получаем props
    if args.player:
        print(f"Получение props для {args.player}...")
        props_df = pd.DataFrame(
            [{"player_name": args.player, "line": None, "over_odds": None, "under_odds": None}]
        )
    else:
        print("Получение сегодняшних props...")
        props_df = get_todays_props()
        
        if props_df.empty:
            print("⚠ Нет props на сегодня. Запустите Get_Player_Props.py --fetch")
            return
    
    # Делаем предсказания
    print(f"Анализ {len(props_df)} игроков...")
    results = predict_props(model, calibrator, props_df)
    
    # Фильтруем по edge
    if args.min_edge > 0:
        results = results[results["edge"] >= args.min_edge]
    
    # Выводим результаты
    print_predictions(results)


if __name__ == "__main__":
    main()
