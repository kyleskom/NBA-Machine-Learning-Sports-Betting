"""
Создание датасета для обучения модели предсказания player props.
Объединяет статистику игроков с историческими линиями букмекеров.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PLAYER_STATS_DB = BASE_DIR / "Data" / "PlayerStats.sqlite"
PLAYER_PROPS_DB = BASE_DIR / "Data" / "PlayerProps.sqlite"
OUTPUT_DB = BASE_DIR / "Data" / "player_props_dataset.sqlite"


def get_player_features(player_id, game_date, lookback_games=10):
    """
    Получить фичи для игрока на конкретную дату
    
    Фичи:
    - Средние очки за последние N игр
    - Тренд (растёт/падает)
    - Стабильность (std)
    - Минуты на площадке
    - Home/Away
    - Соперник
    - Days rest
    """
    with sqlite3.connect(PLAYER_STATS_DB) as con:
        # Получаем последние N игр до указанной даты
        query = """
            SELECT 
                game_date,
                points,
                minutes_played,
                rebounds,
                assists,
                home_away,
                opponent,
                fg_pct,
                fg3_pct,
                ft_pct
            FROM game_stats
            WHERE player_id = ?
            AND game_date < ?
            ORDER BY game_date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(
            query, 
            con, 
            params=[player_id, game_date, lookback_games]
        )
    
    if df.empty or len(df) < 3:
        return None
    
    # Рассчитываем фичи
    features = {
        # Основная статистика
        'avg_points_L5': df['points'].head(5).mean(),
        'avg_points_L10': df['points'].mean(),
        'avg_minutes': df['minutes_played'].mean(),
        
        # Стабильность
        'points_std': df['points'].std(),
        'consistency': 1 / (df['points'].std() + 1),
        
        # Тренд (последние 3 vs предыдущие 3)
        'trend': df['points'].head(3).mean() - df['points'].tail(3).mean(),
        
        # Процент попаданий
        'avg_fg_pct': df['fg_pct'].mean(),
        'avg_fg3_pct': df['fg3_pct'].mean(),
        'avg_ft_pct': df['ft_pct'].mean(),
        
        # Максимумы и минимумы
        'max_points_L10': df['points'].max(),
        'min_points_L10': df['points'].min(),
        
        # Вспомогательная статистика
        'avg_rebounds': df['rebounds'].mean(),
        'avg_assists': df['assists'].mean(),
        
        # Процент игр дома
        'home_games_pct': (df['home_away'] == 'HOME').sum() / len(df),
    }
    
    # Days rest (дни с последней игры)
    if len(df) > 0:
        last_game = pd.to_datetime(df['game_date'].iloc[0])
        current_game = pd.to_datetime(game_date)
        features['days_rest'] = (current_game - last_game).days
    else:
        features['days_rest'] = 7
    
    return features


def create_training_dataset(start_date=None, end_date=None):
    """
    Создать датасет для обучения модели.
    
    Каждая строка: 
    - Фичи игрока на момент игры
    - Линия букмекера
    - Факт (over/under)
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).date()
    if end_date is None:
        end_date = datetime.now().date()
    
    # Получаем все исторические props
    with sqlite3.connect(PLAYER_PROPS_DB) as con:
        query = """
            SELECT 
                player_name,
                player_id,
                line,
                game_date,
                over_odds,
                under_odds
            FROM player_props
            WHERE DATE(game_date) BETWEEN ? AND ?
            ORDER BY game_date
        """
        props_df = pd.read_sql_query(query, con, params=[start_date, end_date])
    
    if props_df.empty:
        print("Нет исторических props для создания датасета")
        return None
    
    # Получаем фактические результаты из game_stats
    with sqlite3.connect(PLAYER_STATS_DB) as con:
        query = """
            SELECT 
                player_id,
                game_date,
                points as actual_points
            FROM game_stats
            WHERE DATE(game_date) BETWEEN ? AND ?
        """
        results_df = pd.read_sql_query(query, con, params=[start_date, end_date])
    
    # Джойним props с результатами
    merged = props_df.merge(
        results_df,
        left_on=['player_id', 'game_date'],
        right_on=['player_id', 'game_date'],
        how='inner'
    )
    
    if merged.empty:
        print("Нет совпадений между props и результатами")
        return None
    
    # Создаём таргет: 1 = over, 0 = under
    merged['target'] = (merged['actual_points'] > merged['line']).astype(int)
    merged['line_diff'] = merged['actual_points'] - merged['line']
    
    # Добавляем фичи для каждой строки
    dataset_rows = []
    
    for idx, row in merged.iterrows():
        if idx % 100 == 0:
            print(f"Обработано {idx}/{len(merged)} записей...")
        
        features = get_player_features(
            row['player_id'],
            row['game_date'],
            lookback_games=10
        )
        
        if features is None:
            continue
        
        # Объединяем всё в одну строку
        dataset_row = {
            'player_id': row['player_id'],
            'player_name': row['player_name'],
            'game_date': row['game_date'],
            'line': row['line'],
            'actual_points': row['actual_points'],
            'target': row['target'],
            'line_diff': row['line_diff'],
            'over_odds': row['over_odds'],
            'under_odds': row['under_odds'],
            **features
        }
        
        dataset_rows.append(dataset_row)
    
    if not dataset_rows:
        print("Не удалось создать датасет")
        return None
    
    dataset = pd.DataFrame(dataset_rows)
    
    # Сохраняем в базу
    with sqlite3.connect(OUTPUT_DB) as con:
        dataset.to_sql('player_props_training', con, if_exists='replace', index=False)
    
    print(f"\n✓ Датасет создан: {len(dataset)} записей")
    print(f"  Over: {dataset['target'].sum()} ({dataset['target'].mean()*100:.1f}%)")
    print(f"  Under: {(1-dataset['target']).sum()} ({(1-dataset['target']).mean()*100:.1f}%)")
    print(f"  Средняя разница: {dataset['line_diff'].mean():.2f} очков")
    
    return dataset


def get_feature_importance_preview(dataset):
    """Предварительный анализ важности фич"""
    feature_cols = [col for col in dataset.columns 
                   if col not in ['player_id', 'player_name', 'game_date', 
                                  'target', 'actual_points', 'line_diff']]
    
    X = dataset[feature_cols]
    y = dataset['target']
    
    # Простая корреляция
    correlations = {}
    for col in feature_cols:
        if dataset[col].dtype in [np.float64, np.int64]:
            corr = dataset[col].corr(y)
            correlations[col] = abs(corr)
    
    # Сортируем по важности
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== Топ-10 важных фич (корреляция с таргетом) ===")
    for feature, corr in sorted_features[:10]:
        print(f"{feature:25s}: {corr:.4f}")


def analyze_lines_accuracy():
    """Анализ точности букмекерских линий"""
    with sqlite3.connect(OUTPUT_DB) as con:
        df = pd.read_sql_query("SELECT * FROM player_props_training", con)
    
    if df.empty:
        print("Датасет пуст")
        return
    
    # Группируем по линиям
    df['line_bucket'] = (df['line'] // 5) * 5  # Округляем до 5
    
    accuracy_by_line = df.groupby('line_bucket').agg({
        'target': ['mean', 'count']
    }).round(3)
    
    print("\n=== Точность линий (% OVER) ===")
    print(accuracy_by_line)
    
    # Анализ по разнице с линией
    print("\n=== Распределение разницы с линией ===")
    print(df['line_diff'].describe())


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Создание датасета player props")
    parser.add_argument("--days-back", type=int, default=365, help="Сколько дней назад брать данные")
    parser.add_argument("--analyze", action="store_true", help="Анализ датасета")
    
    args = parser.parse_args()
    
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    
    # Создаём датасет
    start_date = (datetime.now() - timedelta(days=args.days_back)).date()
    dataset = create_training_dataset(start_date=start_date)
    
    if dataset is not None and args.analyze:
        get_feature_importance_preview(dataset)
        analyze_lines_accuracy()


if __name__ == "__main__":
    main()
