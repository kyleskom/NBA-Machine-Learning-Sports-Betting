"""
Модуль для получения статистики игроков NBA и сохранения в базу данных.
Использует бесплатные API для получения актуальных данных.
"""

import argparse
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.fspath(BASE_DIR))

DB_PATH = BASE_DIR / "Data" / "PlayerStats.sqlite"
MIN_DELAY_SECONDS = 1
MAX_DELAY_SECONDS = 3


class PlayerStatsAPI:
    """Класс для работы с NBA Stats API"""
    
    BASE_URL = "https://stats.nba.com/stats"
    
    HEADERS = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Host": "stats.nba.com",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true"
    }
    
    @classmethod
    def get_player_game_logs(cls, player_id, season="2024-25"):
        """Получить игровую статистику конкретного игрока"""
        url = f"{cls.BASE_URL}/playergamelogs"
        params = {
            "PlayerID": player_id,
            "Season": season,
            "SeasonType": "Regular Season",
            "LeagueID": "00"
        }
        
        try:
            response = requests.get(url, headers=cls.HEADERS, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                result_set = data['resultSets'][0]
                df = pd.DataFrame(result_set['rowSet'], columns=result_set['headers'])
                return df
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Ошибка получения данных игрока {player_id}: {e}")
            return pd.DataFrame()
    
    @classmethod
    def get_all_players(cls, season="2024-25"):
        """Получить список всех активных игроков"""
        url = f"{cls.BASE_URL}/leaguedashplayerstats"
        params = {
            "Season": season,
            "SeasonType": "Regular Season",
            "PerMode": "PerGame",
            "LeagueID": "00",
            "MeasureType": "Base"
        }
        
        try:
            response = requests.get(url, headers=cls.HEADERS, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                result_set = data['resultSets'][0]
                df = pd.DataFrame(result_set['rowSet'], columns=result_set['headers'])
                return df
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Ошибка получения списка игроков: {e}")
            return pd.DataFrame()
    
    @classmethod
    def get_player_stats_today(cls, date_str=None):
        """Получить статистику всех игроков на конкретную дату"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        url = f"{cls.BASE_URL}/leaguedashplayerstats"
        params = {
            "DateFrom": date_str,
            "DateTo": date_str,
            "Season": "2024-25",
            "SeasonType": "Regular Season",
            "PerMode": "Totals",
            "LeagueID": "00"
        }
        
        try:
            response = requests.get(url, headers=cls.HEADERS, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                result_set = data['resultSets'][0]
                df = pd.DataFrame(result_set['rowSet'], columns=result_set['headers'])
                return df
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Ошибка получения статистики на {date_str}: {e}")
            return pd.DataFrame()


def create_database():
    """Создать структуру базы данных для статистики игроков"""
    with sqlite3.connect(DB_PATH) as con:
        # Таблица с информацией об игроках
        con.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id INTEGER PRIMARY KEY,
                player_name TEXT NOT NULL,
                team_id INTEGER,
                team_name TEXT,
                position TEXT,
                height TEXT,
                weight TEXT,
                age INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица с игровой статистикой
        con.execute("""
            CREATE TABLE IF NOT EXISTS game_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                game_id TEXT NOT NULL,
                game_date DATE NOT NULL,
                matchup TEXT,
                home_away TEXT,
                opponent TEXT,
                minutes_played REAL,
                points INTEGER,
                rebounds INTEGER,
                assists INTEGER,
                steals INTEGER,
                blocks INTEGER,
                turnovers INTEGER,
                fgm INTEGER,
                fga INTEGER,
                fg_pct REAL,
                fg3m INTEGER,
                fg3a INTEGER,
                fg3_pct REAL,
                ftm INTEGER,
                fta INTEGER,
                ft_pct REAL,
                plus_minus INTEGER,
                FOREIGN KEY (player_id) REFERENCES players(player_id),
                UNIQUE(player_id, game_id)
            )
        """)
        
        # Таблица со средней статистикой (последние N игр)
        con.execute("""
            CREATE TABLE IF NOT EXISTS rolling_averages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                date DATE NOT NULL,
                games_count INTEGER,
                avg_points REAL,
                avg_minutes REAL,
                avg_rebounds REAL,
                avg_assists REAL,
                consistency_score REAL,
                trend REAL,
                FOREIGN KEY (player_id) REFERENCES players(player_id),
                UNIQUE(player_id, date, games_count)
            )
        """)
        
        con.commit()
        print("✓ База данных создана успешно")


def calculate_rolling_stats(player_id, games_window=10):
    """Рассчитать скользящие средние для игрока"""
    with sqlite3.connect(DB_PATH) as con:
        query = """
            SELECT game_date, points, minutes_played, rebounds, assists
            FROM game_stats
            WHERE player_id = ?
            ORDER BY game_date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, con, params=[player_id, games_window])
        
        if df.empty or len(df) < 3:
            return None
        
        stats = {
            'player_id': player_id,
            'date': datetime.now().date(),
            'games_count': len(df),
            'avg_points': df['points'].mean(),
            'avg_minutes': df['minutes_played'].mean(),
            'avg_rebounds': df['rebounds'].mean(),
            'avg_assists': df['assists'].mean(),
            'consistency_score': 1 / (df['points'].std() + 1),  # Чем меньше разброс, тем лучше
            'trend': (df['points'].iloc[:3].mean() - df['points'].iloc[-3:].mean())  # Тренд
        }
        
        return stats


def update_player_info():
    """Обновить информацию о всех активных игроках"""
    print("Получение списка активных игроков...")
    df = PlayerStatsAPI.get_all_players()
    
    if df.empty:
        print("Не удалось получить список игроков")
        return
    
    with sqlite3.connect(DB_PATH) as con:
        for _, row in df.iterrows():
            con.execute("""
                INSERT OR REPLACE INTO players 
                (player_id, player_name, team_id, team_name, age, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row.get('PLAYER_ID'),
                row.get('PLAYER_NAME'),
                row.get('TEAM_ID'),
                row.get('TEAM_ABBREVIATION'),
                row.get('AGE'),
                datetime.now()
            ))
        con.commit()
    
    print(f"✓ Обновлено {len(df)} игроков")


def fetch_recent_games(days_back=30):
    """Получить статистику игр за последние N дней"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"Получение статистики с {start_date} по {end_date}...")
    
    current_date = start_date
    games_added = 0
    
    while current_date <= end_date:
        print(f"Обработка {current_date}...")
        df = PlayerStatsAPI.get_player_stats_today(current_date.strftime("%Y-%m-%d"))
        
        if not df.empty:
            with sqlite3.connect(DB_PATH) as con:
                for _, row in df.iterrows():
                    try:
                        # Извлекаем данные из строки
                        matchup = row.get('MATCHUP', '')
                        home_away = 'HOME' if 'vs.' in matchup else 'AWAY'
                        opponent = matchup.split('vs.' if home_away == 'HOME' else '@')[-1].strip()
                        
                        con.execute("""
                            INSERT OR IGNORE INTO game_stats
                            (player_id, game_id, game_date, matchup, home_away, opponent,
                             minutes_played, points, rebounds, assists, steals, blocks, turnovers,
                             fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, plus_minus)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            row.get('PLAYER_ID'),
                            f"{row.get('PLAYER_ID')}_{current_date}",
                            current_date,
                            matchup,
                            home_away,
                            opponent,
                            row.get('MIN', 0),
                            row.get('PTS', 0),
                            row.get('REB', 0),
                            row.get('AST', 0),
                            row.get('STL', 0),
                            row.get('BLK', 0),
                            row.get('TOV', 0),
                            row.get('FGM', 0),
                            row.get('FGA', 0),
                            row.get('FG_PCT', 0),
                            row.get('FG3M', 0),
                            row.get('FG3A', 0),
                            row.get('FG3_PCT', 0),
                            row.get('FTM', 0),
                            row.get('FTA', 0),
                            row.get('FT_PCT', 0),
                            row.get('PLUS_MINUS', 0)
                        ))
                        games_added += 1
                    except Exception as e:
                        print(f"Ошибка добавления игры: {e}")
                        continue
                
                con.commit()
            
            print(f"  Добавлено {len(df)} записей")
        
        current_date += timedelta(days=1)
        time.sleep(random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
    
    print(f"✓ Всего добавлено {games_added} игр")


def update_rolling_averages(windows=[5, 10, 15]):
    """Обновить скользящие средние для всех игроков"""
    with sqlite3.connect(DB_PATH) as con:
        cursor = con.execute("SELECT player_id FROM players")
        player_ids = [row[0] for row in cursor.fetchall()]
    
    print(f"Расчёт скользящих средних для {len(player_ids)} игроков...")
    
    for player_id in player_ids:
        for window in windows:
            stats = calculate_rolling_stats(player_id, window)
            if stats:
                with sqlite3.connect(DB_PATH) as con:
                    con.execute("""
                        INSERT OR REPLACE INTO rolling_averages
                        (player_id, date, games_count, avg_points, avg_minutes, 
                         avg_rebounds, avg_assists, consistency_score, trend)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        stats['player_id'],
                        stats['date'],
                        stats['games_count'],
                        stats['avg_points'],
                        stats['avg_minutes'],
                        stats['avg_rebounds'],
                        stats['avg_assists'],
                        stats['consistency_score'],
                        stats['trend']
                    ))
                    con.commit()
    
    print("✓ Скользящие средние обновлены")


def main():
    parser = argparse.ArgumentParser(description="Получение статистики игроков NBA")
    parser.add_argument("--init", action="store_true", help="Инициализировать базу данных")
    parser.add_argument("--update-players", action="store_true", help="Обновить список игроков")
    parser.add_argument("--fetch-games", type=int, default=30, help="Получить игры за N дней")
    parser.add_argument("--calc-averages", action="store_true", help="Рассчитать скользящие средние")
    parser.add_argument("--full-update", action="store_true", help="Полное обновление всех данных")
    
    args = parser.parse_args()
    
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if args.init or not DB_PATH.exists():
        create_database()
    
    if args.full_update:
        update_player_info()
        fetch_recent_games(args.fetch_games)
        update_rolling_averages()
    else:
        if args.update_players:
            update_player_info()
        if args.fetch_games:
            fetch_recent_games(args.fetch_games)
        if args.calc_averages:
            update_rolling_averages()


if __name__ == "__main__":
    main()
