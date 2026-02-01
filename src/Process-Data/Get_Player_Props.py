"""
Модуль для получения player props (over/under) с букмекеров.
Поддерживает несколько источников бесплатных данных.
"""

import argparse
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "Data" / "PlayerProps.sqlite"


class PlayerPropsAPI:
    """Класс для работы с API букмекерских линий"""
    
    @staticmethod
    def get_props_from_prizepicks():
        """
        PrizePicks - популярная DFS платформа с бесплатным API
        Предоставляет линии для различных статов игроков
        """
        url = "https://api.prizepicks.com/projections"
        params = {
            "league_id": 7  # 7 = NBA
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            props = []
            
            if 'data' in data:
                for item in data['data']:
                    if item.get('attributes', {}).get('stat_type') == 'Points':
                        props.append({
                            'player_id': item.get('relationships', {}).get('new_player', {}).get('data', {}).get('id'),
                            'player_name': None,  # Нужно джойнить с другой таблицей
                            'line': item.get('attributes', {}).get('line_score'),
                            'stat_type': 'Points',
                            'game_date': item.get('attributes', {}).get('start_time'),
                            'source': 'prizepicks'
                        })
            
            # Получаем имена игроков
            if 'included' in data:
                player_map = {}
                for item in data['included']:
                    if item.get('type') == 'new_player':
                        player_map[item['id']] = item.get('attributes', {}).get('display_name')
                
                for prop in props:
                    if prop['player_id'] in player_map:
                        prop['player_name'] = player_map[prop['player_id']]
            
            return props
            
        except Exception as e:
            print(f"Ошибка получения данных PrizePicks: {e}")
            return []
    
    @staticmethod
    def get_props_from_underdog():
        """
        Underdog Fantasy - ещё одна DFS платформа
        """
        url = "https://api.underdogfantasy.com/beta/v3/over_under_lines"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            props = []
            
            if 'over_under_lines' in data:
                for item in data['over_under_lines']:
                    if item.get('stat') == 'Points' and item.get('sport') == 'NBA':
                        props.append({
                            'player_id': None,
                            'player_name': item.get('player_name'),
                            'line': item.get('stat_value'),
                            'stat_type': 'Points',
                            'game_date': item.get('match', {}).get('start_time'),
                            'over_odds': item.get('over_odds'),
                            'under_odds': item.get('under_odds'),
                            'source': 'underdog'
                        })
            
            return props
            
        except Exception as e:
            print(f"Ошибка получения данных Underdog: {e}")
            return []
    
    @staticmethod
    def get_props_from_odds_api():
        """
        The Odds API - агрегатор букмекерских линий
        Требует API ключ, но есть бесплатный tier (500 запросов/мес)
        """
        # Можно получить бесплатный ключ на https://the-odds-api.com/
        api_key = "YOUR_API_KEY_HERE"  # Пользователь должен заменить
        
        if api_key == "YOUR_API_KEY_HERE":
            print("⚠ Установите API ключ для The Odds API")
            return []
        
        url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "player_points",
            "oddsFormat": "american"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            props = []
            
            for game in data:
                for bookmaker in game.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'player_points':
                            for outcome in market.get('outcomes', []):
                                props.append({
                                    'player_name': outcome.get('description'),
                                    'line': outcome.get('point'),
                                    'stat_type': 'Points',
                                    'over_odds': outcome.get('price') if outcome.get('name') == 'Over' else None,
                                    'under_odds': outcome.get('price') if outcome.get('name') == 'Under' else None,
                                    'game_date': game.get('commence_time'),
                                    'bookmaker': bookmaker.get('key'),
                                    'source': 'odds_api'
                                })
            
            return props
            
        except Exception as e:
            print(f"Ошибка получения данных The Odds API: {e}")
            return []


def create_database():
    """Создать структуру базы данных для player props"""
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS player_props (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                player_id INTEGER,
                line REAL NOT NULL,
                stat_type TEXT DEFAULT 'Points',
                over_odds INTEGER,
                under_odds INTEGER,
                game_date TIMESTAMP,
                source TEXT,
                bookmaker TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_name, line, game_date, source)
            )
        """)
        
        # Индексы для быстрого поиска
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_date 
            ON player_props(player_name, game_date)
        """)
        
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_date 
            ON player_props(game_date)
        """)
        
        con.commit()
        print("✓ База данных player props создана")


def fetch_all_props():
    """Получить props из всех доступных источников"""
    all_props = []
    
    print("Получение props из PrizePicks...")
    all_props.extend(PlayerPropsAPI.get_props_from_prizepicks())
    time.sleep(1)
    
    print("Получение props из Underdog...")
    all_props.extend(PlayerPropsAPI.get_props_from_underdog())
    time.sleep(1)
    
    print("Получение props из The Odds API...")
    all_props.extend(PlayerPropsAPI.get_props_from_odds_api())
    
    return all_props


def save_props_to_db(props):
    """Сохранить props в базу данных"""
    if not props:
        print("Нет данных для сохранения")
        return
    
    with sqlite3.connect(DB_PATH) as con:
        saved = 0
        for prop in props:
            try:
                con.execute("""
                    INSERT OR IGNORE INTO player_props
                    (player_name, player_id, line, stat_type, over_odds, under_odds, 
                     game_date, source, bookmaker)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prop.get('player_name'),
                    prop.get('player_id'),
                    prop.get('line'),
                    prop.get('stat_type', 'Points'),
                    prop.get('over_odds'),
                    prop.get('under_odds'),
                    prop.get('game_date'),
                    prop.get('source'),
                    prop.get('bookmaker')
                ))
                saved += 1
            except Exception as e:
                print(f"Ошибка сохранения prop: {e}")
        
        con.commit()
    
    print(f"✓ Сохранено {saved} props")


def get_todays_props():
    """Получить props на сегодняшние игры"""
    today = datetime.now().date()
    
    with sqlite3.connect(DB_PATH) as con:
        query = """
            SELECT player_name, line, over_odds, under_odds, source, bookmaker
            FROM player_props
            WHERE DATE(game_date) = ?
            ORDER BY player_name, line
        """
        df = pd.read_sql_query(query, con, params=[today])
    
    return df


def get_props_for_player(player_name, days_ahead=1):
    """Получить все доступные линии для конкретного игрока"""
    end_date = datetime.now() + timedelta(days=days_ahead)
    
    with sqlite3.connect(DB_PATH) as con:
        query = """
            SELECT * FROM player_props
            WHERE player_name LIKE ?
            AND game_date <= ?
            AND game_date >= CURRENT_TIMESTAMP
            ORDER BY game_date, source
        """
        df = pd.read_sql_query(
            query, 
            con, 
            params=[f"%{player_name}%", end_date]
        )
    
    return df


def aggregate_props():
    """
    Агрегировать props от разных букмекеров
    Находит среднюю линию и лучшие odds
    """
    today = datetime.now().date()
    
    with sqlite3.connect(DB_PATH) as con:
        query = """
            SELECT 
                player_name,
                AVG(line) as avg_line,
                MIN(line) as min_line,
                MAX(line) as max_line,
                COUNT(DISTINCT source) as sources_count,
                MAX(over_odds) as best_over_odds,
                MAX(under_odds) as best_under_odds
            FROM player_props
            WHERE DATE(game_date) = ?
            GROUP BY player_name
            HAVING sources_count >= 2
            ORDER BY player_name
        """
        df = pd.read_sql_query(query, con, params=[today])
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Получение player props линий")
    parser.add_argument("--init", action="store_true", help="Инициализировать БД")
    parser.add_argument("--fetch", action="store_true", help="Получить актуальные props")
    parser.add_argument("--show-today", action="store_true", help="Показать сегодняшние props")
    parser.add_argument("--player", type=str, help="Показать props для игрока")
    parser.add_argument("--aggregate", action="store_true", help="Агрегировать props")
    
    args = parser.parse_args()
    
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if args.init or not DB_PATH.exists():
        create_database()
    
    if args.fetch:
        props = fetch_all_props()
        save_props_to_db(props)
    
    if args.show_today:
        df = get_todays_props()
        if not df.empty:
            print("\n=== Сегодняшние Player Props ===")
            print(df.to_string(index=False))
        else:
            print("Нет props на сегодня")
    
    if args.player:
        df = get_props_for_player(args.player)
        if not df.empty:
            print(f"\n=== Props для {args.player} ===")
            print(df.to_string(index=False))
        else:
            print(f"Нет props для {args.player}")
    
    if args.aggregate:
        df = aggregate_props()
        if not df.empty:
            print("\n=== Агрегированные Props ===")
            print(df.to_string(index=False))
        else:
            print("Нет данных для агрегации")


if __name__ == "__main__":
    main()
