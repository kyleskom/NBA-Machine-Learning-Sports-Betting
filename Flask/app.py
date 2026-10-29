from datetime import date
from flask import Flask, render_template,jsonify
from functools import lru_cache
import os
from pathlib import Path
import sys
import requests
import time
import toml

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from main import predict_board
from src.Predict.service import template_games
from src.Utils.season import runtime_season


RAPIDAPI_BASE_URL = "https://tank01-fantasy-stats.p.rapidapi.com"
RAPIDAPI_HOST = "tank01-fantasy-stats.p.rapidapi.com"
REQUEST_TIMEOUT_SECONDS = 15


@lru_cache()
def fetch_fanduel(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="fanduel")

@lru_cache()
def fetch_draftkings(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="draftkings")

@lru_cache()
def fetch_betmgm(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="betmgm")

def fetch_game_data(sportsbook="fanduel"):
    return template_games(predict_board(sportsbook))


def get_ttl_hash(seconds=600):
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)


app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')


@app.route("/")
def index():
    fanduel = fetch_fanduel(ttl_hash=get_ttl_hash())
    draftkings = fetch_draftkings(ttl_hash=get_ttl_hash())
    betmgm = fetch_betmgm(ttl_hash=get_ttl_hash())

    return render_template('index.html', today=date.today(), data={"fanduel": fanduel, "draftkings": draftkings, "betmgm": betmgm})


@app.route('/api/predictions/<sportsbook>')
def api_predictions(sportsbook):
    return jsonify(predict_board(sportsbook))




def rapidapi_headers():
    key = os.environ.get("RAPIDAPI_KEY")
    if not key:
        return None
    return {"x-rapidapi-key": key, "x-rapidapi-host": RAPIDAPI_HOST}


def current_season_start_year():
    config = toml.load(BASE_DIR / "config.toml")
    return runtime_season(config).split("-", 1)[0]


def rapidapi_get(path, params):
    headers = rapidapi_headers()
    if headers is None:
        return None, "RAPIDAPI_KEY is not configured"
    try:
        response = requests.get(f"{RAPIDAPI_BASE_URL}/{path}", headers=headers, params=params,
                                timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json(), None
    except (requests.RequestException, ValueError) as exc:
        return None, str(exc)


def get_player_data(team_abv):
    """Fetch and normalize a team roster for the roster modal."""
    data, error = rapidapi_get("getNBATeamRoster", {"teamAbv": team_abv})
    if error or data.get("statusCode") != 200:
        return {"success": False, "error": error or "Failed to fetch team data"}
    players = []
    for player in data.get("body", {}).get("roster", []):
        injury = player.get("injury") or {}
        injury_status = injury.get("designation") or "Healthy"
        if injury_status != "Healthy" and injury.get("description"):
            injury_status += f" - {injury['description']}"
        players.append({
            "name": player.get("longName"), "shortName": player.get("shortName"),
            "headshot": player.get("nbaComHeadshot"), "injury": injury_status,
            "position": player.get("pos"), "height": player.get("height"),
            "weight": player.get("weight"), "college": player.get("college"),
            "experience": player.get("exp"), "jerseyNum": player.get("jerseyNum"),
            "playerId": player.get("playerID"), "birthDate": player.get("bDay"),
        })
    return {"success": True, "players": players}


@app.route("/team-data/<team_name>")
def team_data(team_name):
    # Convert full team name to abbreviation using the existing dictionary
    team_abv = team_abbreviations.get(team_name)
    
    if not team_abv:
        return jsonify({
            'success': False,
            'error': f'Team abbreviation not found for {team_name}'
        })
    
    # Fetch and return the player data
    result = get_player_data(team_abv)
    return jsonify(result)


    
@app.route("/player-stats/<player_id>")
def player_stats(player_id):
    info_data, info_error = rapidapi_get("getNBAPlayerInfo", {"playerID": player_id})
    games_data, games_error = rapidapi_get("getNBAGamesForPlayer", {
        "playerID": player_id, "season": current_season_start_year(),
    })
    if info_error or games_error:
        return jsonify({"success": False, "error": info_error or games_error})
    if info_data.get('statusCode') == 200 and games_data.get('statusCode') == 200:
        games = list(games_data['body'].values())
        games.sort(key=lambda x: x['gameID'], reverse=True)
        recent_games = games[:10]
        player_info = info_data['body']
        injury_status = player_info.get('injury') or "Healthy"
        return jsonify({
            'success': True,
            'games': recent_games,
            'player': {
                'name': player_info.get('longName'), 'position': player_info.get('pos'),
                'number': player_info.get('jerseyNum'), 'height': player_info.get('height'),
                'weight': player_info.get('weight'), 'team': player_info.get('team'),
                'college': player_info.get('college'), 'experience': player_info.get('exp'),
                'headshot': player_info.get('nbaComHeadshot'), 'injury': injury_status,
            },
        })
    return jsonify({'success': False, 'error': 'Failed to fetch player data'})

        
team_abbreviations = {
    'Orlando Magic': 'ORL',
    'Minnesota Timberwolves': 'MIN',
    'Miami Heat': 'MIA',
    'Boston Celtics': 'BOS',
    'LA Clippers': 'LAC',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Atlanta Hawks': 'ATL',
    'Cleveland Cavaliers': 'CLE',
    'Toronto Raptors': 'TOR',
    'Washington Wizards': 'WAS',
    'Phoenix Suns': 'PHO',
    'San Antonio Spurs': 'SA',
    'Chicago Bulls': 'CHI',
    'Charlotte Hornets': 'CHA',
    'Philadelphia 76ers': 'PHI',
    'New Orleans Pelicans': 'NO',
    'Sacramento Kings': 'SAC',
    'Dallas Mavericks': 'DAL',
    'Houston Rockets': 'HOU',
    'Brooklyn Nets': 'BKN',
    'New York Knicks': 'NY',
    'Utah Jazz': 'UTA',
    'Oklahoma City Thunder': 'OKC',
    'Portland Trail Blazers': 'POR',
    'Indiana Pacers': 'IND',
    'Milwaukee Bucks': 'MIL',
    'Golden State Warriors': 'GS',
    'Memphis Grizzlies': 'MEM',
    'Los Angeles Lakers': 'LAL'
}
