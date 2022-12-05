from src.Utils.tools import get_odds
import json
from datetime import datetime

# api_key = 
url = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds/'\
    '?apiKey=2db533701475e3191e96b9f9115053aa&regions=us&markets=h2h,totals'\
        '&oddsFormat=american&bookmakers=fanduel,betmgm,draftkings'

def parse_odds():
    return
def generate_odds():
    games = get_odds(url)
    output = []
    for game in games:
        try:
            fanduel = next((x for x in game['bookmakers'] if x['key'] == 'fanduel'), None)
            fanduel_h2h = next((x for x in fanduel['markets'] if x['key'] == 'h2h'), None)
            fanduel_total = next((x for x in fanduel['markets'] if x['key'] == 'totals'), None)

            output.append(
                {
                    'id': [game['home_team'], game['away_team']],
                    'home':  next((x for x in fanduel_h2h['outcomes'] if x['name'] == game['home_team']), None)['price'],
                    'away':  next((x for x in fanduel_h2h['outcomes'] if x['name'] == game['away_team']), None)['price'],
                    'o/u': fanduel_total['outcomes'][0]['point']
                }
            )
        except:
            print('Error for', game)
    with open('./Odds-Input/odds-input-'+ datetime.today().strftime('%Y-%m-%d') + '.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

def main():
   generate_odds()

if __name__ == '__main__':
    main()