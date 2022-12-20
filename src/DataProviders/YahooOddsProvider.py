import requests
import json

class YahooOddsProvider:
    data_url = 'https://sports.yahoo.com/nba/odds/'

    def __init__(self, data_url):
       self.data_url = data_url

    
    def get_odds(self):
        """Function returning odds from yahoo server's json content

        Returns:
            dictionary: [teamName: { money_line_odds: val, under_over_odds: val }]
        """
        dict_res = {}
        response_data = requests.get(self.data_url)    
        startIndex = response_data.text.find('SportsbookStore') + len('SportsbookStore":')
        endIndex = response_data.text.find('ScoresStore') - 2
        sportsbook_data = json.loads(response_data.text[startIndex:endIndex])

        for game_name in sportsbook_data['state']['leagueOddsList']:
            game = sportsbook_data['state']['leagueOddsByGame'][game_name]

            # Get team names
            home_team_name = game['homeTeam']['location'] + ' ' + game['homeTeam']['nickname']
            away_team_name = game['awayTeam']['location'] + ' ' + game['awayTeam']['nickname']
            
            # Get money line bet values
            money_line_bet = next((x for x in game['bets'] if x['baseCategory'] == 'MONEY_LINE'), None)
            if money_line_bet is not None: 
                money_line_home_value = next((x for x in money_line_bet['options'] if x['name'] == home_team_name))['americanOdds']
                money_line_away_value = next((x for x in money_line_bet['options'] if x['name'] == away_team_name))['americanOdds']
            else: 
                continue
            
            # Get totals bet value
            totals_bet = next((x for x in game['bets'] if x['baseCategory'] == 'TOTALS'), None)
            if totals_bet is not None: 
                totals_value = float(totals_bet['options'][0]['optionDetails'][0]['value'])
            else: 
                continue
            
            dict_res[home_team_name + ':' + away_team_name] =  { 
                'under_over_odds': totals_value,
                home_team_name: { 'money_line_odds': money_line_home_value }, 
                away_team_name: { 'money_line_odds': money_line_away_value }
            }

        return dict_res