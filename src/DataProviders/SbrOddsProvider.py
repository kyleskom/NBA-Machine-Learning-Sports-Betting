from sbrscrape import Scoreboard

class SbrOddsProvider:
    
    """ Abbreviations dictionary for team location which are sometimes saved with abbrev instead of full name. 
    Moneyline options name require always full name
    Returns:
        string: Full location name
    """    

    def __init__(self, sportsbook="fanduel"):
        
       self.games = Scoreboard(sport="NBA").games
       self.sportsbook = sportsbook

    
    def get_odds(self):
        """Function returning odds from Sbr server's json content

        Returns:
            dictionary: [home_team_name + ':' + away_team_name: { home_team: money_line_odds, away_team: money_line_odds }, under_over_odds: val]
        """
        dict_res = {}
        for game in self.games:
            # Get team names
            home_team_name = game['home_team'].replace("Los Angeles Clippers", "LA Clippers")
            away_team_name = game['away_team'].replace("Los Angeles Clippers", "LA Clippers")
            
            # Get money line bet values
            money_line_home_value = game['home_ml'][self.sportsbook]
            money_line_away_value = game['away_ml'][self.sportsbook]
            
            # Get totals bet value
            totals_value = game['total'][self.sportsbook]
            
            dict_res[home_team_name + ':' + away_team_name] =  { 
                'under_over_odds': totals_value,
                home_team_name: { 'money_line_odds': money_line_home_value }, 
                away_team_name: { 'money_line_odds': money_line_away_value }
            }
        return dict_res