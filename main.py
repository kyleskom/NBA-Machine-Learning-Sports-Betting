from src.tools import get_teams_playing_today, get_json_data, to_data_frame


todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2019/scores/00_todays_scores.json'

data = get_json_data(todays_games_url)
df = to_data_frame(data)

url = 'https://stats.nba.com/stats/leaguedashteamstats?' \
      'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
      'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
      'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' \
      'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
      'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
      'Season=2019-20&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
      'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='

data = get_json_data(url)
df = to_data_frame(data)

games = get_teams_playing_today()
print('DONE')
