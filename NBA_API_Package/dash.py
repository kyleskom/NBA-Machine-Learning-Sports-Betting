from nba_api.stats.endpoints import leaguedashteamstats

# basic request
league_dash = leaguedashteamstats.LeagueDashTeamStats()

print(league_dash.get_json())