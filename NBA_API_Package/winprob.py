from nba_api.stats.endpoints import winprobabilitypbp
from Proxy_Service.load_proxy import _get_proxy

proxy = _get_proxy()

# headers
custom_headers = {
    'Host': 'stats.nba.com',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}

# basic request
win_prob = winprobabilitypbp.WinProbabilityPBP(game_id='0022400307', timeout=30, proxy=proxy)

print(win_prob.get_json())
