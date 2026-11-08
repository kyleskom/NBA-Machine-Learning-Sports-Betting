"""Stable NBA franchise IDs with explicit historical/display aliases."""
import re
import unicodedata

_NAMES = [
    (1610612737, 'ATL', 'Atlanta Hawks'), (1610612738, 'BOS', 'Boston Celtics'),
    (1610612751, 'BKN', 'Brooklyn Nets'), (1610612766, 'CHA', 'Charlotte Hornets'),
    (1610612741, 'CHI', 'Chicago Bulls'), (1610612739, 'CLE', 'Cleveland Cavaliers'),
    (1610612742, 'DAL', 'Dallas Mavericks'), (1610612743, 'DEN', 'Denver Nuggets'),
    (1610612765, 'DET', 'Detroit Pistons'), (1610612744, 'GSW', 'Golden State Warriors'),
    (1610612745, 'HOU', 'Houston Rockets'), (1610612754, 'IND', 'Indiana Pacers'),
    (1610612746, 'LAC', 'LA Clippers'), (1610612747, 'LAL', 'Los Angeles Lakers'),
    (1610612763, 'MEM', 'Memphis Grizzlies'), (1610612748, 'MIA', 'Miami Heat'),
    (1610612749, 'MIL', 'Milwaukee Bucks'), (1610612750, 'MIN', 'Minnesota Timberwolves'),
    (1610612740, 'NOP', 'New Orleans Pelicans'), (1610612752, 'NYK', 'New York Knicks'),
    (1610612760, 'OKC', 'Oklahoma City Thunder'), (1610612753, 'ORL', 'Orlando Magic'),
    (1610612755, 'PHI', 'Philadelphia 76ers'), (1610612756, 'PHX', 'Phoenix Suns'),
    (1610612757, 'POR', 'Portland Trail Blazers'), (1610612758, 'SAC', 'Sacramento Kings'),
    (1610612759, 'SAS', 'San Antonio Spurs'), (1610612761, 'TOR', 'Toronto Raptors'),
    (1610612762, 'UTA', 'Utah Jazz'), (1610612764, 'WAS', 'Washington Wizards'),
]
TEAM_NAMES = {i: n for i, a, n in _NAMES}
ALIASES = {str(x).casefold(): i for i, a, n in _NAMES for x in (i, a, n)}
for alias, canonical in {'Los Angeles Clippers':'LAC','NJN':'BKN','New Jersey Nets':'BKN',
    'Charlotte Bobcats':'CHA','CHH':'CHA','NOH':'NOP','NOK':'NOP','New Orleans Hornets':'NOP',
    'New Orleans/Oklahoma City Hornets':'NOP','SEA':'OKC','Seattle SuperSonics':'OKC',
    'GS':'GSW','NY':'NYK','SA':'SAS','NO':'NOP','PHO':'PHX','WSH':'WAS','UTAH':'UTA'}.items():
    ALIASES[alias.casefold()] = ALIASES[canonical.casefold()]


def team_id(value):
    key = str(value).strip().casefold()
    if key.endswith('.0'):
        key = key[:-2]
    if key not in ALIASES:
        raise ValueError(f'Unknown NBA team identity: {value!r}')
    return ALIASES[key]


def team_name(value):
    return TEAM_NAMES[team_id(value)]


def name_key(value):
    value = unicodedata.normalize('NFKD', str(value))
    return re.sub(r'[^a-z0-9]', '', value.encode('ascii', 'ignore').decode().lower())
