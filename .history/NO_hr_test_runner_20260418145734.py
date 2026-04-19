"""
No-HR Game Prediction Backtest Runner — Grid Search Edition
Fetches all data ONCE per date, then sweeps all weight/blend combinations
in memory (no extra API calls per config).

Goal: predict which games will have ZERO home runs hit.
Method: for each game, compute P(no HR) = ∏(1 - hr_prob_i) across all batters,
then flag the top N games with highest no-HR probability each day.
Metric: accuracy — what % of flagged games actually had 0 HRs.

Special cases:
  Tampa Bay  — always uses 2024 Savant PF (Tropicana Field; 2025 was temp stadium)
  Athletics  — only 2025/2026 Savant PF; no hardcoded Coliseum fallback (new stadium 2025)
  2026 stats — only included for dates >= 2026-04-01

Usage: python no_hr_game_backtest_runner.py
"""
import math, time, io, json, os, itertools
from datetime import datetime
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from requests.adapters import HTTPAdapter
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})
_adapter = HTTPAdapter(pool_connections=20, pool_maxsize=50)
session.mount('https://', _adapter)
session.mount('http://',  _adapter)

# ── fixed parameters ───────────────────────────────────────────────────────────
PITCHER_SEASON   = 2025
LEAGUE_HR_PA     = 0.034
LEAGUE_HR9       = 1.05
AVG_PA_GAME      = 3.0
MIN_PA           = 30
MIN_SPLIT_PA     = 20    # minimum PA vs a handedness to use individual split data
MIN_IP           = 20.0
CALIBRATION      = 0.95
MAX_PITCHER_VULN = 1.7
LEAGUE_BRL_PA    = 0.0615
LEAGUE_EXIT_VELO = 88.5
LEAGUE_HARD_HIT  = 0.385
MIN_BVP_PA       = 3     # minimum career PA vs a pitcher to use BvP data
MIN_HA_PA        = 20    # minimum home or away PA to use home/away split
GAME_UTC_HOUR    = 20        # ~7 PM EDT for weather index
DAY_GAME_CUTOFF  = 18        # before 18:00 UTC → day game

SPRING_START = '2026-02-20'
SPRING_END   = '2026-03-25'
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SPRING_CACHE = os.path.join(_SCRIPT_DIR, 'spring_2026_cache.json')
CACHE_DIR    = _SCRIPT_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

PITCH_MATCHUP_CAP = (0.70, 1.40)
BAT_TRACKING_CAP  = (0.75, 1.35)
WEATHER_CAP       = (0.75, 1.35)
SPRING_CAP        = (0.85, 1.20)
PARK_CAP          = (0.50, 2.00)
POWER_CAP         = (0.35, 2.00)
RECENT_FORM_CAP   = (0.25, 4.00)
BVP_CAP           = (0.50, 2.00)
HOME_AWAY_CAP     = (0.60, 1.75)
WEATHER_HIST_CAP  = (0.80, 1.25)  # historical HR rate in temp/wind bucket

PITCH_GROUPS = {
    'FF':'FB','FT':'FB','SI':'FB','FC':'FB',
    'SL':'SL','ST':'SL',
    'CU':'CB','KC':'CB','SV':'CB',
    'CH':'CH','FS':'CH','FO':'CH',
}

# ── grid search configurations ─────────────────────────────────────────────────
SEASON_WEIGHT_CONFIGS = [
    ('25×1  + 26×1',    {2025:  1.00, 2026: 1.00}),
]

PF_BLEND_CONFIGS = [
    ('PF: 85/15',      {2025: 0.85, 2026: 0.15}),
]

# ── tuning grid ────────────────────────────────────────────────────────────────
TUNE_PARAM_GRID = {
    'games_to_pick':  [1, 2, 3, 5],         # how many low-HR-prob games to flag per day
    'pwr_brl_exp':    [0.10, 0.20, 0.35, 0],# barrel rate exponent
    'pwr_ev_exp':     [0.15, 0.30, 0.50, 0],# exit velo exponent
    'form_cap_hi':    [1.50, 2.00, 3.00, 0],# recent batter form cap (high)
    'form_min_pa':    [10, 20, 30],          # min recent PA to activate form signal
    'vuln_cap':       [1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20],
    'matchup_scale':  [0.03, 0.05, 0.08, 0.01, 0],
    'bvp_weight':     [0.0, 0.25, 0.5, 0.75, 1.0],
    'calibration':    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

_keys   = list(TUNE_PARAM_GRID.keys())
_values = list(TUNE_PARAM_GRID.values())
ALL_TUNE_CONFIGS = [dict(zip(_keys, combo)) for combo in itertools.product(*_values)]
print(f'Tuning grid: {len(ALL_TUNE_CONFIGS)} configs')

# ── team/stadium tables ────────────────────────────────────────────────────────
CLUB_TO_TEAM = {
    'D-backs':'Arizona Diamondbacks','Braves':'Atlanta Braves',
    'Orioles':'Baltimore Orioles','Red Sox':'Boston Red Sox',
    'Cubs':'Chicago Cubs','White Sox':'Chicago White Sox',
    'Reds':'Cincinnati Reds','Guardians':'Cleveland Guardians',
    'Rockies':'Colorado Rockies','Tigers':'Detroit Tigers',
    'Astros':'Houston Astros','Royals':'Kansas City Royals',
    'Angels':'Los Angeles Angels','Dodgers':'Los Angeles Dodgers',
    'Marlins':'Miami Marlins','Brewers':'Milwaukee Brewers',
    'Twins':'Minnesota Twins','Mets':'New York Mets',
    'Yankees':'New York Yankees','Athletics':'Athletics',
    'Phillies':'Philadelphia Phillies','Pirates':'Pittsburgh Pirates',
    'Padres':'San Diego Padres','Giants':'San Francisco Giants',
    'Mariners':'Seattle Mariners','Cardinals':'St. Louis Cardinals',
    'Rays':'Tampa Bay Rays','Rangers':'Texas Rangers',
    'Blue Jays':'Toronto Blue Jays','Nationals':'Washington Nationals',
}

PLATOON = {
    ('R','L'):1.12,('L','R'):1.08,
    ('R','R'):1.00,('L','L'):0.88,
    ('S','L'):1.05,('S','R'):1.05,
}

# ── Hardcoded day/night park factors ──────────────────────────────────────────
TAMPA_PF_DAY   = 1.06
TAMPA_PF_NIGHT = 0.95

PF_DAY_2025 = {
    'Athletics':             1.07,
    'Colorado Rockies':      1.03,
    'Baltimore Orioles':     1.33,
    'Toronto Blue Jays':     1.12,
    'Philadelphia Phillies': 1.00,
    'Detroit Tigers':        1.09,
    'Los Angeles Dodgers':   1.29,
    'Boston Red Sox':        0.89,
    'Minnesota Twins':       0.89,
    'Chicago Cubs':          1.14,
    'San Francisco Giants':  0.96,
    'New York Yankees':      1.21,
    'Milwaukee Brewers':     0.94,
    'Arizona Diamondbacks':  0.76,
    'Washington Nationals':  0.85,
    'Atlanta Braves':        1.12,
    'New York Mets':         0.98,
    'Cleveland Guardians':   0.86,
    'San Diego Padres':      0.84,
    'Cincinnati Reds':       1.13,
    'Los Angeles Angels':    0.99,
    'Pittsburgh Pirates':    0.57,
    'Chicago White Sox':     1.00,
    'St. Louis Cardinals':   0.73,
    'Kansas City Royals':    0.63,
    'Seattle Mariners':      0.97,
    'Miami Marlins':         0.63,
    'Houston Astros':        0.95,
    'Texas Rangers':         0.68,
}

PF_NIGHT_2025 = {
    'Colorado Rockies':      1.12,
    'Athletics':             1.14,
    'Detroit Tigers':        1.20,
    'Los Angeles Dodgers':   1.40,
    'Arizona Diamondbacks':  0.96,
    'Los Angeles Angels':    1.13,
    'Boston Red Sox':        0.80,
    'Washington Nationals':  1.00,
    'Toronto Blue Jays':     1.24,
    'Cincinnati Reds':       1.09,
    'Minnesota Twins':       0.85,
    'Philadelphia Phillies': 1.27,
    'Atlanta Braves':        1.02,
    'New York Mets':         1.03,
    'Miami Marlins':         1.00,
    'Chicago White Sox':     0.93,
    'Kansas City Royals':    0.96,
    'Houston Astros':        1.10,
    'Milwaukee Brewers':     1.01,
    'San Francisco Giants':  0.76,
    'St. Louis Cardinals':   0.81,
    'Baltimore Orioles':     1.13,
    'Pittsburgh Pirates':    0.73,
    'New York Yankees':      1.04,
    'Chicago Cubs':          1.01,
    'San Diego Padres':      0.95,
    'Cleveland Guardians':   0.90,
    'Texas Rangers':         0.85,
    'Seattle Mariners':      0.98,
}

PF_DAY_2026 = {
    'Houston Astros':        2.29,
    'Seattle Mariners':      2.12,
    'Kansas City Royals':    1.45,
    'Chicago Cubs':          1.30,
    'Philadelphia Phillies': 1.19,
    'Toronto Blue Jays':     1.16,
    'Cincinnati Reds':       1.07,
    'Miami Marlins':         1.04,
    'New York Mets':         0.99,
    'Milwaukee Brewers':     0.98,
    'San Diego Padres':      0.86,
    'San Francisco Giants':  0.83,
    'Arizona Diamondbacks':  0.68,
    'Atlanta Braves':        0.57,
    'St. Louis Cardinals':   0.50,
    'Baltimore Orioles':     0.44,
}

PF_NIGHT_2026 = {
    'Kansas City Royals':    3.30,
    'Arizona Diamondbacks':  1.45,
    'Toronto Blue Jays':     1.89,
    'Baltimore Orioles':     0.72,
    'Seattle Mariners':      1.43,
    'Houston Astros':        2.53,
    'San Diego Padres':      1.22,
    'San Francisco Giants':  1.29,
    'Los Angeles Dodgers':   0.98,
    'Milwaukee Brewers':     3.43,
    'St. Louis Cardinals':   0.68,
    'Miami Marlins':         0.53,
    'Cincinnati Reds':       1.25,
    'Atlanta Braves':        0.58,
    'Chicago Cubs':          0.27,
}

STADIUMS = {
    'Arizona Diamondbacks': {'lat':33.4455,'lon':-112.0667,'cf':180,'indoor':True},
    'Atlanta Braves':       {'lat':33.8908,'lon': -84.4678,'cf':320,'indoor':False},
    'Baltimore Orioles':    {'lat':39.2839,'lon': -76.6218,'cf':220,'indoor':False},
    'Boston Red Sox':       {'lat':42.3467,'lon': -71.0972,'cf':100,'indoor':False},
    'Chicago Cubs':         {'lat':41.9484,'lon': -87.6553,'cf':350,'indoor':False},
    'Chicago White Sox':    {'lat':41.8299,'lon': -87.6338,'cf':  5,'indoor':False},
    'Cincinnati Reds':      {'lat':39.0975,'lon': -84.5072,'cf':345,'indoor':False},
    'Cleveland Guardians':  {'lat':41.4962,'lon': -81.6852,'cf':355,'indoor':False},
    'Colorado Rockies':     {'lat':39.7560,'lon':-104.9942,'cf':345,'indoor':False},
    'Detroit Tigers':       {'lat':42.3390,'lon': -83.0485,'cf':335,'indoor':False},
    'Houston Astros':       {'lat':29.7572,'lon': -95.3558,'cf':350,'indoor':True},
    'Kansas City Royals':   {'lat':39.0517,'lon': -94.4803,'cf':350,'indoor':False},
    'Los Angeles Angels':   {'lat':33.8003,'lon':-117.8827,'cf':350,'indoor':False},
    'Los Angeles Dodgers':  {'lat':34.0739,'lon':-118.2400,'cf':355,'indoor':False},
    'Miami Marlins':        {'lat':25.7781,'lon': -80.2196,'cf':350,'indoor':True},
    'Milwaukee Brewers':    {'lat':43.0280,'lon': -87.9712,'cf':355,'indoor':True},
    'Minnesota Twins':      {'lat':44.9817,'lon': -93.2775,'cf':335,'indoor':False},
    'New York Mets':        {'lat':40.7571,'lon': -73.8458,'cf':345,'indoor':False},
    'New York Yankees':     {'lat':40.8296,'lon': -73.9262,'cf':345,'indoor':False},
    'Oakland Athletics':    {'lat':37.7516,'lon':-122.2005,'cf':350,'indoor':False},
    'Athletics':            {'lat':38.5905,'lon':-121.4997,'cf':345,'indoor':False},
    'Philadelphia Phillies':{'lat':39.9061,'lon': -75.1665,'cf':330,'indoor':False},
    'Pittsburgh Pirates':   {'lat':40.4469,'lon': -80.0058,'cf':345,'indoor':False},
    'San Diego Padres':     {'lat':32.7076,'lon':-117.1570,'cf':350,'indoor':False},
    'San Francisco Giants': {'lat':37.7786,'lon':-122.3893,'cf':340,'indoor':False},
    'Seattle Mariners':     {'lat':47.5914,'lon':-122.3323,'cf':355,'indoor':True},
    'St. Louis Cardinals':  {'lat':38.6226,'lon': -90.1928,'cf':350,'indoor':False},
    'Tampa Bay Rays':       {'lat':27.7683,'lon': -82.6534,'cf':330,'indoor':True},
    'Texas Rangers':        {'lat':32.7474,'lon': -97.0832,'cf':350,'indoor':True},
    'Toronto Blue Jays':    {'lat':43.6414,'lon': -79.3894,'cf':340,'indoor':True},
    'Washington Nationals': {'lat':38.8730,'lon': -77.0074,'cf':350,'indoor':False},
}

BAT_SPEED_MEAN = 71.0
BLAST_MEAN_ACT = 0.072


# ── disk cache helpers ─────────────────────────────────────────────────────────

def load_cache(key):
    """Return unpickled data for key, or None if not cached."""
    import pickle
    path = os.path.join(CACHE_DIR, key + '.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def save_cache(key, data):
    """Pickle data under key in CACHE_DIR."""
    import pickle
    path = os.path.join(CACHE_DIR, key + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  Cached → {os.path.basename(path)}')


# ── park factor helpers ────────────────────────────────────────────────────────

def _blend_pf(pf25, pf26, w25, w26):
    if pf25 is None and pf26 is None:
        return 1.0
    if pf26 is None or w26 <= 0:
        return pf25 or 1.0
    if pf25 is None:
        return pf26
    total = w25 + w26
    return (w25 * pf25 + w26 * pf26) / total


def build_dn_park_table(pf_blend):
    w25 = pf_blend.get(2025, 1.0)
    w26 = pf_blend.get(2026, 0.0)
    table = {}
    all_teams = set(PF_DAY_2025) | set(PF_NIGHT_2025)
    all_teams.add('Tampa Bay Rays')
    for team in all_teams:
        if 'Tampa Bay' in team:
            table[(team, True)]  = TAMPA_PF_DAY
            table[(team, False)] = TAMPA_PF_NIGHT
            continue
        table[(team, True)]  = _blend_pf(
            PF_DAY_2025.get(team), PF_DAY_2026.get(team), w25, w26)
        table[(team, False)] = _blend_pf(
            PF_NIGHT_2025.get(team), PF_NIGHT_2026.get(team), w25, w26)
    return table


# ── MLB API helpers ────────────────────────────────────────────────────────────

def get_games(date_str):
    r = session.get('https://statsapi.mlb.com/api/v1/schedule',
        params={'sportId':1,'date':date_str,'hydrate':'probablePitcher,lineups,team'},
        timeout=10)
    r.raise_for_status()
    games = []
    for de in r.json().get('dates', []):
        for g in de.get('games', []):
            home = g['teams']['home']; away = g['teams']['away']
            lin  = g.get('lineups', {})
            games.append({
                'game_pk':           g['gamePk'],
                'game_date':         g.get('gameDate', ''),
                'home_team':         home['team']['name'],
                'away_team':         away['team']['name'],
                'home_team_id':      home['team']['id'],
                'away_team_id':      away['team']['id'],
                'home_pitcher_id':   home.get('probablePitcher',{}).get('id'),
                'home_pitcher_name': home.get('probablePitcher',{}).get('fullName','TBD'),
                'away_pitcher_id':   away.get('probablePitcher',{}).get('id'),
                'away_pitcher_name': away.get('probablePitcher',{}).get('fullName','TBD'),
                'home_lineup_ids':   [p['id'] for p in lin.get('homePlayers',[])],
                'away_lineup_ids':   [p['id'] for p in lin.get('awayPlayers',[])],
            })
    return games


def is_day_game(game_date_str):
    try:
        dt = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
        return dt.hour < DAY_GAME_CUTOFF
    except Exception:
        return False


def get_roster(team_id):
    r = session.get(f'https://statsapi.mlb.com/api/v1/teams/{team_id}/roster',
        params={'rosterType':'active'}, timeout=10)
    r.raise_for_status()
    return [p['person']['id'] for p in r.json().get('roster', [])]


def parse_ip(ip_str):
    try:
        parts = str(ip_str).split('.')
        return int(parts[0]) + int(parts[1] if len(parts) > 1 else 0) / 3
    except:
        return 0.0


def fetch_stats_bulk(player_ids, group, season, chunk_size=50):
    result = {}
    ids_list = list(player_ids)
    for i in range(0, len(ids_list), chunk_size):
        chunk = ids_list[i:i+chunk_size]
        try:
            r = session.get('https://statsapi.mlb.com/api/v1/people',
                params={'personIds':','.join(str(x) for x in chunk),
                        'hydrate':f'stats(group=[{group}],type=[season],season={season})'},
                timeout=15)
            r.raise_for_status()
            for person in r.json().get('people', []):
                pid = person['id']
                stat_obj = None
                for se in person.get('stats', []):
                    if se.get('group',{}).get('displayName','').lower() == group:
                        splits = se.get('splits', [])
                        if splits:
                            stat_obj = splits[0].get('stat', {})
                        break
                result[pid] = {
                    'name':       person.get('fullName',''),
                    'bat_side':   person.get('batSide', {}).get('code','R'),
                    'pitch_hand': person.get('pitchHand',{}).get('code','R'),
                    'stat':       stat_obj or {},
                }
        except Exception as e:
            print(f'  Stats chunk error: {e}')
        time.sleep(0.15)
    return result


def fetch_raw_hitting(player_ids, season):
    bulk = fetch_stats_bulk(player_ids, 'hitting', season)
    result = {}
    for pid, data in bulk.items():
        st = data.get('stat', {})
        if not st:
            continue
        pa = st.get('plateAppearances', 0) or st.get('atBats', 0)
        hr = st.get('homeRuns', 0)
        if pa >= MIN_PA:
            result[pid] = {
                'name':       data['name'],
                'bat_side':   data['bat_side'],
                'pitch_hand': data['pitch_hand'],
                'pa': pa, 'hr': hr,
            }
    return result


def fetch_hitting_splits(player_ids, season, chunk_size=50):
    result = {}
    ids_list = list(player_ids)
    for i in range(0, len(ids_list), chunk_size):
        chunk = ids_list[i:i+chunk_size]
        try:
            r = session.get('https://statsapi.mlb.com/api/v1/people',
                params={'personIds': ','.join(str(x) for x in chunk),
                        'hydrate': f'stats(group=[hitting],type=[statSplits],season={season})'},
                timeout=15)
            r.raise_for_status()
            for person in r.json().get('people', []):
                pid = person['id']
                entry = {'vs_L': {}, 'vs_R': {}, 'home': {}, 'away': {}}
                for se in person.get('stats', []):
                    if se.get('type', {}).get('displayName', '').lower() != 'statsplits':
                        continue
                    for split in se.get('splits', []):
                        code = split.get('split', {}).get('code', '')
                        st   = split.get('stat', {})
                        pa   = st.get('plateAppearances', 0) or st.get('atBats', 0)
                        hr   = st.get('homeRuns', 0)
                        if code == 'vl':
                            entry['vs_L'] = {'pa': pa, 'hr': hr}
                        elif code == 'vr':
                            entry['vs_R'] = {'pa': pa, 'hr': hr}
                        elif code == 'h':
                            entry['home'] = {'pa': pa, 'hr': hr}
                        elif code in ('a', 'r'):
                            entry['away'] = {'pa': pa, 'hr': hr}
                result[pid] = entry
        except Exception as e:
            print(f'  Splits chunk error: {e}')
        time.sleep(0.15)
    return result


def combine_hitting(raw_by_season, weights):
    active_years = {yr: w for yr, w in weights.items() if w > 0 and yr in raw_by_season}
    all_pids = set()
    for yr in active_years:
        all_pids |= set(raw_by_season[yr])

    result = {}
    for pid in all_pids:
        meta  = None
        w_pa  = 0.0
        w_hr  = 0.0
        for yr, w in active_years.items():
            row = raw_by_season[yr].get(pid)
            if row is None:
                continue
            if meta is None:
                meta = row
            w_pa += row['pa'] * w
            w_hr += row['hr'] * w
        if meta is None or w_pa < MIN_PA:
            continue
        result[pid] = {
            'name':       meta['name'],
            'bat_side':   meta['bat_side'],
            'pitch_hand': meta['pitch_hand'],
            'stat': {'homeRuns': w_hr, 'plateAppearances': w_pa},
        }
    return result


def fetch_statcast_power(year=2025):
    try:
        r = session.get(
            f'https://baseballsavant.mlb.com/leaderboard/statcast?year={year}'
            f'&position=batter&team=&csv=true&min_pa=50', timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        result = {}
        for _, row in df.iterrows():
            pid = int(row['player_id'])
            result[pid] = {
                'brl_pa':    float(row.get('brl_pa',              0) or 0) / 100,
                'exit_velo': float(row.get('avg_exit_velocity',   0) or 0),
                'hard_hit':  float(row.get('hard_hit_percent',    0) or 0) / 100,
            }
        return result
    except Exception as e:
        print(f'  Power data fetch failed: {e}')
        return {}


def fetch_bat_tracking(year=2025):
    global BAT_SPEED_MEAN, BLAST_MEAN_ACT
    try:
        r = session.get(
            f'https://baseballsavant.mlb.com/leaderboard/bat-tracking?year={year}&csv=true',
            timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        result = {}
        for _, row in df.iterrows():
            result[int(row['id'])] = {
                'bat_speed':       float(row.get('avg_bat_speed', 0) or 0),
                'blast_per_swing': float(row.get('blast_per_swing', 0) or 0),
            }
        speeds = [v['bat_speed'] for v in result.values() if v['bat_speed'] > 0]
        blasts = [v['blast_per_swing'] for v in result.values() if v['blast_per_swing'] > 0]
        if speeds: BAT_SPEED_MEAN = sum(speeds) / len(speeds)
        if blasts: BLAST_MEAN_ACT = sum(blasts) / len(blasts)
        print(f'  Bat tracking: {len(result)} players  speed={BAT_SPEED_MEAN:.1f} mph  blast={BLAST_MEAN_ACT:.3f}')
        return result
    except Exception as e:
        print(f'  Bat tracking failed: {e}')
        return {}


def fetch_pitch_arsenal(entity_type='batter', year=2025):
    try:
        r = session.get(
            f'https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats'
            f'?type={entity_type}&pitchType=&year={year}&team=&csv=true', timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        result = {}
        for _, row in df.iterrows():
            pid   = int(row['player_id'])
            ptype = str(row.get('pitch_type','')).strip().upper()
            if pid not in result: result[pid] = {}
            result[pid][ptype] = {
                'usage': float(row.get('pitch_usage',0) or 0) / 100,
                'rv100': float(row.get('run_value_per_100',0) or 0),
            }
        return result
    except Exception as e:
        print(f'  Pitch arsenal ({entity_type}) failed: {e}')
        return {}


def fetch_weather(games, date_str):
    results = {}
    coord_cache = {}
    for g in games:
        stadium = STADIUMS.get(g['home_team'], {})
        if stadium.get('indoor', True):
            continue
        lat, lon = stadium['lat'], stadium['lon']
        key = (lat, lon)
        if key not in coord_cache:
            try:
                r = session.get('https://archive-api.open-meteo.com/v1/archive',
                    params={'latitude':lat,'longitude':lon,
                            'hourly':'temperature_2m,wind_speed_10m,wind_direction_10m',
                            'temperature_unit':'fahrenheit','wind_speed_unit':'mph',
                            'timezone':'UTC','start_date':date_str,'end_date':date_str},
                    timeout=10)
                r.raise_for_status()
                h = r.json().get('hourly', {})
                coord_cache[key] = {
                    'temps':h.get('temperature_2m',[]),
                    'winds':h.get('wind_speed_10m',[]),
                    'dirs': h.get('wind_direction_10m',[]),
                }
                time.sleep(0.12)
            except Exception as e:
                print(f'  Weather {g["home_team"]}: {e}')
                coord_cache[key] = {}
        hourly = coord_cache.get(key, {})
        if hourly:
            try:
                dt  = datetime.fromisoformat(g['game_date'].replace('Z', '+00:00'))
                idx = min(dt.hour, 23)
            except Exception:
                idx = 23
            results[g['game_pk']] = {
                'temp_f':  hourly['temps'][idx] if hourly.get('temps') else 70,
                'wind_mph':hourly['winds'][idx] if hourly.get('winds') else 5,
                'wind_dir':hourly['dirs'][idx]  if hourly.get('dirs')  else 0,
            }
    return results


def fetch_spring_stats():
    if os.path.exists(SPRING_CACHE):
        with open(SPRING_CACHE, encoding='utf-8') as f:
            data = json.load(f)
        print(f'  Spring cache: {len(data)} players')
        return {int(k): v for k, v in data.items()}

    print('  Fetching spring training schedule...')
    r = session.get('https://statsapi.mlb.com/api/v1/schedule/games/',
        params={'sportId':1,'gameType':'S','startDate':SPRING_START,'endDate':SPRING_END},
        timeout=10)
    r.raise_for_status()
    game_pks = [g['gamePk'] for de in r.json().get('dates',[]) for g in de.get('games',[])]
    print(f'  {len(game_pks)} spring games...')

    stats = {}
    def fetch_box(pk):
        try:
            resp = session.get(f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live', timeout=15)
            resp.raise_for_status()
            teams = resp.json().get('liveData',{}).get('boxscore',{}).get('teams',{})
            rows = {}
            for side in ('home','away'):
                for pd_data in teams.get(side,{}).get('players',{}).values():
                    bat = pd_data.get('stats',{}).get('batting',{})
                    pa  = bat.get('plateAppearances',0) or bat.get('atBats',0)
                    if not pa: continue
                    pid = pd_data['person']['id']
                    rows[pid] = {'name':pd_data['person']['fullName'],
                                 'pa':pa,'hr':bat.get('homeRuns',0)}
            return rows
        except: return {}

    with ThreadPoolExecutor(max_workers=10) as ex:
        for res in as_completed({ex.submit(fetch_box, pk): pk for pk in game_pks}):
            for pid, row in res.result().items():
                if pid in stats:
                    stats[pid]['pa'] += row['pa']; stats[pid]['hr'] += row['hr']
                else:
                    stats[pid] = row

    os.makedirs(os.path.dirname(SPRING_CACHE), exist_ok=True)
    with open(SPRING_CACHE, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in stats.items()}, f)
    print(f'  Cached spring: {len(stats)} players')
    return stats


def fetch_recent_hitting(player_ids, date_str, days_back=30):
    from datetime import timedelta
    dt    = datetime.strptime(date_str, '%Y-%m-%d')
    end   = (dt - timedelta(days=1)).strftime('%Y-%m-%d')
    start = (dt - timedelta(days=days_back)).strftime('%Y-%m-%d')
    result = {}
    ids_list = list(player_ids)
    for i in range(0, len(ids_list), 50):
        chunk = ids_list[i:i+50]
        try:
            r = session.get('https://statsapi.mlb.com/api/v1/people',
                params={'personIds': ','.join(str(x) for x in chunk),
                        'hydrate': f'stats(group=[hitting],type=[dateRange],startDate={start},endDate={end},gameType=[S,R])'},
                timeout=15)
            r.raise_for_status()
            for person in r.json().get('people', []):
                pid = person['id']
                for se in person.get('stats', []):
                    splits = se.get('splits', [])
                    if splits:
                        st = splits[0].get('stat', {})
                        pa = st.get('plateAppearances', 0) or st.get('atBats', 0)
                        hr = st.get('homeRuns', 0)
                        if pa > 0:
                            result[pid] = {'pa': pa, 'hr': hr}
                        break
        except Exception as e:
            print(f'  Recent hitting chunk error: {e}')
        time.sleep(0.15)
    return result


def fetch_pitcher_recent(pitcher_ids, date_str, days_back=30):
    from datetime import timedelta
    dt    = datetime.strptime(date_str, '%Y-%m-%d')
    end   = (dt - timedelta(days=1)).strftime('%Y-%m-%d')
    start = (dt - timedelta(days=days_back)).strftime('%Y-%m-%d')
    result = {}
    ids_list = [p for p in pitcher_ids if p is not None]
    for i in range(0, len(ids_list), 50):
        chunk = ids_list[i:i+50]
        try:
            r = session.get('https://statsapi.mlb.com/api/v1/people',
                params={'personIds': ','.join(str(x) for x in chunk),
                        'hydrate': f'stats(group=[pitching],type=[dateRange],startDate={start},endDate={end},gameType=[S,R])'},
                timeout=15)
            r.raise_for_status()
            for person in r.json().get('people', []):
                pid = person['id']
                for se in person.get('stats', []):
                    splits = se.get('splits', [])
                    if splits:
                        st = splits[0].get('stat', {})
                        ip = parse_ip(st.get('inningsPitched', '0.0'))
                        hr = st.get('homeRuns', 0)
                        result[pid] = {'ip': ip, 'hr': hr}
                        break
        except Exception as e:
            print(f'  Pitcher recent chunk error: {e}')
        time.sleep(0.15)
    return result


def fetch_bvp_career(batter_ids, pitcher_ids):
    pit_set = {p for p in pitcher_ids if p is not None}
    result  = {}
    for bid in batter_ids:
        if bid is None:
            continue
        for season in (2023, 2024, 2025, 2026):
            try:
                r = session.get(
                    f'https://statsapi.mlb.com/api/v1/people/{bid}/stats',
                    params={'stats': 'vsPlayer', 'group': 'hitting',
                            'gameType': 'R', 'season': season},
                    timeout=15)
                r.raise_for_status()
                for se in r.json().get('stats', []):
                    for split in se.get('splits', []):
                        pitcher = split.get('pitcher') or split.get('opponent') or {}
                        opp_pid = pitcher.get('id')
                        if not opp_pid or opp_pid not in pit_set:
                            continue
                        st = split.get('stat', {})
                        pa = st.get('plateAppearances', 0) or st.get('atBats', 0)
                        hr = st.get('homeRuns', 0)
                        existing = result.setdefault(bid, {}).get(opp_pid, {'pa': 0, 'hr': 0})
                        result[bid][opp_pid] = {
                            'pa': existing['pa'] + pa,
                            'hr': existing['hr'] + hr,
                        }
            except Exception as e:
                print(f'  BvP {bid} vs {season}: {e}')
            time.sleep(0.08)
    return result


def get_actual_hrs(date_str):
    r = session.get('https://statsapi.mlb.com/api/v1/schedule/games/',
        params={'sportId':1,'date':date_str}, timeout=10)
    r.raise_for_status()
    dates = r.json().get('dates', [])
    if not dates: return {}
    actual = {}
    def fetch_box(pk):
        try:
            resp = session.get(f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live', timeout=15)
            resp.raise_for_status()
            teams = resp.json().get('liveData',{}).get('boxscore',{}).get('teams',{})
            hits = {}
            for side in ('home','away'):
                for pd_data in teams.get(side,{}).get('players',{}).values():
                    h = pd_data.get('stats',{}).get('batting',{}).get('homeRuns',0)
                    if h:
                        hits[pd_data['person']['id']] = {
                            'name':pd_data['person']['fullName'],'hrs':h}
            return hits
        except Exception as e:
            print(f'  Boxscore {pk}: {e}')
            return {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for res in as_completed({ex.submit(fetch_box, g['gamePk']): g
                                  for g in dates[0].get('games',[])}):
            actual.update(res.result())
    return actual


# ── factor helpers ─────────────────────────────────────────────────────────────

def get_recent_form_factor(bid, recent_stats, batter_meta):
    rd   = recent_stats.get(bid, {})
    r_pa = rd.get('pa', 0)
    r_hr = rd.get('hr', 0)
    if r_pa < 15:
        return 1.0
    bs          = batter_meta.get('stat', {})
    season_pa   = bs.get('plateAppearances', 0) or 1
    season_hr   = bs.get('homeRuns', 0)
    if season_pa <= 0 or season_hr == 0:
        return 1.0
    recent_rate = r_hr / r_pa
    season_rate = season_hr / season_pa
    raw_factor  = recent_rate / season_rate if season_rate > 0 else 1.0
    conf        = min(r_pa, 60) / 60
    blended     = conf * raw_factor + (1 - conf) * 1.0
    return max(min(blended, RECENT_FORM_CAP[1]), RECENT_FORM_CAP[0])


def get_pitcher_recent_factor(opp_pid, pit_stats, pitcher_recent):
    rd   = pitcher_recent.get(opp_pid, {})
    r_ip = rd.get('ip', 0)
    r_hr = rd.get('hr', 0)
    if r_ip < 10:
        return 1.0
    p = pit_stats.get(opp_pid, {})
    if not p or not p.get('stat'):
        return 1.0
    ps    = p['stat']
    s_ip  = parse_ip(ps.get('inningsPitched', '0.0'))
    s_hr  = ps.get('homeRuns', 0)
    if s_ip < MIN_IP or s_hr == 0:
        return 1.0
    season_hr9 = s_hr / s_ip * 9
    recent_hr9 = r_hr / r_ip * 9
    raw_factor = recent_hr9 / season_hr9 if season_hr9 > 0 else 1.0
    conf       = min(r_ip, 30) / 30
    blended    = conf * raw_factor + (1 - conf) * 1.0
    return max(min(blended, 1.60), 0.50)


def get_bvp_factor(bid, opp_pid, bvp_career, batter_meta):
    matchup = bvp_career.get(bid, {}).get(opp_pid, {})
    m_pa    = matchup.get('pa', 0)
    m_hr    = matchup.get('hr', 0)
    if m_pa < MIN_BVP_PA:
        return 1.0
    bs          = batter_meta.get('stat', {})
    overall_pa  = bs.get('plateAppearances', 0) or 1
    overall_hr  = bs.get('homeRuns', 0)
    if overall_pa <= 0 or overall_hr == 0:
        return 1.0
    matchup_rate = m_hr / m_pa
    overall_rate = overall_hr / overall_pa
    raw_factor   = matchup_rate / overall_rate if overall_rate > 0 else 1.0
    conf         = min(m_pa, 50) / 50
    blended      = conf * raw_factor + (1 - conf) * 1.0
    return max(min(blended, BVP_CAP[1]), BVP_CAP[0])


def get_home_away_factor(bid, is_home, splits, batter_meta):
    side = 'home' if is_home else 'away'
    sd   = splits.get(bid, {}).get(side, {})
    s_pa = sd.get('pa', 0)
    s_hr = sd.get('hr', 0)
    if s_pa < MIN_HA_PA:
        return 1.0
    bs           = batter_meta.get('stat', {})
    overall_pa   = bs.get('plateAppearances', 0) or 1
    overall_hr   = bs.get('homeRuns', 0)
    if overall_pa <= 0 or overall_hr == 0:
        return 1.0
    split_rate   = s_hr / s_pa
    overall_rate = overall_hr / overall_pa
    raw_factor   = split_rate / overall_rate if overall_rate > 0 else 1.0
    conf         = min(s_pa, 200) / 200
    blended      = conf * raw_factor + (1 - conf) * 1.0
    return max(min(blended, HOME_AWAY_CAP[1]), HOME_AWAY_CAP[0])


def get_bat_tracking_factor(batter_id, bat_tracking):
    bt    = bat_tracking.get(batter_id, {})
    blast = bt.get('blast_per_swing', BLAST_MEAN_ACT)
    speed = bt.get('bat_speed', BAT_SPEED_MEAN)
    bf = (max(blast, 0.001) / max(BLAST_MEAN_ACT, 0.001)) ** 0.20
    sf = (max(speed, 40) / BAT_SPEED_MEAN) ** 0.25
    return max(min((bf * sf) ** 0.5, BAT_TRACKING_CAP[1]), BAT_TRACKING_CAP[0])


def calc_weather_factor(home_team, game_pk, weather):
    if game_pk not in weather: return 1.0
    w   = weather[game_pk]
    cf  = STADIUMS.get(home_team, {}).get('cf', 350)
    tf  = 1.0 + 0.007 * (w['temp_f'] - 70)
    opp = (cf + 180) % 360
    ang = abs(((w['wind_dir'] - opp) % 360) - 180)
    out = w['wind_mph'] * math.cos(math.radians(ang))
    wf  = 1.0 + out * 0.007
    return max(min(tf * wf, WEATHER_CAP[1]), WEATHER_CAP[0])


def get_platoon_factor(bid, ph, splits, batter_meta):
    side = 'vs_L' if ph == 'L' else 'vs_R'
    sd   = splits.get(bid, {}).get(side, {})
    s_pa = sd.get('pa', 0)
    s_hr = sd.get('hr', 0)

    bat_side = batter_meta.get('bat_side', 'R')
    fallback = PLATOON.get((bat_side, ph), 1.0)

    if s_pa < MIN_SPLIT_PA:
        return fallback

    bs          = batter_meta.get('stat', {})
    overall_pa  = bs.get('plateAppearances', 0) or 1
    overall_hr  = bs.get('homeRuns', 0)
    if overall_pa <= 0 or overall_hr <= 0:
        return fallback

    split_rate   = s_hr / s_pa
    overall_rate = overall_hr / overall_pa
    raw_factor   = split_rate / overall_rate if overall_rate > 0 else 1.0
    conf    = min(s_pa, 200) / 200
    blended = conf * raw_factor + (1 - conf) * fallback
    return max(min(blended, 1.75), 0.40)


def get_spring_factor(batter_id, spring_stats):
    st = spring_stats.get(batter_id, {})
    pa = st.get('pa', 0); hrs = st.get('hr', 0)
    if pa < 20: return 1.0
    conf = min(pa, 100) / 100
    adj  = conf * (hrs / pa) + (1 - conf) * LEAGUE_HR_PA
    return max(min(1.0 + 0.15 * (adj / LEAGUE_HR_PA - 1.0), SPRING_CAP[1]), SPRING_CAP[0])


def get_weather_hist_factor(bid, game_pk, home_team, weather, weather_splits, batter_meta):
    """
    Compare this batter's historical HR rate in the current game's temperature
    and wind bucket against their overall rate.  Returns 1.0 for indoor games
    or any player/bucket with insufficient history (< 30 PA).
    """
    if STADIUMS.get(home_team, {}).get('indoor', True):
        return 1.0
    ws = weather_splits.get(bid)
    if not ws:
        return 1.0
    wx = weather.get(game_pk)
    if not wx:
        return 1.0
    bs = batter_meta.get('stat', {})
    overall_pa = bs.get('plateAppearances', 0) or 1
    overall_hr = bs.get('homeRuns', 0)
    if overall_pa <= 0 or overall_hr == 0:
        return 1.0
    overall_rate = overall_hr / overall_pa

    temp = wx['temp_f']
    if temp < 55:
        tbucket = 'temp_cold'
    elif temp > 75:
        tbucket = 'temp_hot'
    else:
        tbucket = 'temp_mild'

    cf_dir = STADIUMS.get(home_team, {}).get('cf', 350)
    opp = (cf_dir + 180) % 360
    ang = abs(((wx['wind_dir'] - opp) % 360) - 180)
    eff_wind = wx['wind_mph'] * math.cos(math.radians(ang))
    if eff_wind > 8:
        wbucket = 'wind_out'
    elif eff_wind < -8:
        wbucket = 'wind_in'
    else:
        wbucket = 'wind_calm'

    def _bucket_factor(bucket):
        bd = ws.get(bucket, {})
        b_pa = bd.get('pa', 0)
        b_hr = bd.get('hr', 0)
        if b_pa < 30 or overall_rate <= 0:
            return 1.0
        b_rate = b_hr / b_pa
        conf   = min(b_pa, 200) / 200
        raw    = b_rate / overall_rate
        return max(min(conf * raw + (1 - conf), WEATHER_HIST_CAP[1]), WEATHER_HIST_CAP[0])

    return max(min(_bucket_factor(tbucket) * _bucket_factor(wbucket),
                   WEATHER_HIST_CAP[1]), WEATHER_HIST_CAP[0])


# ── scorer ─────────────────────────────────────────────────────────────────────

def score_batter(bid, opp_pid, home_team, game_pk, is_day, is_home,
                 batter_stats, pitcher_stats, barrel_data, dn_park,
                 bat_tracking, bvp, pars, splits,
                 recent_bat, pitcher_recent, bvp_career,
                 weather, spring_stats, weather_splits, cfg):
    b = batter_stats.get(bid, {})
    p = pitcher_stats.get(opp_pid, {})
    if not b or not b.get('stat'): return None
    bs = b['stat']
    pa = bs.get('plateAppearances', 0) or bs.get('atBats', 0)
    hr = bs.get('homeRuns', 0)
    if not pa or pa < MIN_PA: return None

    conf  = min(pa, 300) / 300
    rate  = conf * (hr / pa) + (1 - conf) * LEAGUE_HR_PA

    ph = 'R'; vuln = 1.0
    if p and p.get('stat'):
        ps = p['stat']
        ip = parse_ip(ps.get('inningsPitched', '0.0'))
        ph = p.get('pitch_hand', 'R')
        if ip >= MIN_IP:
            vuln = min((ps.get('homeRuns',0) / ip * 9) / LEAGUE_HR9, cfg['vuln_cap'])

    park     = dn_park.get((home_team, is_day), 1.0)
    platoon  = get_platoon_factor(bid, ph, splits, b)

    d   = barrel_data.get(bid, {})
    brl = d.get('brl_pa',    LEAGUE_BRL_PA)    if isinstance(d, dict) else LEAGUE_BRL_PA
    ev  = d.get('exit_velo', LEAGUE_EXIT_VELO) if isinstance(d, dict) else LEAGUE_EXIT_VELO
    hh  = d.get('hard_hit',  LEAGUE_HARD_HIT)  if isinstance(d, dict) else LEAGUE_HARD_HIT
    brl_f = (max(brl, 0.001) / LEAGUE_BRL_PA)   ** cfg['pwr_brl_exp']
    ev_f  = (max(ev,  60)    / LEAGUE_EXIT_VELO) ** cfg['pwr_ev_exp']
    hh_f  = (max(hh,  0.001) / LEAGUE_HARD_HIT)  ** 0.15
    pwr_f = max(min(brl_f * ev_f * hh_f, POWER_CAP[1]), POWER_CAP[0])

    bt_f  = get_bat_tracking_factor(bid, bat_tracking)

    bp = bvp.get(bid, {}); pp = pars.get(opp_pid, {})
    if bp and pp:
        wrv = 0.0; wu = 0.0
        for p_code, p_data in pp.items():
            p_grp = PITCH_GROUPS.get(p_code, p_code)
            usage = p_data.get('usage', 0)
            b_rv  = bp.get(p_code, {}).get('rv100')
            if b_rv is None:
                for b_code, b_data in bp.items():
                    if PITCH_GROUPS.get(b_code, b_code) == p_grp:
                        b_rv = b_data.get('rv100', 0); break
            if b_rv is not None:
                wrv += usage * b_rv; wu += usage
        pm_f = (max(min(math.exp((wrv / wu) * cfg['matchup_scale']),
                        PITCH_MATCHUP_CAP[1]), PITCH_MATCHUP_CAP[0])
                if wu >= 0.40 else 1.0)
    else:
        pm_f = 1.0

    wx_f  = calc_weather_factor(home_team, game_pk, weather)
    st_f  = get_spring_factor(bid, spring_stats)

    rd   = recent_bat.get(bid, {})
    r_pa = rd.get('pa', 0); r_hr = rd.get('hr', 0)
    if r_pa >= cfg['form_min_pa'] and bs.get('homeRuns', 0) > 0:
        recent_rate = r_hr / r_pa
        season_rate = hr / pa
        raw_f  = recent_rate / season_rate if season_rate > 0 else 1.0
        conf_f = min(r_pa, 60) / 60
        form_f = max(min(conf_f * raw_f + (1 - conf_f) * 1.0,
                         cfg['form_cap_hi']), RECENT_FORM_CAP[0])
    else:
        form_f = 1.0

    pit_rf = get_pitcher_recent_factor(opp_pid, pitcher_stats, pitcher_recent)
    ha_f   = get_home_away_factor(bid, is_home, splits, b)
    wx_hist_f = get_weather_hist_factor(bid, game_pk, home_team, weather, weather_splits, b)

    raw_bvp = get_bvp_factor(bid, opp_pid, bvp_career, b)
    bvp_f   = 1.0 + cfg['bvp_weight'] * (raw_bvp - 1.0)

    exp_hr  = (rate * vuln * park * platoon * pwr_f * bt_f * pm_f
               * wx_f * st_f * form_f * pit_rf * bvp_f * ha_f * wx_hist_f
               * AVG_PA_GAME * cfg['calibration'])
    hr_prob = 1 - math.exp(-exp_hr)

    return {
        'batter_id': bid, 'name': b['name'], 'bat_side': b.get('bat_side', 'R'),
        'hr_prob': round(hr_prob, 4),
    }


# ── main ───────────────────────────────────────────────────────────────────────

TEST_DATES = [
    '2026-03-27','2026-03-28','2026-03-29','2026-03-30',
    '2026-03-31','2026-04-01','2026-04-02','2026-04-03',
    '2026-04-04','2026-04-05','2026-04-06','2026-04-07',
    '2026-04-08','2026-04-09','2026-04-10','2026-04-11',
    '2026-04-12','2026-04-13','2026-04-14','2026-04-15',
]

# ── 1. Fetch season-level shared data (once) ──────────────────────────────────
print('='*68)
print('  LOADING SEASON-LEVEL DATA')
print('='*68)

print('  Power data (barrel/exit velo/hard-hit)...')
barrel = load_cache(f'barrel_{PITCHER_SEASON}')
if barrel is None:
    barrel = fetch_statcast_power(PITCHER_SEASON)
    save_cache(f'barrel_{PITCHER_SEASON}', barrel)
print(f'  {len(barrel)} players')

print('  Bat tracking...')
_bat_cache = load_cache(f'bat_tracking_{PITCHER_SEASON}')
if _bat_cache is None:
    bat_t = fetch_bat_tracking(PITCHER_SEASON)
    save_cache(f'bat_tracking_{PITCHER_SEASON}', {'bat_t': bat_t, 'speed_mean': BAT_SPEED_MEAN, 'blast_mean': BLAST_MEAN_ACT})
else:
    bat_t          = _bat_cache['bat_t']
    BAT_SPEED_MEAN = _bat_cache['speed_mean']
    BLAST_MEAN_ACT = _bat_cache['blast_mean']
    print(f'  Bat tracking: {len(bat_t)} players (from cache)')

print('  Batter arsenal...')
bvp = load_cache(f'arsenal_batter_{PITCHER_SEASON}')
if bvp is None:
    bvp = fetch_pitch_arsenal('batter', PITCHER_SEASON)
    save_cache(f'arsenal_batter_{PITCHER_SEASON}', bvp)
print(f'  {len(bvp)} batters')

print('  Pitcher arsenal...')
pars = load_cache(f'arsenal_pitcher_{PITCHER_SEASON}')
if pars is None:
    pars = fetch_pitch_arsenal('pitcher', PITCHER_SEASON)
    save_cache(f'arsenal_pitcher_{PITCHER_SEASON}', pars)
print(f'  {len(pars)} pitchers')

print('  Spring training stats...')
spring = fetch_spring_stats()

print('  Park factors: using hardcoded 2025/2026 day/night tables')

print('  Career BvP lookup...')
bvp_career_global = load_cache('bvp_career')
if bvp_career_global is None:
    bvp_career_global = {}
    print('  No bvp_career.pkl found — BvP disabled')
else:
    n_pairs = sum(len(v) for v in bvp_career_global.values())
    print(f'  {len(bvp_career_global)} batters, {n_pairs:,} pairs')

print('  Weather performance splits...')
weather_splits_global = load_cache('weather_splits') or {}
print(f'  {len(weather_splits_global)} batters with weather split history')

shared_data = {
    'barrel': barrel, 'bat_tracking': bat_t,
    'bvp': bvp, 'pars': pars, 'spring': spring,
}

# ── 2. Fetch per-date data (once per date) ────────────────────────────────────
print()
print('='*68)
print('  FETCHING PER-DATE DATA')
print('='*68)

date_cache = {}
for date_str in TEST_DATES:
    cache_key = f'date_{date_str}'
    cached = load_cache(cache_key)
    if cached is not None:
        print(f'\n  --- {date_str} --- (cached)')
        date_cache[date_str] = cached
        continue

    print(f'\n  --- {date_str} ---')
    games = get_games(date_str)
    day_c = sum(1 for g in games if is_day_game(g['game_date']))
    print(f'  {len(games)} games ({day_c} day / {len(games)-day_c} night)')

    batter_map = {}
    for g in games:
        pk = g['game_pk']
        if not g['home_lineup_ids']:
            continue                       # skip — no starting lineup available
        hi, ai = g['home_lineup_ids'], g['away_lineup_ids']
        batter_map[pk] = {
            'home': hi, 'away': ai,
            'home_team': g['home_team'], 'away_team': g['away_team'],
            'is_day': is_day_game(g['game_date']),
        }
    games = [g for g in games if g['game_pk'] in batter_map]   # keep only games with lineups

    all_bat = {bid for v in batter_map.values() for s in ('home','away') for bid in v[s]}
    all_pit = {g[k] for g in games for k in ('home_pitcher_id','away_pitcher_id') if g[k]}
    print(f'  {len(games)} games with starting lineups  |  Batters: {len(all_bat)}  Pitchers: {len(all_pit)}')

    print('  Hitting 2025...')
    raw_25 = fetch_raw_hitting(all_bat, 2025)
    print(f'  {len(raw_25)} with ≥{MIN_PA} PA')
    print('  Splits 2025 (vs L/R)...')
    splits_25 = fetch_hitting_splits(all_bat, 2025)
    print(f'  {len(splits_25)} batters with split data')

    print('  Hitting 2026...')
    raw_26 = fetch_raw_hitting(all_bat, 2026)
    print(f'  {len(raw_26)} with ≥{MIN_PA} PA in 2026')

    print('  Pitcher stats...')
    pit_stats = fetch_stats_bulk(all_pit, 'pitching', PITCHER_SEASON)
    print(f'  {sum(1 for v in pit_stats.values() if v["stat"])} with data')

    recent_bat     = {}
    pitcher_recent = {}

    # Use global BvP cache (subset for today's batters)
    bvp_career_day = {bid: bvp_career_global.get(bid, {}) for bid in all_bat}
    # Use global weather splits (subset for today's batters)
    ws_day = {bid: weather_splits_global[bid] for bid in all_bat if bid in weather_splits_global}

    print('  Weather...')
    weather = fetch_weather(games, date_str)
    outdoor = sum(1 for g in games if not STADIUMS.get(g['home_team'],{}).get('indoor',True))
    print(f'  {len(weather)}/{outdoor} outdoor')

    print('  Actual HRs...')
    actual = get_actual_hrs(date_str)
    print(f'  {len(actual)} HR hitters')

    date_cache[date_str] = {
        'games': games, 'batter_map': batter_map,
        'raw_25': raw_25, 'raw_26': raw_26,
        'splits_25': splits_25,
        'pit_stats': pit_stats,
        'recent_bat': recent_bat, 'pitcher_recent': pitcher_recent,
        'bvp_career': bvp_career_day, 'weather_splits': ws_day,
        'weather': weather,
        'actual': actual, 'all_bat': all_bat,
    }
    save_cache(cache_key, date_cache[date_str])
    time.sleep(0.5)

# ── 3. Grid search ────────────────────────────────────────────────────────────
print()
print('='*68)
print(f'  TUNING GRID SEARCH  ({len(ALL_TUNE_CONFIGS)} configs × {len(TEST_DATES)} dates)')
print('='*68)
print()

sw_label, sw_weights = SEASON_WEIGHT_CONFIGS[0]
pf_label, pf_blend   = PF_BLEND_CONFIGS[0]
dn_park = build_dn_park_table(pf_blend)

date_batter_stats = {}
for date_str in TEST_DATES:
    dc = date_cache[date_str]
    raw_by_season = {}
    for yr, key in [(2025, 'raw_25'), (2026, 'raw_26')]:
        if sw_weights.get(yr, 0) > 0:
            raw_by_season[yr] = dc[key]
    date_batter_stats[date_str] = combine_hitting(raw_by_season, sw_weights)

print('  Pre-computing batter-game rows...')
pregame_rows = {}
for date_str in TEST_DATES:
    dc   = date_cache[date_str]
    rows = []
    for g in dc['games']:
        pk   = g['game_pk']
        info = dc['batter_map'][pk]
        for side, opp_id, is_home in [
            ('away', g['home_pitcher_id'], False),
            ('home', g['away_pitcher_id'], True),
        ]:
            for bid in info[side]:
                rows.append((bid, opp_id, info['home_team'], pk,
                              info['is_day'], is_home,
                              info['away_team']))
    pregame_rows[date_str] = rows

grid_results = []
n_configs    = len(ALL_TUNE_CONFIGS)

for ci, cfg in enumerate(ALL_TUNE_CONFIGS, 1):
    date_acc   = []
    date_picks = {}

    for date_str in TEST_DATES:
        dc           = date_cache[date_str]
        batter_stats = date_batter_stats[date_str]

        # Score every batter and group by game
        game_rows = {}   # {game_pk: [scored rows]}
        for (bid, opp_id, home_team, pk, is_day, is_home, away_team) in pregame_rows[date_str]:
            row = score_batter(
                bid, opp_id, home_team, pk, is_day, is_home,
                batter_stats, dc['pit_stats'], barrel,
                dn_park, bat_t, bvp, pars,
                dc['splits_25'],
                dc['recent_bat'], dc['pitcher_recent'], dc['bvp_career'],
                dc['weather'], spring, dc.get('weather_splits', {}), cfg,
            )
            if row:
                row['game_pk']   = pk
                row['home_team'] = home_team
                row['away_team'] = away_team
                game_rows.setdefault(pk, []).append(row)

        # Compute P(no HR in game) = ∏(1 - hr_prob) across all scored batters
        game_preds = []
        for pk, rows in game_rows.items():
            p_no_hr = 1.0
            for r in rows:
                p_no_hr *= (1 - r['hr_prob'])
            bm = dc['batter_map'].get(pk, {})
            all_bids = set(bm.get('home', [])) | set(bm.get('away', []))
            had_hr = any(bid in dc['actual'] for bid in all_bids)
            game_preds.append({
                'game_pk':   pk,
                'home_team': rows[0]['home_team'],
                'away_team': rows[0]['away_team'],
                'p_no_hr':   round(p_no_hr, 4),
                'had_hr':    had_hr,
            })

        # Pick the top N games with highest P(no HR)
        game_preds.sort(key=lambda x: -x['p_no_hr'])
        day_picks = game_preds[:cfg['games_to_pick']]
        date_picks[date_str] = day_picks

        if not day_picks:
            date_acc.append(0.0)
            continue
        n_correct = sum(1 for p in day_picks if not p['had_hr'])
        date_acc.append(n_correct / len(day_picks))

    avg_acc = sum(date_acc) / len(date_acc)
    grid_results.append({
        'cfg': cfg, 'avg_prec': avg_acc,
        'per_date_prec': {d: round(date_acc[i], 3) for i, d in enumerate(TEST_DATES)},
        'date_picks': date_picks,
    })

    if ci % 50 == 0 or ci == n_configs:
        print(f'  {ci}/{n_configs} configs done  '
              f'(current best: {max(r["avg_prec"] for r in grid_results):.1%})')

# ── 4. Results ────────────────────────────────────────────────────────────────
grid_results.sort(key=lambda x: -x['avg_prec'])
best     = grid_results[0]
best_cfg = best['cfg']

print()
print('='*68)
print(f'  TOP 20 CONFIGS  (no-HR game prediction, {len(TEST_DATES)} dates)')
print('='*68)
print(f"  {'Rank':<5} {'Acc':>7}  Config")
print(f"  {'-'*68}")
for i, r in enumerate(grid_results[:20], 1):
    c = r['cfg']
    desc = (f"n={c['games_to_pick']} brl={c['pwr_brl_exp']} ev={c['pwr_ev_exp']} "
            f"form_cap={c['form_cap_hi']} form_pa={c['form_min_pa']} "
            f"vuln={c['vuln_cap']} mtch={c['matchup_scale']} "
            f"bvp={c['bvp_weight']}")
    print(f"  {i:<5} {r['avg_prec']:>6.1%}  {desc}")

print()
print('  BEST CONFIG:')
for k, v in best_cfg.items():
    print(f"    {k:<20} = {v}")
print(f"  Avg accuracy: {best['avg_prec']:.1%}")

print()
print('  Per-date accuracy (best config):')
print(f"  {'Date':<12} {'Picks':>6} {'Correct':>8} {'Acc':>7}  {'0-HR Games':>10}")
print(f"  {'-'*52}")
for date_str in TEST_DATES:
    picks     = best['date_picks'][date_str]
    dc        = date_cache[date_str]
    n_correct = sum(1 for p in picks if not p['had_hr'])
    acc       = n_correct / len(picks) if picks else 0.0
    n_zero    = sum(
        1 for pk, bm in dc['batter_map'].items()
        if not any(bid in dc['actual']
                   for bid in set(bm.get('home', [])) | set(bm.get('away', [])))
    )
    print(f"  {date_str:<12} {len(picks):>6} {n_correct:>8} {acc:>6.0%}  {n_zero:>10}")

print()
print(f'  No-HR game predictions — {TEST_DATES[-1]} (best config):')
last_date  = TEST_DATES[-1]
last_picks = best['date_picks'][last_date]
print(f"  {'Matchup':<34} {'P(no HR)':>9}  Correct?")
print(f"  {'-'*58}")
for p in sorted(last_picks, key=lambda x: -x['p_no_hr']):
    matchup = f"{p['away_team']} @ {p['home_team']}"
    correct = not p['had_hr']
    print(f"  {matchup:<34} {p['p_no_hr']:>8.1%}  {'YES (0 HRs)' if correct else 'NO  (HR hit)'}")