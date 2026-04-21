"""
HR Prediction Backtest Runner — Grid Search Edition
Fetches all data ONCE per date, then sweeps all weight/blend combinations
in memory (no extra API calls per config).

Special cases:
  Tampa Bay  — always uses 2024 Savant PF (Tropicana Field; 2025 was temp stadium)
  Athletics  — only 2025/2026 Savant PF; no hardcoded Coliseum fallback (new stadium 2025)
  2026 stats — only included for dates >= 2026-04-01

Usage: python hr_backtest_runner.py
"""
import math, time, io, json, os, itertools, pickle, hashlib
import numpy as np
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
CALIBRATION      = 0.95 #72
MAX_PITCHER_VULN = 1.7 #1.5
LEAGUE_BRL_PA    = 0.0615
LEAGUE_EXIT_VELO = 88.5
LEAGUE_HARD_HIT  = 0.385
MIN_BVP_PA       = 3     # minimum career PA vs a pitcher to use BvP data
MIN_HA_PA        = 20    # minimum home or away PA to use home/away split
GAME_UTC_HOUR    = 20        # ~7 PM EDT for weather index
# Game time slots in Eastern Time (UTC-4 during baseball season)
# day=12-5 PM ET (16-21 UTC), prime=6-8 PM ET (22-24/0 UTC), late=9 PM+ ET (1+ UTC)
DAY_ET_START     = 12        # noon ET
DAY_ET_END       = 17        # 5:59 PM ET
PRIME_ET_START   = 18        # 6 PM ET
PRIME_ET_END     = 20        # 8:59 PM ET
# late = 9 PM ET through midnight+ ET

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
DAY_NIGHT_BATTER_CAP = (0.60, 1.75)  # batter's personal day/night HR split
WEATHER_HIST_CAP  = (0.80, 1.25)  # historical HR rate in temp/wind bucket

PITCH_GROUPS = {
    'FF':'FB','FT':'FB','SI':'FB','FC':'FB',
    'SL':'SL','ST':'SL',
    'CU':'CB','KC':'CB','SV':'CB',
    'CH':'CH','FS':'CH','FO':'CH',
}

# ── grid search configurations ─────────────────────────────────────────────────
# Only 2025 + 2026. 2026 only applied for dates >= USE_2026_FROM.
SEASON_WEIGHT_CONFIGS = [
    ('25×1  + 26×1',    {2025:  1.00, 2026: 1.00})#OG {2025:  1.0, 2026: 1.0}),
]

PF_BLEND_CONFIGS = [
    ('PF: 85/15',      {2025: 0.85, 2026: 0.15}),
]

# ── tuning grid ────────────────────────────────────────────────────────────────
# Each key maps to a list of values to try. All combos are swept in memory
# after data is fetched — no extra API calls per combo.
TUNE_PARAM_GRID = {
    'picks_skip':     [0, 1, 2, 3, 4, 5],         # how many top picks per game to skip
    'pwr_brl_exp':    [0.10, 0.20, 0.35, 0.45, 0],# barrel rate exponent
    'pwr_ev_exp':     [0.15, 0.20, 0.30, 0.40, 0.50, 0],# exit velo exponent
    'form_cap_hi':    [1.00, 1.25, 1.50, 2.00, 2.25, 2.50, 3.00],# recent batter form cap (high)
    'form_min_pa':    [10, 20, 30],            # min recent PA to activate form signal
    'vuln_cap':       [0.5, 0.75, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20],   # pitcher HR/9 vulnerability cap
    'matchup_scale':  [0.3, 0.4, 0.5, 0.7, 0.8, 0.2, 0.9],   # pitch matchup sensitivity
    'bvp_weight':     [0.0, 0.25, 0.5, 0.75, 1.0],    # BvP signal weight (0=off, 1=full)
    'calibration':    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # global scale
}

# Pre-compute all combos as list of dicts
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

# ── Hardcoded HR park factors (Savant index / 100 → multiplicative) ────────────
# Source: Baseball Savant statcast park factors, HR column, condition=All, rolling=1yr
# Updated: 2026-04-20
# Tampa Bay 2025 = Steinbrenner Field (temp venue) → override with Tropicana estimate
TAMPA_PF_2025 = 1.00   # Tropicana estimate for 2025

# 2025 full-season HR park factors (Savant, condition=All)
PF_HR_2025 = {
    'Los Angeles Dodgers':   1.37,
    'Baltimore Orioles':     1.21,
    'Toronto Blue Jays':     1.18,
    'Philadelphia Phillies': 1.17,
    'Detroit Tigers':        1.14,
    'Athletics':             1.12,
    'Cincinnati Reds':       1.11,
    'New York Yankees':      1.11,
    'Los Angeles Angels':    1.11,
    'Chicago Cubs':          1.11,
    'Colorado Rockies':      1.10,
    'Houston Astros':        1.06,
    'Atlanta Braves':        1.05,
    'New York Mets':         1.01,
    'Seattle Mariners':      0.98,
    'Milwaukee Brewers':     0.98,
    'Chicago White Sox':     0.96,
    'Washington Nationals':  0.93,
    'San Diego Padres':      0.91,
    'Arizona Diamondbacks':  0.90,
    'Cleveland Guardians':   0.88,
    'Minnesota Twins':       0.87,
    'Miami Marlins':         0.84,
    'San Francisco Giants':  0.84,
    'Boston Red Sox':        0.84,
    'Kansas City Royals':    0.83,
    'Texas Rangers':         0.80,
    'St. Louis Cardinals':   0.77,
    'Pittsburgh Pirates':    0.66,
}

# 2026 HR park factors (Savant, condition=All, ~3 weeks of data as of 4/20)
PF_HR_2026 = {
    'New York Yankees':      2.06,
    'Athletics':             1.55,
    'Houston Astros':        1.51,
    'Washington Nationals':  1.48,
    'Cincinnati Reds':       1.43,
    'Detroit Tigers':        1.42,
    'Baltimore Orioles':     1.41,
    'Milwaukee Brewers':     1.39,
    'Los Angeles Dodgers':   1.27,
    'Toronto Blue Jays':     1.25,
    'Philadelphia Phillies': 1.22,
    'San Diego Padres':      1.14,
    'Cleveland Guardians':   0.98,
    'Chicago Cubs':          0.97,
    'Kansas City Royals':    0.94,
    'Seattle Mariners':      0.90,
    'Atlanta Braves':        0.88,
    'Chicago White Sox':     0.86,
    'Arizona Diamondbacks':  0.85,
    'New York Mets':         0.85,
    'Tampa Bay Rays':        0.80,
    'Miami Marlins':         0.77,
    'Colorado Rockies':      0.75,
    'St. Louis Cardinals':   0.71,
    'Minnesota Twins':       0.69,
    'Los Angeles Angels':    0.67,
    'Pittsburgh Pirates':    0.58,
    'Texas Rangers':         0.54,
    'San Francisco Giants':  0.49,
    'Boston Red Sox':        0.39,
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

# Module-level bat tracking means — updated after fetch
BAT_SPEED_MEAN = 71.0
BLAST_MEAN_ACT = 0.072


# ── disk cache helpers ─────────────────────────────────────────────────────────

def load_cache(key):
    """Return unpickled data for key, or None if not cached."""
    path = os.path.join(CACHE_DIR, key + '.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def save_cache(key, data):
    """Pickle data under key in CACHE_DIR."""
    path = os.path.join(CACHE_DIR, key + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  Cached → {os.path.basename(path)}')


# ── park factor helpers ────────────────────────────────────────────────────────

def _blend_pf(pf25, pf26, w25, w26):
    """Normalize-blend two park factors. If 2026 is None/missing, use 2025 only."""
    if pf25 is None and pf26 is None:
        return 1.0
    if pf26 is None or w26 <= 0:
        return pf25 or 1.0
    if pf25 is None:
        return pf26
    total = w25 + w26
    return (w25 * pf25 + w26 * pf26) / total


def build_park_table(pf_blend):
    """
    Pre-compute {team: overall_park_factor} for every team.
    Blends day (35%) and night (65%) park factors into a single venue multiplier.
    The batter's personal day/prime/late performance is handled by dn_bat_f.
    """
    DAY_WEIGHT   = 0.35   # ~35% of MLB games are day games
    NIGHT_WEIGHT = 0.65
    w25 = pf_blend.get(2025, 1.0)
    w26 = pf_blend.get(2026, 0.0)
    table = {}

    all_teams = set(PF_DAY_2025) | set(PF_NIGHT_2025)
    all_teams.add('Tampa Bay Rays')

    for team in all_teams:
        if 'Tampa Bay' in team:
            day_pf   = TAMPA_PF_DAY
            night_pf = TAMPA_PF_NIGHT
        else:
            day_pf   = _blend_pf(
                PF_DAY_2025.get(team), PF_DAY_2026.get(team), w25, w26)
            night_pf = _blend_pf(
                PF_NIGHT_2025.get(team), PF_NIGHT_2026.get(team), w25, w26)
        table[team] = DAY_WEIGHT * day_pf + NIGHT_WEIGHT * night_pf

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


def get_game_time_slot(game_date_str):
    """Return 'day', 'prime', or 'late' based on ET hour of first pitch."""
    try:
        dt = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
        et_hour = (dt.hour - 4) % 24   # UTC → Eastern (EDT during baseball season)
        if DAY_ET_START <= et_hour <= DAY_ET_END:
            return 'day'
        elif PRIME_ET_START <= et_hour <= PRIME_ET_END:
            return 'prime'
        else:
            return 'late'
    except Exception:
        return 'prime'  # default to prime time


def get_roster(team_id):
    r = session.get(f'https://statsapi.mlb.com/api/v1/teams/{team_id}/roster',
        params={'rosterType':'active'}, timeout=10)
    r.raise_for_status()
    return [p['person']['id'] for p in r.json().get('roster', [])]


def parse_ip(ip_str):
    if isinstance(ip_str, (int, float)):
        return float(ip_str)
    try:
        parts = str(ip_str).split('.')
        return int(parts[0]) + int(parts[1] if len(parts) > 1 else 0) / 3
    except:
        return 0.0


def blend_pitcher_stats(pit_25, pit_26):
    """
    Combine 2025 and 2026 pitcher HR/IP into a single stat dict.
    Weights 2026 proportionally by IP — early-season data contributes little.
    Returned stat dict uses float IP so parse_ip handles it correctly.
    """
    blended = {}
    for pid in set(pit_25) | set(pit_26):
        p25 = pit_25.get(pid, {})
        p26 = pit_26.get(pid, {})
        meta = p25 if p25 else p26
        st25 = p25.get('stat', {}) if p25 else {}
        st26 = p26.get('stat', {}) if p26 else {}
        ip25 = parse_ip(st25.get('inningsPitched', '0.0'))
        ip26 = parse_ip(st26.get('inningsPitched', '0.0'))
        hr25 = int(st25.get('homeRuns', 0))
        hr26 = int(st26.get('homeRuns', 0))
        total_ip = ip25 + ip26
        if total_ip == 0:
            continue
        blended[pid] = {
            'name':       meta.get('name', ''),
            'pitch_hand': meta.get('pitch_hand', 'R'),
            'stat': {'inningsPitched': total_ip, 'homeRuns': hr25 + hr26},
        }
    return blended


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

    # Determine temp bucket
    temp = wx['temp_f']
    if temp < 55:
        tbucket = 'temp_cold'
    elif temp > 75:
        tbucket = 'temp_hot'
    else:
        tbucket = 'temp_mild'

    # Determine wind bucket using effective wind toward CF
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
    """Single-season raw hitting. Returns {pid: {name, bat_side, pitch_hand, pa, hr}}."""
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
    """
    Fetch vs-LHP, vs-RHP, home, away, day, and night hitting splits.
    Returns {pid: {'vs_L':{hr,pa}, 'vs_R':{hr,pa}, 'home':{hr,pa}, 'away':{hr,pa},
                   'day':{hr,pa}, 'night':{hr,pa}}}
    """
    result = {}
    ids_list = list(player_ids)
    for i in range(0, len(ids_list), chunk_size):
        chunk = ids_list[i:i+chunk_size]
        try:
            r = session.get('https://statsapi.mlb.com/api/v1/people',
                params={'personIds': ','.join(str(x) for x in chunk),
                        'hydrate': f'stats(group=[hitting],type=[statSplits],season={season},sitCodes=[vl,vr,h,a,d,n])'},
                timeout=15)
            r.raise_for_status()
            for person in r.json().get('people', []):
                pid = person['id']
                entry = {'vs_L': {}, 'vs_R': {}, 'home': {}, 'away': {},
                         'day': {}, 'night': {}}
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
                        elif code in ('a', 'r'):   # 'a'=away or 'r'=road
                            entry['away'] = {'pa': pa, 'hr': hr}
                        elif code == 'd':
                            entry['day'] = {'pa': pa, 'hr': hr}
                        elif code == 'n':
                            entry['night'] = {'pa': pa, 'hr': hr}
                result[pid] = entry
        except Exception as e:
            print(f'  Splits chunk error: {e}')
        time.sleep(0.15)
    return result


def combine_hitting(raw_by_season, weights):
    """
    Combine per-season raw hitting into scoring-compatible format.
    raw_by_season: {2025: {pid: {pa,hr,...}}, 2026: {pid: {pa,hr,...}}}
    weights: {2025: 3.0, 2026: 1.0}  (only seasons with weight > 0 are used)
    Returns {pid: {name, bat_side, pitch_hand, stat: {homeRuns, plateAppearances}}}
    """
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
    """Barrel rate, exit velocity, and hard-hit% from Savant statcast leaderboard."""
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
            # Use actual game start hour in UTC; fall back to 23 (7 PM EDT)
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
    """Batter stats over the last `days_back` days ending the day before date_str."""
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
    """Pitcher stats over the last `days_back` days ending the day before date_str."""
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

def get_power_factor(batter_id, barrel_data):
    """Barrel rate + exit velocity + hard-hit% combined into one power multiplier."""
    d   = barrel_data.get(batter_id, {})
    brl = d.get('brl_pa',    LEAGUE_BRL_PA)    if isinstance(d, dict) else LEAGUE_BRL_PA
    ev  = d.get('exit_velo', LEAGUE_EXIT_VELO) if isinstance(d, dict) else LEAGUE_EXIT_VELO
    hh  = d.get('hard_hit',  LEAGUE_HARD_HIT)  if isinstance(d, dict) else LEAGUE_HARD_HIT
    brl_f = (max(brl, 0.001) / LEAGUE_BRL_PA)    ** 0.20
    ev_f  = (max(ev,  60)    / LEAGUE_EXIT_VELO)  ** 0.30
    hh_f  = (max(hh,  0.001) / LEAGUE_HARD_HIT)   ** 0.15
    return max(min(brl_f * ev_f * hh_f, POWER_CAP[1]), POWER_CAP[0])


def get_recent_form_factor(bid, recent_stats, batter_meta):
    """Last-30-day HR/PA vs season rate. Requires ≥15 recent PA."""
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
    """Pitcher's recent 30-day HR/9 vs season HR/9."""
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
    """Career batter vs that specific pitcher HR rate vs batter's overall rate."""
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
    """Batter's HR/PA at home vs away compared to overall rate."""
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


def get_day_night_batter_factor(bid, time_slot, splits, batter_meta):
    """Batter's HR/PA in day vs night games compared to overall rate."""
    side = 'day' if time_slot == 'day' else 'night'  # prime/late both use night split
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
    return max(min(blended, DAY_NIGHT_BATTER_CAP[1]), DAY_NIGHT_BATTER_CAP[0])


def get_bat_tracking_factor(batter_id, bat_tracking):
    bt    = bat_tracking.get(batter_id, {})
    blast = bt.get('blast_per_swing', BLAST_MEAN_ACT)
    speed = bt.get('bat_speed', BAT_SPEED_MEAN)
    bf = (max(blast, 0.001) / max(BLAST_MEAN_ACT, 0.001)) ** 0.20
    sf = (max(speed, 40) / BAT_SPEED_MEAN) ** 0.25
    return max(min((bf * sf) ** 0.5, BAT_TRACKING_CAP[1]), BAT_TRACKING_CAP[0])


def get_pitch_matchup_factor(batter_id, pitcher_id, bvp, pars):
    bp = bvp.get(batter_id, {}); pp = pars.get(pitcher_id, {})
    if not bp or not pp: return 1.0
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
    if wu < 0.40: return 1.0
    return max(min(math.exp((wrv / wu) * 0.05), PITCH_MATCHUP_CAP[1]), PITCH_MATCHUP_CAP[0])


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
    """
    Returns a platoon multiplier based on the batter's actual HR/PA vs that
    pitcher handedness. Falls back to generic PLATOON table when split PA < MIN_SPLIT_PA.
    """
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

    # Shrink toward generic fallback as split sample shrinks
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


# ── scorer ─────────────────────────────────────────────────────────────────────

def score_batter(bid, opp_pid, home_team, game_pk, time_slot, is_home,
                 batter_stats, pitcher_stats, barrel_data, park_tbl,
                 bat_tracking, bvp, pars, splits,
                 recent_bat, pitcher_recent, bvp_career,
                 weather, spring_stats, cfg):
    b = batter_stats.get(bid, {})
    p = pitcher_stats.get(opp_pid, {})
    if not b or not b.get('stat'): return None
    bs = b['stat']
    pa = bs.get('plateAppearances', 0) or bs.get('atBats', 0)
    hr = bs.get('homeRuns', 0)
    if not pa or pa < MIN_PA: return None

    conf  = min(pa, 300) / 300
    rate  = conf * (hr / pa) + (1 - conf) * LEAGUE_HR_PA

    # Pitcher vulnerability — tunable cap
    ph = 'R'; vuln = 1.0
    if p and p.get('stat'):
        ps = p['stat']
        ip = parse_ip(ps.get('inningsPitched', '0.0'))
        ph = p.get('pitch_hand', 'R')
        if ip >= MIN_IP:
            vuln = min((ps.get('homeRuns',0) / ip * 9) / LEAGUE_HR9, cfg['vuln_cap'])

    park     = park_tbl.get(home_team, 1.0)
    platoon  = get_platoon_factor(bid, ph, splits, b)

    # Power factor — tunable exponents
    d   = barrel_data.get(bid, {})
    brl = d.get('brl_pa',    LEAGUE_BRL_PA)    if isinstance(d, dict) else LEAGUE_BRL_PA
    ev  = d.get('exit_velo', LEAGUE_EXIT_VELO) if isinstance(d, dict) else LEAGUE_EXIT_VELO
    hh  = d.get('hard_hit',  LEAGUE_HARD_HIT)  if isinstance(d, dict) else LEAGUE_HARD_HIT
    brl_f = (max(brl, 0.001) / LEAGUE_BRL_PA)   ** cfg['pwr_brl_exp']
    ev_f  = (max(ev,  60)    / LEAGUE_EXIT_VELO) ** cfg['pwr_ev_exp']
    hh_f  = (max(hh,  0.001) / LEAGUE_HARD_HIT)  ** 0.15
    pwr_f = max(min(brl_f * ev_f * hh_f, POWER_CAP[1]), POWER_CAP[0])

    bt_f  = get_bat_tracking_factor(bid, bat_tracking)

    # Pitch matchup — tunable sensitivity scale
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

    # Recent batter form — tunable cap and min PA
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
    dn_bat_f = get_day_night_batter_factor(bid, time_slot, splits, b)

    # BvP — tunable weight (0=off, 1=full signal)
    raw_bvp = get_bvp_factor(bid, opp_pid, bvp_career, b)
    bvp_f   = 1.0 + cfg['bvp_weight'] * (raw_bvp - 1.0)

    exp_hr  = (rate * vuln * park * platoon * pwr_f * bt_f * pm_f
               * wx_f * st_f * form_f * pit_rf * bvp_f * ha_f
               * dn_bat_f * AVG_PA_GAME * cfg['calibration'])
    hr_prob = 1 - math.exp(-exp_hr)

    return {
        'batter_id': bid, 'name': b['name'], 'bat_side': b.get('bat_side', 'R'),
        'hr_prob': round(hr_prob, 4),
    }


def _precompute_row(bid, opp_id, home_team, pk, time_slot, is_home, away_team,
                    batter_stats, pit_stats, barrel, park_tbl, bat_t,
                    bvp_arsenal, pars, splits, recent_bat, pitcher_recent,
                    bvp_career, weather, spring, weather_splits):
    """Pre-compute all cfg-invariant factors for one batter-game row.

    Returns a tuple of pre-computed values, or None if the batter is ineligible.
    The grid-search inner loop then only needs cheap arithmetic per config.
    """
    b = batter_stats.get(bid, {})
    if not b or not b.get('stat'):
        return None
    bs = b['stat']
    pa = bs.get('plateAppearances', 0) or bs.get('atBats', 0)
    hr = bs.get('homeRuns', 0)
    if not pa or pa < MIN_PA:
        return None

    conf = min(pa, 300) / 300
    rate = conf * (hr / pa) + (1 - conf) * LEAGUE_HR_PA

    # Pitcher: store raw (uncapped) vulnerability so the cap can be tuned per cfg
    p = pit_stats.get(opp_id, {})
    ph = 'R'
    raw_vuln = 1.0
    if p and p.get('stat'):
        ps = p['stat']
        ip = parse_ip(ps.get('inningsPitched', '0.0'))
        ph = p.get('pitch_hand', 'R')
        if ip >= MIN_IP:
            raw_vuln = (ps.get('homeRuns', 0) / ip * 9) / LEAGUE_HR9

    park    = park_tbl.get(home_team, 1.0)
    platoon = get_platoon_factor(bid, ph, splits, b)

    # Power: store base ratios so cfg exponents can be applied per config
    d   = barrel.get(bid, {})
    brl = d.get('brl_pa',    LEAGUE_BRL_PA)    if isinstance(d, dict) else LEAGUE_BRL_PA
    ev  = d.get('exit_velo', LEAGUE_EXIT_VELO) if isinstance(d, dict) else LEAGUE_EXIT_VELO
    hh  = d.get('hard_hit',  LEAGUE_HARD_HIT)  if isinstance(d, dict) else LEAGUE_HARD_HIT
    brl_ratio = max(brl, 0.001) / LEAGUE_BRL_PA
    ev_ratio  = max(ev, 60)     / LEAGUE_EXIT_VELO
    hh_f      = (max(hh, 0.001) / LEAGUE_HARD_HIT) ** 0.15   # exponent is constant

    bt_f = get_bat_tracking_factor(bid, bat_t)

    # Pitch matchup: store normalized RV; matchup_scale applied per cfg
    bp = bvp_arsenal.get(bid, {}); pp = pars.get(opp_id, {})
    norm_rv = None
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
        if wu >= 0.40:
            norm_rv = wrv / wu

    wx_f = calc_weather_factor(home_team, pk, weather)
    st_f = get_spring_factor(bid, spring)

    # Recent form raw data; form_min_pa and form_cap_hi are tunable per cfg
    rd   = recent_bat.get(bid, {})
    r_pa = rd.get('pa', 0); r_hr = rd.get('hr', 0)

    pit_rf     = get_pitcher_recent_factor(opp_id, pit_stats, pitcher_recent)
    ha_f       = get_home_away_factor(bid, is_home, splits, b)
    dn_bat_f   = get_day_night_batter_factor(bid, time_slot, splits, b)
    raw_bvp    = get_bvp_factor(bid, opp_id, bvp_career, b)   # bvp_weight applied per cfg
    wx_hist_f  = get_weather_hist_factor(bid, pk, home_team, weather, weather_splits, b)

    return (bid, pk, away_team, home_team,
            rate, raw_vuln, park, platoon,
            brl_ratio, ev_ratio, hh_f,
            bt_f, norm_rv,
            wx_f, st_f,
            r_pa, r_hr, hr, pa,
            pit_rf, ha_f, raw_bvp,
            b['name'], wx_hist_f, dn_bat_f)


# ── main ───────────────────────────────────────────────────────────────────────

# All completed regular-season dates since opening day 2026-03-27.
# Add new dates here as games complete — the rest of the script handles them automatically.
TEST_DATES = [
    '2026-03-27','2026-03-28','2026-03-29','2026-03-30',
    '2026-03-31','2026-04-01','2026-04-02','2026-04-03',
    '2026-04-04','2026-04-05','2026-04-06','2026-04-07',
    '2026-04-08','2026-04-09','2026-04-10','2026-04-11',
    '2026-04-12','2026-04-13','2026-04-14','2026-04-15',
    '2026-04-16', '2026-04-17',
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
    save_cache(f'bat_tracking_{PITCHER_SEASON}',
               {'bat_t': bat_t, 'speed_mean': BAT_SPEED_MEAN, 'blast_mean': BLAST_MEAN_ACT})
else:
    bat_t         = _bat_cache['bat_t']
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
spring = fetch_spring_stats()   # already self-caching via SPRING_CACHE JSON

print('  Park factors: using hardcoded 2025/2026 day/night tables')

print('  Career BvP lookup...')
bvp_career_global = load_cache('bvp_career')
if bvp_career_global is None:
    print('  WARNING: bvp_career.pkl not found — BvP factor disabled.')
    bvp_career_global = {}
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
    _dc = load_cache(f'date_{date_str}')
    if _dc is not None:
        print(f'\n  --- {date_str} --- (cached)')
        date_cache[date_str] = _dc
        continue

    print(f'\n  --- {date_str} ---')
    games = get_games(date_str)
    slots = {}
    for g in games:
        s = get_game_time_slot(g['game_date'])
        slots[s] = slots.get(s, 0) + 1
    print(f'  {len(games)} games ({slots.get("day",0)} day / {slots.get("prime",0)} prime / {slots.get("late",0)} late)')

    batter_map = {}
    for g in games:
        pk = g['game_pk']
        if g['home_lineup_ids']:
            hi, ai = g['home_lineup_ids'], g['away_lineup_ids']
        else:
            hi = get_roster(g['home_team_id']); ai = get_roster(g['away_team_id'])
            time.sleep(0.3)
        batter_map[pk] = {
            'home': hi, 'away': ai,
            'home_team': g['home_team'], 'away_team': g['away_team'],
            'time_slot': get_game_time_slot(g['game_date']),
        }

    all_bat = {bid for v in batter_map.values() for s in ('home','away') for bid in v[s]}
    all_pit = {g[k] for g in games for k in ('home_pitcher_id','away_pitcher_id') if g[k]}
    print(f'  Batters: {len(all_bat)}  Pitchers: {len(all_pit)}')


    # Recent form disabled — not enough 2026 data yet; factors return 1.0
    recent_bat     = {}
    pitcher_recent = {}

    from datetime import timedelta
    prev_date = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')

    print('  Fetching stats concurrently...')
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_25  = pool.submit(fetch_raw_hitting, all_bat, 2025)
        fut_26  = pool.submit(fetch_raw_hitting, all_bat, 2026)
        fut_pt  = pool.submit(fetch_stats_bulk, all_pit, 'pitching', PITCHER_SEASON)
        fut_wx  = pool.submit(fetch_weather, games, date_str)
        fut_ac  = pool.submit(get_actual_hrs, date_str)
        fut_pac = pool.submit(get_actual_hrs, prev_date)
        raw_25      = fut_25.result()
        raw_26      = fut_26.result()
        pit_stats   = fut_pt.result()
        weather     = fut_wx.result()
        actual      = fut_ac.result()
        prev_actual = fut_pac.result()

    outdoor = sum(1 for g in games if not STADIUMS.get(g['home_team'],{}).get('indoor',True))
    prev_hr_hitters = set(prev_actual.keys())
    print(f'  Hitting 2025: {len(raw_25)} with ≥{MIN_PA} PA')
    print(f'  Hitting 2026: {len(raw_26)} with ≥{MIN_PA} PA')
    print(f'  Pitcher stats: {sum(1 for v in pit_stats.values() if v["stat"])} with data')
    print(f'  Weather: {len(weather)}/{outdoor} outdoor')
    print(f'  Actual HRs: {len(actual)} HR hitters')
    print(f'  Prev-day HRs: {len(prev_hr_hitters)} batters hit HR yesterday')

    date_cache[date_str] = {
        'games': games, 'batter_map': batter_map,
        'raw_25': raw_25, 'raw_26': raw_26,
        'pit_stats': pit_stats,
        'recent_bat': recent_bat, 'pitcher_recent': pitcher_recent,
        'bvp_career': bvp_career_global,
        'weather': weather,
        'actual': actual, 'all_bat': all_bat,
        'prev_hr_hitters': prev_hr_hitters,
    }
    save_cache(f'date_{date_str}', date_cache[date_str])
    time.sleep(0.5)

# ── 2b. Global splits (fetched once for ALL batters across all dates) ─────────
all_batter_ids = set()
for dc in date_cache.values():
    all_batter_ids.update(dc['all_bat'])
print(f'\n  Hitting splits 2025 (all {len(all_batter_ids)} batters)...')
splits_2025_global = load_cache('splits_2025')
if splits_2025_global is None:
    splits_2025_global = fetch_hitting_splits(all_batter_ids, 2025)
    save_cache('splits_2025', splits_2025_global)
    print(f'  Fetched & cached: {len(splits_2025_global)} batters')
else:
    # If new batters appeared (new dates added), fetch the missing ones
    missing = all_batter_ids - set(splits_2025_global.keys())
    if missing:
        print(f'  {len(missing)} new batters — fetching splits...')
        new_splits = fetch_hitting_splits(missing, 2025)
        splits_2025_global.update(new_splits)
        save_cache('splits_2025', splits_2025_global)
        print(f'  Updated cache: {len(splits_2025_global)} batters total')
    else:
        print(f'  From cache: {len(splits_2025_global)} batters')

# ── 3. Grid search ────────────────────────────────────────────────────────────
print()
print('='*68)
print(f'  TUNING GRID SEARCH  ({len(ALL_TUNE_CONFIGS)} configs × {len(TEST_DATES)} dates)')
print('='*68)
print()

PICKS_SKIP  = 1   # skip this many top-ranked players per game
PICKS_TAKE  = 5   # then take this many
PICKS_PER_GAME = PICKS_TAKE

# Pre-build batter_stats and park_tbl once (single season/pf config)
sw_label, sw_weights = SEASON_WEIGHT_CONFIGS[0]
pf_label, pf_blend   = PF_BLEND_CONFIGS[0]
park_tbl = build_park_table(pf_blend)

# Pre-build per-date batter_stats
date_batter_stats = {}
for date_str in TEST_DATES:
    dc = date_cache[date_str]
    raw_by_season = {}
    for yr, key in [(2025, 'raw_25'), (2026, 'raw_26')]:
        if sw_weights.get(yr, 0) > 0:
            raw_by_season[yr] = dc[key]
    date_batter_stats[date_str] = combine_hitting(raw_by_season, sw_weights)

# Pre-compute all cfg-invariant factors once per batter-game row.
# The grid search inner loop then only does cheap arithmetic per config.
print('  Building 2026 starters list from starting lineups...')
starters_2026 = set()
for date_str in TEST_DATES:
    dc = date_cache[date_str]
    for g in dc['games']:
        hi = g.get('home_lineup_ids', [])
        ai = g.get('away_lineup_ids', [])
        if hi:
            starters_2026.update(hi)
        if ai:
            starters_2026.update(ai)
print(f'  {len(starters_2026)} unique starters found across {len(TEST_DATES)} dates')

print('  Pre-computing invariant row factors (starters only)...')
precomputed_rows = {}   # {date_str: [row_tuple, ...]}
for date_str in TEST_DATES:
    dc           = date_cache[date_str]
    batter_stats = date_batter_stats[date_str]
    pit_blended  = blend_pitcher_stats(dc['pit_stats'], dc.get('pit_stats_26', {}))
    wx_splits    = dc.get('weather_splits', weather_splits_global)
    rows = []
    for g in dc['games']:
        pk   = g['game_pk']
        info = dc['batter_map'][pk]
        for side, opp_id, is_home in [
            ('away', g['home_pitcher_id'], False),
            ('home', g['away_pitcher_id'], True),
        ]:
            for bid in info[side]:
                if bid not in starters_2026:
                    continue
                row = _precompute_row(
                    bid, opp_id, info['home_team'], pk,
                    info['time_slot'], is_home, info['away_team'],
                    batter_stats, pit_blended, barrel, park_tbl, bat_t,
                    bvp, pars, splits_2025_global,
                    dc['recent_bat'], dc['pitcher_recent'], dc['bvp_career'],
                    dc['weather'], spring, wx_splits,
                )
                if row is not None:
                    rows.append(row)
    precomputed_rows[date_str] = rows
    print(f'    {date_str}: {len(rows)} eligible rows')

_grid_key = 'grid_' + hashlib.md5(
    ('v6_park_split' + str(sorted(TUNE_PARAM_GRID.items())) + ','.join(TEST_DATES)).encode()
).hexdigest()[:12]
_grid_cache = load_cache(_grid_key)

if _grid_cache is not None:
    all_avg_prec, all_date_prec, grid_results = _grid_cache
    print(f'  Loaded {len(ALL_TUNE_CONFIGS):,}-config grid results from cache  [{_grid_key}]')
else:
    # ── Vectorised setup ─────────────────────────────────────────────────────
    xp = np
    print('  Device: CPU NumPy')

    # Convert one date's precomputed rows into a dict of numpy arrays.
    # Tuple layout (matches _precompute_row return):
    #   0:bid  1:pk  2:away  3:home  4:rate  5:raw_vuln  6:park  7:platoon
    #   8:brl_ratio  9:ev_ratio  10:hh_f  11:bt_f  12:norm_rv
    #   13:wx_f  14:st_f  15:r_pa  16:r_hr  17:season_hr  18:season_pa
    #   19:pit_rf  20:ha_f  21:raw_bvp  22:name  23:wx_hist_f  24:dn_bat_f
    def _rows_to_np(rows):
        if not rows:
            return None
        cols    = list(zip(*rows))
        norm_rv = cols[12]
        return {
            'bids':      np.array(cols[0],  dtype=np.int64),
            'pks':       np.array(cols[1],  dtype=np.int64),
            'names':     list(cols[22]),
            'away_t':    list(cols[2]),
            'home_t':    list(cols[3]),
            'rate':      np.array(cols[4],  dtype=np.float32),
            'raw_vuln':  np.array(cols[5],  dtype=np.float32),
            'park':      np.array(cols[6],  dtype=np.float32),
            'platoon':   np.array(cols[7],  dtype=np.float32),
            'brl_ratio': np.array(cols[8],  dtype=np.float32),
            'ev_ratio':  np.array(cols[9],  dtype=np.float32),
            'hh_f':      np.array(cols[10], dtype=np.float32),
            'bt_f':      np.array(cols[11], dtype=np.float32),
            'norm_rv':   np.array([v if v is not None else 0.0 for v in norm_rv],
                                  dtype=np.float32),
            'has_norm':  np.array([v is not None for v in norm_rv], dtype=np.bool_),
            'wx_f':      np.array(cols[13], dtype=np.float32),
            'st_f':      np.array(cols[14], dtype=np.float32),
            'r_pa':      np.array(cols[15], dtype=np.float32),
            'r_hr':      np.array(cols[16], dtype=np.float32),
            'season_hr': np.array(cols[17], dtype=np.float32),
            'season_pa': np.array(cols[18], dtype=np.float32),
            'pit_rf':    np.array(cols[19], dtype=np.float32),
            'ha_f':      np.array(cols[20], dtype=np.float32),
            'raw_bvp':   np.array(cols[21], dtype=np.float32),
            'wx_hist_f': np.array(cols[23], dtype=np.float32),
            'dn_bat_f':  np.array(cols[24], dtype=np.float32),
        }

    # Build per-date numpy arrays + eligible-row game groups + actual-HR labels
    date_arrs       = {}   # {date: dict of np arrays}
    date_gmeta      = {}   # {date: {pk: np.array of eligible row indices}}
    date_actual_arr = {}   # {date: np.int8, 1 if batter hit a HR that day}

    for date_str in TEST_DATES:
        dc   = date_cache[date_str]
        arrs = _rows_to_np(precomputed_rows[date_str])
        date_arrs[date_str] = arrs
        if arrs is None:
            date_gmeta[date_str]      = {}
            date_actual_arr[date_str] = np.zeros(0, dtype=np.int8)
            continue

        N        = len(arrs['bids'])
        prev_hr  = dc['prev_hr_hitters']
        actual   = dc['actual']
        eligible = np.array([int(b) not in prev_hr for b in arrs['bids']], dtype=np.bool_)
        date_actual_arr[date_str] = np.array(
            [1 if actual.get(int(b), {}).get('hrs', 0) > 0 else 0
             for b in arrs['bids']], dtype=np.int8)

        game_elig = {}
        for i in range(N):
            if eligible[i]:
                pk_i = int(arrs['pks'][i])
                if pk_i not in game_elig:
                    game_elig[pk_i] = []
                game_elig[pk_i].append(i)
        date_gmeta[date_str] = {pk: np.array(v) for pk, v in game_elig.items()}

    # Config parameter arrays (n_configs,) float32
    _cfgk    = ['vuln_cap','pwr_brl_exp','pwr_ev_exp','matchup_scale',
                 'form_min_pa','form_cap_hi','bvp_weight','calibration']
    cfg_np   = {k: np.array([c[k] for c in ALL_TUNE_CONFIGS], dtype=np.float32)
                for k in _cfgk}
    cfg_skip = np.array([c['picks_skip'] for c in ALL_TUNE_CONFIGS], dtype=np.int32)

    # Upload static row arrays once — they stay resident across all batches
    date_gpu = {}
    _row_keys = ('rate','raw_vuln','park','platoon','brl_ratio','ev_ratio','hh_f',
                 'bt_f','norm_rv','has_norm','wx_f','st_f','r_pa','r_hr',
                 'season_hr','season_pa','pit_rf','ha_f','raw_bvp','wx_hist_f',
                 'dn_bat_f')
    for date_str in TEST_DATES:
        arrs = date_arrs[date_str]
        date_gpu[date_str] = (
            {k: xp.asarray(arrs[k]) for k in _row_keys} if arrs is not None else None
        )

    # Scalar caps as Python floats (avoids repeated attribute lookups in inner loop)
    _pmc0  = float(PITCH_MATCHUP_CAP[0]);  _pmc1 = float(PITCH_MATCHUP_CAP[1])
    _pc0   = float(POWER_CAP[0]);          _pc1  = float(POWER_CAP[1])
    _rfc0  = float(RECENT_FORM_CAP[0])
    _avpa  = float(AVG_PA_GAME)

    # ── Vectorised grid search ──────────────────────────────────────────────────
    BATCH_SIZE   = 50_000
    n_configs    = len(ALL_TUNE_CONFIGS)
    n_dates      = len(TEST_DATES)
    all_avg_prec  = np.zeros(n_configs, dtype=np.float64)
    all_date_prec = np.zeros((n_configs, n_dates), dtype=np.float32)

    for batch_start in range(0, n_configs, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_configs)
        B         = batch_end - batch_start

        # Config arrays shaped (B, 1) so they broadcast against row arrays (N,)
        def _gc(k):
            return xp.asarray(cfg_np[k][batch_start:batch_end, np.newaxis])
        g_vc  = _gc('vuln_cap')
        g_bre = _gc('pwr_brl_exp')
        g_eve = _gc('pwr_ev_exp')
        g_ms  = _gc('matchup_scale')
        g_fpa = _gc('form_min_pa')
        g_fch = _gc('form_cap_hi')
        g_bvw = _gc('bvp_weight')
        g_cal = _gc('calibration')
        skip_b = cfg_skip[batch_start:batch_end]   # (B,) CPU int32

        batch_hits  = np.zeros((B, n_dates), dtype=np.int32)
        batch_npick = np.zeros((B, n_dates), dtype=np.int32)

        for di, date_str in enumerate(TEST_DATES):
            garrs = date_gpu[date_str]
            if garrs is None:
                continue

            # ── Score every (config, row) pair in one broadcast ─────
            vuln  = xp.minimum(garrs['raw_vuln'], g_vc)                    # (B, N)
            brl_f = garrs['brl_ratio'] ** g_bre                             # (B, N)
            ev_f  = garrs['ev_ratio']  ** g_eve                             # (B, N)
            pwr_f = xp.clip(brl_f * ev_f * garrs['hh_f'], _pc0, _pc1)     # (B, N)

            pm_raw = xp.exp(garrs['norm_rv'] * g_ms)                       # (B, N)
            pm_f   = xp.where(garrs['has_norm'],
                              xp.clip(pm_raw, _pmc0, _pmc1), 1.0)          # (B, N)

            # Recent form — vectorised conditional
            sr     = garrs['season_hr'] / xp.maximum(garrs['season_pa'], 1e-6)
            rr     = xp.where(garrs['r_pa'] > 0,
                              garrs['r_hr'] / xp.maximum(garrs['r_pa'], 1e-6), 0.0)
            raw_f  = xp.where(sr > 0, rr / xp.maximum(sr, 1e-6), 1.0)
            conf_f = xp.minimum(garrs['r_pa'], 60.0) / 60.0
            blend  = conf_f * raw_f + (1.0 - conf_f)
            active = ((garrs['r_pa'] >= g_fpa) &
                      (garrs['season_hr'] > 0) &
                      (garrs['season_pa'] > 0))                             # (B, N)
            form_f = xp.where(active, xp.clip(blend, _rfc0, g_fch), 1.0)  # (B, N)

            bvp_f  = 1.0 + g_bvw * (garrs['raw_bvp'] - 1.0)               # (B, N)
            exp_hr = (garrs['rate'] * vuln * garrs['park'] * garrs['platoon']
                      * pwr_f * garrs['bt_f'] * pm_f
                      * garrs['wx_f'] * garrs['st_f'] * form_f
                      * garrs['pit_rf'] * bvp_f * garrs['ha_f']
                      * garrs['wx_hist_f'] * garrs['dn_bat_f']
                      * _avpa * g_cal)                                      # (B, N)
            hr_prob = 1.0 - xp.exp(-exp_hr)                                # (B, N)

            # Transfer result to CPU (no-op when xp is np)
            hp = xp.asnumpy(hr_prob) if xp is not np else hr_prob          # (B, N)

            # ── Vectorised pick selection (CPU NumPy, per game per picks_skip) ────
            gmeta      = date_gmeta[date_str]
            actual_arr = date_actual_arr[date_str]

            for s in range(4):                        # picks_skip ∈ {0,1,2,3}
                cmask = (skip_b == s)
                if not cmask.any():
                    continue
                Bs   = int(cmask.sum())
                hp_s = hp[cmask, :]                   # (Bs, N)
                hits_s = np.zeros(Bs, dtype=np.int32)
                nps    = 0

                for elig_idx in gmeta.values():
                    n_elig = len(elig_idx)
                    if n_elig <= s:
                        continue
                    take = min(PICKS_TAKE, n_elig - s)
                    need = s + take

                    probs = hp_s[:, elig_idx]                              # (Bs, n_elig)
                    # argpartition is O(n) vs argsort O(n log n) — faster for large n_elig
                    kth   = min(need, n_elig) - 1
                    top   = np.argpartition(-probs, kth, axis=1)[:, :need] # (Bs, need)
                    tp    = probs[np.arange(Bs)[:, None], top]             # (Bs, need)
                    order = np.argsort(-tp, axis=1)[:, s:s + take]        # (Bs, take)
                    sel   = top[np.arange(Bs)[:, None], order]             # (Bs, take)
                    sel_g = elig_idx[sel]                                  # (Bs, take)

                    hits_s += actual_arr[sel_g].sum(axis=1)
                    nps    += take

                batch_hits[cmask, di]  += hits_s
                batch_npick[cmask, di] += nps

        prec = np.where(batch_npick > 0,
                        batch_hits / np.maximum(batch_npick, 1), 0.0)
        all_avg_prec[batch_start:batch_end]     = prec.mean(axis=1)
        all_date_prec[batch_start:batch_end, :] = prec.astype(np.float32)

        done = batch_end
        if done % 200_000 < BATCH_SIZE or done == n_configs:
            print(f'  {done:>8,}/{n_configs:,} done   '
                  f'best so far: {all_avg_prec[:done].max():.1%}')

    # ── Sort and build grid_results ───────────────────────────────────────────────
    sorted_ci = np.argsort(-all_avg_prec)
    grid_results = []
    for rank_i in sorted_ci[:20]:
        grid_results.append({
            'cfg':          ALL_TUNE_CONFIGS[int(rank_i)],
            'avg_prec':     float(all_avg_prec[rank_i]),
            'per_date_prec': {d: round(float(all_date_prec[rank_i, di]), 3)
                              for di, d in enumerate(TEST_DATES)},
            'date_picks':   {},   # filled for best config below
        })

    # Re-run the best config once (fast single-config Python loop) to collect picks
    _best_cfg = grid_results[0]['cfg']
    _best_date_picks = {}
    for date_str in TEST_DATES:
        dc      = date_cache[date_str]
        prev_hr = dc['prev_hr_hitters']
        _picks  = {}
        for row in precomputed_rows[date_str]:
            (bid, pk, away_team, home_team,
             rate, raw_vuln, park, platoon,
             brl_ratio, ev_ratio, hh_f,
             bt_f, norm_rv, wx_f, st_f,
             r_pa, r_hr, season_hr, season_pa,
             pit_rf, ha_f, raw_bvp, name, wx_hist_f, dn_bat_f) = row
            vuln  = min(raw_vuln, _best_cfg['vuln_cap'])
            brl_f = brl_ratio ** _best_cfg['pwr_brl_exp']
            ev_f  = ev_ratio  ** _best_cfg['pwr_ev_exp']
            pwr_f = max(min(brl_f * ev_f * hh_f, POWER_CAP[1]), POWER_CAP[0])
            pm_f  = (max(min(math.exp(norm_rv * _best_cfg['matchup_scale']),
                             PITCH_MATCHUP_CAP[1]), PITCH_MATCHUP_CAP[0])
                     if norm_rv is not None else 1.0)
            if r_pa >= _best_cfg['form_min_pa'] and season_hr > 0 and season_pa > 0:
                sr     = season_hr / season_pa
                raw_f  = (r_hr / r_pa) / sr if sr > 0 and r_pa > 0 else 1.0
                conf_f = min(r_pa, 60) / 60
                form_f = max(min(conf_f * raw_f + (1 - conf_f),
                                 _best_cfg['form_cap_hi']), RECENT_FORM_CAP[0])
            else:
                form_f = 1.0
            bvp_f  = 1.0 + _best_cfg['bvp_weight'] * (raw_bvp - 1.0)
            exp_hr = (rate * vuln * park * platoon * pwr_f * bt_f * pm_f
                      * wx_f * st_f * form_f * pit_rf * bvp_f * ha_f
                      * wx_hist_f * dn_bat_f * AVG_PA_GAME * _best_cfg['calibration'])
            hr_prob = 1 - math.exp(-exp_hr)
            _picks.setdefault(pk, []).append({
                'batter_id': bid, 'name': name, 'hr_prob': round(hr_prob, 4),
                'game_pk': pk, 'home_team': home_team, 'away_team': away_team,
            })
        _day = []
        for pk, game_rows in _picks.items():
            elig = [r for r in game_rows if r['batter_id'] not in prev_hr]
            elig.sort(key=lambda x: -x['hr_prob'])
            skip = _best_cfg['picks_skip']
            _day.extend(elig[skip:skip + PICKS_TAKE])
        _best_date_picks[date_str] = _day
    grid_results[0]['date_picks'] = _best_date_picks
    save_cache(_grid_key, (all_avg_prec, all_date_prec, grid_results))
    print(f'  Grid results saved to cache  [{_grid_key}]')

# ── 4. Results ────────────────────────────────────────────────────────────────
grid_results.sort(key=lambda x: -x['avg_prec'])
best  = grid_results[0]
best_cfg = best['cfg']

print()
print('='*68)
print(f'  TOP 20 CONFIGS  (top {PICKS_PER_GAME} picks/game, {len(TEST_DATES)} dates)')
print('='*68)
print(f"  {'Rank':<5} {'Prec':>7}  Config")
print(f"  {'-'*68}")
for i, r in enumerate(grid_results[:20], 1):
    c = r['cfg']
    desc = (f"brl={c['pwr_brl_exp']} ev={c['pwr_ev_exp']} "
            f"form_cap={c['form_cap_hi']} form_pa={c['form_min_pa']} "
            f"vuln={c['vuln_cap']} mtch={c['matchup_scale']} "
            f"bvp={c['bvp_weight']}")
    print(f"  {i:<5} {r['avg_prec']:>6.1%}  {desc}")

print()
print('  BEST CONFIG:')
for k, v in best_cfg.items():
    print(f"    {k:<20} = {v}")
print(f"  Avg precision: {best['avg_prec']:.1%}")

print()
print('  Per-date precision (best config):')
print(f"  {'Date':<12} {'Picks':>6} {'Hits':>6} {'Prec':>7}  {'Actual HRs':>10}")
print(f"  {'-'*48}")
for date_str in TEST_DATES:
    picks  = best['date_picks'][date_str]
    n_hits = sum(1 for p in picks
                 if date_cache[date_str]['actual'].get(p['batter_id'], {}).get('hrs', 0) > 0)
    pr    = n_hits / len(picks) if picks else 0.0
    n_act = len(date_cache[date_str]['actual'])
    print(f"  {date_str:<12} {len(picks):>6} {n_hits:>6} {pr:>6.0%}  {n_act:>10}")

print()
print(f'  Per-game picks — {TEST_DATES[-1]} (best config):')
last_date  = TEST_DATES[-1]
last_picks = best['date_picks'][last_date]
last_dc    = date_cache[last_date]
print(f"  {'Matchup':<34} {'Player':<24} {'prob':>6}  Hit?")
print(f"  {'-'*72}")
for p in sorted(last_picks, key=lambda x: (x['game_pk'], -x['hr_prob'])):
    hit     = last_dc['actual'].get(p['batter_id'], {}).get('hrs', 0) > 0
    matchup = f"{p['away_team']} @ {p['home_team']}"
    print(f"  {matchup:<34} {p['name']:<24} {p['hr_prob']:>5.1%}  {'YES' if hit else '---'}")
