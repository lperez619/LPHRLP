"""
HR Prediction Trainer — Logistic Regression Edition
Trains a logistic regression on log-transformed batter-game features.
Training data: 2024 + 2025 full seasons (fetched once, cached).
Validation data: 2026 dates (same data pipeline as before).
Objective: log-loss over every batter-game row (~300× more signal than
precision@top-K).  Gradient descent finds the optimal continuous weighting
of all features — no need for a discrete grid over exponents / caps.

Small grid kept only for non-learnable pick-selection knobs:
  picks_skip, picks_take.

Special cases:
  Tampa Bay  — always uses 2024 Savant PF (Tropicana Field; 2025 was temp venue)
  Athletics  — only 2025/2026 Savant PF; no hardcoded Coliseum fallback (new stadium 2025)

Usage: python "HR Trainer.py"
"""
import math, time, io, json, os, itertools, pickle, hashlib, warnings
import numpy as np
from datetime import datetime
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

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

# ── pick selection grid (the only knobs left) ──────────────────────────────────
PICK_SKIP_OPTIONS = [0, 1, 2, 3]
PICK_TAKE_OPTIONS = [3, 5, 7, 10]

# ── training config ────────────────────────────────────────────────────────────
TRAIN_YEARS = [2024, 2025]           # full seasons to train on
USE_HISTORICAL_TRAINING = True       # False → leave-one-date-out CV on 2026 only

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

# ── Hardcoded day/night park factors (index / 100 → multiplicative) ────────────
# Source: user-supplied Savant-style data.
# Teams absent from a 2026 table → that slot uses 2025 only (no blend).
# Tampa Bay 2025 = Steinbrenner Field (temp venue) → overridden with Tropicana values.

# Tampa Bay 2026 values (Tropicana Field — overrides 2025 which was temp venue)
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

# 2026 DAY — only 16 teams listed; others use 2025 only
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

# 2026 NIGHT — only 15 teams (Phillies dropped: 0 HRs = no usable data yet)
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
    print(f'  Cached -> {os.path.basename(path)}')


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


# ── zone matchup data ─────────────────────────────────────────────────────────
LEAGUE_AVG_SLG = 0.400  # approximate league SLG for zone matchup baseline
ZONE_IDS = ['01','02','03','04','05','06','07','08','09','11','12','13','14']

def _fetch_zone_single(pid, group, season):
    """Fetch hotColdZones for a single player. Returns {zone_id: value} or None."""
    try:
        r = session.get(
            f'https://statsapi.mlb.com/api/v1/people/{pid}/stats',
            params={'stats': 'hotColdZones', 'group': group, 'season': season},
            timeout=10)
        r.raise_for_status()
        stats = r.json().get('stats', [])
        if not stats:
            return None
        splits = stats[0].get('splits', [])
        # For batters: use sluggingPercentage; for pitchers: use numberOfPitches
        target_stat = 'sluggingPercentage' if group == 'hitting' else 'numberOfPitches'
        for s in splits:
            if s['stat']['name'] == target_stat:
                zones = {}
                for z in s['stat']['zones']:
                    val = z.get('value', '0')
                    try:
                        zones[z['zone']] = float(val) if val else 0.0
                    except (ValueError, TypeError):
                        zones[z['zone']] = 0.0
                return zones if zones else None
        return None
    except Exception:
        return None


def fetch_batter_zones(batter_ids, season=2025):
    """Fetch batter zone SLG data. Returns {pid: {zone_id: slg}}."""
    result = {}
    ids = list(batter_ids)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch_zone_single, pid, 'hitting', season): pid
                for pid in ids}
        for fut in as_completed(futs):
            pid = futs[fut]
            data = fut.result()
            if data:
                result[pid] = data
    return result


def fetch_pitcher_zones(pitcher_ids, season=2025):
    """Fetch pitcher zone pitch-count data. Returns {pid: {zone_id: count}}."""
    result = {}
    ids = [p for p in pitcher_ids if p is not None]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch_zone_single, pid, 'pitching', season): pid
                for pid in ids}
        for fut in as_completed(futs):
            pid = futs[fut]
            data = fut.result()
            if data:
                result[pid] = data
    return result


def compute_zone_matchup(batter_id, pitcher_id, batter_zones, pitcher_zones):
    """Compute zone-weighted SLG matchup factor.

    Returns ratio of batter's weighted SLG (weighted by pitcher zone usage)
    to league average SLG. Values > 1 mean the batter is strong where the
    pitcher throws; < 1 means the batter is weak there.
    """
    bz = batter_zones.get(batter_id)
    pz = pitcher_zones.get(pitcher_id)
    if not bz or not pz:
        return 1.0  # neutral if no data

    # Convert pitcher counts to percentages
    total = sum(pz.get(z, 0) for z in ZONE_IDS)
    if total < 50:  # too few pitches to be meaningful
        return 1.0

    weighted_slg = 0.0
    usage_sum = 0.0
    for z in ZONE_IDS:
        pct = pz.get(z, 0) / total
        slg = bz.get(z, LEAGUE_AVG_SLG)
        weighted_slg += pct * slg
        usage_sum += pct

    if usage_sum < 0.5:
        return 1.0

    raw = weighted_slg / LEAGUE_AVG_SLG
    # Shrink toward 1.0 based on data quality
    return max(min(raw, 2.0), 0.40)


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
                    bvp_career, weather, spring, weather_splits,
                    batter_zones=None, pitcher_zones=None):
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
    zone_f     = compute_zone_matchup(bid, opp_id,
                                      batter_zones or {}, pitcher_zones or {})

    return (bid, pk, away_team, home_team,
            rate, raw_vuln, park, platoon,
            brl_ratio, ev_ratio, hh_f,
            bt_f, norm_rv,
            wx_f, st_f,
            r_pa, r_hr, hr, pa,
            pit_rf, ha_f, raw_bvp,
            b['name'], wx_hist_f, dn_bat_f, zone_f)


# ── feature extraction for logistic regression ─────────────────────────────────
# Log-transformed versions of the multiplicative factors.  The logistic
# regression sees ∑ wᵢ·log(fᵢ) which is equivalent to log(∏ fᵢ^wᵢ) —
# a continuously-weighted version of the old multiplicative product.
#
# Features are split into CORE (available in historical data) and ADJ
# (only populated for 2026 live dates — weather, BvP, form, etc.).
# Historical training uses core features only; adjustment features are
# applied as post-hoc logit offsets for the historical model, or included
# directly in the 2026-only CV model.

CORE_FEATURE_NAMES = [
    'log_rate',      # batter regressed HR/PA
    'log_vuln',      # pitcher HR/9 ratio vs league (uncapped)
    'log_park',      # venue park factor
    'log_platoon',   # platoon factor (batter vs pitcher hand)
    'log_power',     # combined barrel × exit velo × hard-hit (was 3 separate)
    'log_bt',        # bat tracking (speed × blast)
    'norm_rv',       # pitch arsenal matchup (weighted run value, linear)
    'has_norm_rv',   # indicator: arsenal matchup data available
    'log_ha',        # batter home/away split
    'log_dn_bat',    # batter day/night time-slot split
    'rate_x_vuln',   # interaction: log_rate × log_vuln
    'rate_x_park',   # interaction: log_rate × log_park
]

ADJ_FEATURE_NAMES = [
    'log_wx',        # weather factor (temp + wind)
    'log_st',        # spring training factor
    'log_pit_rf',    # pitcher recent 30-day form
    'log_bvp',       # batter vs pitcher career factor
    'log_wx_hist',   # batter weather-historical split
    'log_form',      # batter recent 30-day form
    'log_zone_match',# pitcher zone tendency vs batter zone SLG
    'lineup_pos',    # normalised batting order position (0=leadoff, 1=9th)
]

ALL_FEATURE_NAMES = CORE_FEATURE_NAMES + ADJ_FEATURE_NAMES

def _safe_log(x):
    """Log that never returns -inf."""
    return math.log(max(x, 1e-8))


def row_to_core_features(row):
    """Core features (12-dim) — available in both historical and live data."""
    (bid, pk, away_team, home_team,
     rate, raw_vuln, park, platoon,
     brl_ratio, ev_ratio, hh_f,
     bt_f, norm_rv,
     wx_f, st_f,
     r_pa, r_hr, season_hr, season_pa,
     pit_rf, ha_f, raw_bvp,
     name, wx_hist_f, dn_bat_f, zone_f) = row

    log_rate = _safe_log(rate)
    log_vuln = _safe_log(raw_vuln)
    log_park = _safe_log(park)

    return np.array([
        log_rate,
        log_vuln,
        log_park,
        _safe_log(platoon),
        _safe_log(brl_ratio * ev_ratio * hh_f),   # combined power
        _safe_log(bt_f),
        norm_rv if norm_rv is not None else 0.0,
        1.0 if norm_rv is not None else 0.0,
        _safe_log(ha_f),
        _safe_log(dn_bat_f),
        log_rate * log_vuln,    # interaction
        log_rate * log_park,    # interaction
    ], dtype=np.float32)


def row_to_adj_features(row):
    """Adjustment features (8-dim) — only populated for live 2026 data."""
    (bid, pk, away_team, home_team,
     rate, raw_vuln, park, platoon,
     brl_ratio, ev_ratio, hh_f,
     bt_f, norm_rv,
     wx_f, st_f,
     r_pa, r_hr, season_hr, season_pa,
     pit_rf, ha_f, raw_bvp,
     name, wx_hist_f, dn_bat_f, zone_f) = row

    form_log = 0.0
    if r_pa >= 15 and season_hr > 0 and season_pa > 0:
        sr = season_hr / season_pa
        rr = (r_hr / r_pa) if r_pa > 0 else 0.0
        form_log = _safe_log(max(rr / max(sr, 1e-8), 0.01))

    return np.array([
        _safe_log(wx_f),
        _safe_log(st_f),
        _safe_log(pit_rf),
        _safe_log(raw_bvp),
        _safe_log(wx_hist_f),
        form_log,
        _safe_log(zone_f),
        0.5,             # lineup_pos placeholder — overwritten in section 3a
    ], dtype=np.float32)


def row_to_all_features(row):
    """Full feature vector (20-dim) — for 2026 data where all features populated."""
    return np.concatenate([row_to_core_features(row), row_to_adj_features(row)])


# ── historical training-data pipeline ──────────────────────────────────────────

def fetch_season_schedule(year):
    """Fetch all completed regular-season game metadata for *year*."""
    games = []
    for month in range(3, 11):
        start = f'{year}-{month:02d}-01'
        end   = f'{year}-{month + 1:02d}-01' if month < 10 else f'{year}-11-01'
        try:
            r = session.get('https://statsapi.mlb.com/api/v1/schedule',
                params={'sportId': 1, 'startDate': start, 'endDate': end,
                        'gameType': 'R', 'hydrate': 'probablePitcher,team'},
                timeout=30)
            r.raise_for_status()
            for de in r.json().get('dates', []):
                for g in de.get('games', []):
                    if g.get('status', {}).get('abstractGameState') != 'Final':
                        continue
                    home = g['teams']['home']; away = g['teams']['away']
                    games.append({
                        'game_pk':         g['gamePk'],
                        'game_date':       g.get('gameDate', ''),
                        'home_team':       home['team']['name'],
                        'away_team':       away['team']['name'],
                        'home_pitcher_id': home.get('probablePitcher', {}).get('id'),
                        'away_pitcher_id': away.get('probablePitcher', {}).get('id'),
                    })
        except Exception as e:
            print(f'    Schedule fetch error ({year}-{month:02d}): {e}')
        time.sleep(0.2)
    return games


def fetch_season_boxscores(game_pks, max_workers=10):
    """Return {game_pk: {'home': [{batter_id,name,pa,hr}], 'away': [...]}}."""
    results = {}
    def _fetch_one(pk):
        try:
            r = session.get(
                f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live', timeout=15)
            r.raise_for_status()
            teams = r.json().get('liveData', {}).get('boxscore', {}).get('teams', {})
            out = {}
            for side in ('home', 'away'):
                batters = []
                for pd_data in teams.get(side, {}).get('players', {}).values():
                    bat = pd_data.get('stats', {}).get('batting', {})
                    pa  = bat.get('plateAppearances', 0) or bat.get('atBats', 0)
                    if pa > 0:
                        batters.append({
                            'batter_id': pd_data['person']['id'],
                            'name':      pd_data['person']['fullName'],
                            'pa': pa, 'hr': bat.get('homeRuns', 0),
                        })
                out[side] = batters
            return pk, out
        except Exception:
            return pk, {'home': [], 'away': []}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_one, pk): pk for pk in game_pks}
        done = 0
        for f in as_completed(futs):
            pk, data = f.result()
            results[pk] = data
            done += 1
            if done % 500 == 0:
                print(f'      {done}/{len(game_pks)} boxscores')
    return results


def build_training_data(year):
    """Build (X, y, meta) training set for one full season.  Cached."""
    cache_key = f'train_core_v2_{year}'
    cached = load_cache(cache_key)
    if cached is not None:
        n = len(cached['y'])
        hr = int(cached['y'].sum())
        print(f'  {year}: loaded from cache — {n} rows, {hr} HRs ({hr/n:.1%})')
        return cached

    print(f'\n  Building training data for {year}...')

    # 1 ── schedule
    print(f'    Fetching {year} schedule...')
    games = fetch_season_schedule(year)
    print(f'    {len(games)} final regular-season games')

    # 2 ── boxscores
    box_cache_key = f'boxscores_{year}'
    boxscores = load_cache(box_cache_key)
    if boxscores is None:
        print(f'    Fetching boxscores...')
        boxscores = fetch_season_boxscores([g['game_pk'] for g in games])
        save_cache(box_cache_key, boxscores)
    print(f'    {len(boxscores)} boxscores')

    # 3 ── collect all player IDs
    all_bat = set(); all_pit = set()
    for g in games:
        if g['home_pitcher_id']: all_pit.add(g['home_pitcher_id'])
        if g['away_pitcher_id']: all_pit.add(g['away_pitcher_id'])
    for box in boxscores.values():
        for side in ('home', 'away'):
            for b in box.get(side, []):
                all_bat.add(b['batter_id'])
    print(f'    {len(all_bat)} batters, {len(all_pit)} pitchers')

    # 4 ── season-level data  (end-of-year stats — slight lookahead, acceptable)
    stat_cache_key = f'train_stats_{year}'
    stat_cache = load_cache(stat_cache_key)
    if stat_cache is None:
        print(f'    Fetching season batting stats...')
        raw_bat = fetch_raw_hitting(all_bat, year)
        print(f'    Fetching season pitching stats...')
        pit_st  = fetch_stats_bulk(all_pit, 'pitching', year)
        print(f'    Fetching hitting splits...')
        splt    = fetch_hitting_splits(all_bat, year)
        print(f'    Fetching Savant power data...')
        brrl    = fetch_statcast_power(year)
        print(f'    Fetching bat tracking...')
        bt      = fetch_bat_tracking(year) if year >= 2024 else {}
        print(f'    Fetching batter arsenal...')
        b_ars   = fetch_pitch_arsenal('batter', year)
        print(f'    Fetching pitcher arsenal...')
        p_ars   = fetch_pitch_arsenal('pitcher', year)
        stat_cache = {'raw_bat': raw_bat, 'pit_st': pit_st, 'splits': splt,
                      'barrel': brrl, 'bat_tracking': bt,
                      'b_arsenal': b_ars, 'p_arsenal': p_ars}
        save_cache(stat_cache_key, stat_cache)
    else:
        print(f'    Season stats loaded from cache')
    raw_bat = stat_cache['raw_bat']
    pit_st  = stat_cache['pit_st']
    splt    = stat_cache['splits']
    brrl    = stat_cache['barrel']
    bt      = stat_cache['bat_tracking']
    b_ars   = stat_cache['b_arsenal']
    p_ars   = stat_cache['p_arsenal']

    # Build scorer-compatible batter_stats dict
    batter_stats = {}
    for pid, data in raw_bat.items():
        batter_stats[pid] = {
            'name':       data['name'],
            'bat_side':   data['bat_side'],
            'pitch_hand': data.get('pitch_hand', 'R'),
            'stat': {'homeRuns': data['hr'], 'plateAppearances': data['pa']},
        }

    ptbl = build_park_table({2025: 1.0, 2026: 0.0})    # use 2025 PFs as proxy
    empty = {}

    # 5 ── iterate games → rows
    game_map = {g['game_pk']: g for g in games}
    X_rows, y_rows, meta_rows = [], [], []
    for game_pk, box in boxscores.items():
        g = game_map.get(game_pk)
        if g is None:
            continue
        ts = get_game_time_slot(g.get('game_date', ''))
        for side, opp_key, is_home in [
            ('away', 'home_pitcher_id', False),
            ('home', 'away_pitcher_id', True),
        ]:
            opp_pid = g.get(opp_key)
            for batter in box.get(side, []):
                bid = batter['batter_id']
                row = _precompute_row(
                    bid, opp_pid, g['home_team'], game_pk,
                    ts, is_home, g['away_team'],
                    batter_stats, pit_st, brrl, ptbl, bt,
                    b_ars, p_ars, splt,
                    empty, empty, empty,   # recent_bat, pitcher_recent, bvp_career
                    empty, empty, empty,   # weather, spring, weather_splits
                )
                if row is not None:
                    X_rows.append(row_to_core_features(row))
                    y_rows.append(1 if batter['hr'] > 0 else 0)
                    meta_rows.append({'batter_id': bid, 'game_pk': game_pk,
                                      'name': batter['name'],
                                      'home_team': g['home_team'],
                                      'away_team': g['away_team']})

    X = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0, len(CORE_FEATURE_NAMES)), dtype=np.float32)
    y = np.array(y_rows, dtype=np.int8)
    result = {'X': X, 'y': y, 'meta': meta_rows}
    save_cache(cache_key, result)
    hr = int(y.sum())
    print(f'    Built {len(y)} rows, {hr} HRs ({hr/len(y):.1%})')
    return result


# ── main ───────────────────────────────────────────────────────────────────────

# All completed regular-season dates since opening day 2026-03-27.
# Add new dates here as games complete — the rest of the script handles them automatically.
TEST_DATES = [
    '2026-03-27','2026-03-28','2026-03-29','2026-03-30',
    '2026-03-31','2026-04-01','2026-04-02','2026-04-03',
    '2026-04-04','2026-04-05','2026-04-06','2026-04-07',
    '2026-04-08','2026-04-09','2026-04-10','2026-04-11',
    '2026-04-12','2026-04-13','2026-04-14','2026-04-15',
    '2026-04-16','2026-04-17',
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

# ── 2c. Zone data (fetched once for all batters and pitchers) ─────────────────
all_pitcher_ids = set()
for dc in date_cache.values():
    for g in dc['games']:
        for k in ('home_pitcher_id', 'away_pitcher_id'):
            if g.get(k):
                all_pitcher_ids.add(g[k])

print(f'\n  Batter zone data (all {len(all_batter_ids)} batters)...')
batter_zones_global = load_cache('batter_zones_2025')
if batter_zones_global is None:
    batter_zones_global = fetch_batter_zones(all_batter_ids)
    save_cache('batter_zones_2025', batter_zones_global)
    print(f'  Fetched & cached: {len(batter_zones_global)} batters with zone data')
else:
    missing = all_batter_ids - set(batter_zones_global.keys())
    if missing:
        print(f'  {len(missing)} new batters -- fetching zones...')
        new_zones = fetch_batter_zones(missing)
        batter_zones_global.update(new_zones)
        save_cache('batter_zones_2025', batter_zones_global)
        print(f'  Updated cache: {len(batter_zones_global)} batters total')
    else:
        print(f'  From cache: {len(batter_zones_global)} batters')

print(f'  Pitcher zone data (all {len(all_pitcher_ids)} pitchers)...')
pitcher_zones_global = load_cache('pitcher_zones_2025')
if pitcher_zones_global is None:
    pitcher_zones_global = fetch_pitcher_zones(all_pitcher_ids)
    save_cache('pitcher_zones_2025', pitcher_zones_global)
    print(f'  Fetched & cached: {len(pitcher_zones_global)} pitchers with zone data')
else:
    missing = all_pitcher_ids - set(pitcher_zones_global.keys())
    if missing:
        print(f'  {len(missing)} new pitchers -- fetching zones...')
        new_zones = fetch_pitcher_zones(missing)
        pitcher_zones_global.update(new_zones)
        save_cache('pitcher_zones_2025', pitcher_zones_global)
        print(f'  Updated cache: {len(pitcher_zones_global)} pitchers total')
    else:
        print(f'  From cache: {len(pitcher_zones_global)} pitchers')

# ── 3. Feature extraction + model training ─────────────────────────────────────
print()
print('='*68)
print('  FEATURE EXTRACTION + LOGISTIC REGRESSION')
print('='*68)
print()

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

# Build 2026 starters list
print('  Building 2026 starters list from starting lineups...')
starters_2026 = set()
for date_str in TEST_DATES:
    dc = date_cache[date_str]
    for g in dc['games']:
        hi = g.get('home_lineup_ids', [])
        ai = g.get('away_lineup_ids', [])
        if hi: starters_2026.update(hi)
        if ai: starters_2026.update(ai)
print(f'  {len(starters_2026)} unique starters found across {len(TEST_DATES)} dates')

# Pre-compute row tuples (same as before — reused for feature extraction)
print('  Pre-computing invariant row factors (starters only)...')
precomputed_rows = {}
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
            lineup = info[side]
            for lineup_idx, bid in enumerate(lineup):
                if bid not in starters_2026:
                    continue
                row = _precompute_row(
                    bid, opp_id, info['home_team'], pk,
                    info['time_slot'], is_home, info['away_team'],
                    batter_stats, pit_blended, barrel, park_tbl, bat_t,
                    bvp, pars, splits_2025_global,
                    dc['recent_bat'], dc['pitcher_recent'], dc['bvp_career'],
                    dc['weather'], spring, wx_splits,
                    batter_zones_global, pitcher_zones_global,
                )
                if row is not None:
                    lpos = lineup_idx / 8.0 if len(lineup) > 1 else 0.5
                    rows.append((row, lpos))
    precomputed_rows[date_str] = rows
    print(f'    {date_str}: {len(rows)} eligible rows')

# ── 3a. Build 2026 validation feature matrices ────────────────────────────────
print('\n  Building 2026 feature matrices...')
val_core_rows = []
val_adj_rows  = []
val_y_rows    = []
val_meta      = []        # [{batter_id, game_pk, name, home_team, away_team, date}]
val_date_idx  = {}        # {date_str: [row indices]}
val_game_idx  = {}        # {date_str: {game_pk: [row indices]}}
val_eligible  = []        # all batters eligible (prev-day HR skip removed)

for date_str in TEST_DATES:
    dc = date_cache[date_str]
    actual   = dc['actual']
    start_i  = len(val_y_rows)
    game_map = {}
    for row, lpos in precomputed_rows[date_str]:
        bid = row[0]; pk = row[1]
        core_feat = row_to_core_features(row)
        adj_feat  = row_to_adj_features(row)
        adj_feat[-1] = lpos                       # overwrite lineup_pos placeholder
        val_core_rows.append(core_feat)
        val_adj_rows.append(adj_feat)
        val_y_rows.append(1 if actual.get(bid, {}).get('hrs', 0) > 0 else 0)
        val_meta.append({
            'batter_id': bid, 'game_pk': pk, 'name': row[22],
            'home_team': row[3], 'away_team': row[2], 'date': date_str,
        })
        val_eligible.append(True)                  # no SKIP filter
        game_map.setdefault(pk, []).append(len(val_y_rows) - 1)
    val_date_idx[date_str] = list(range(start_i, len(val_y_rows)))
    val_game_idx[date_str] = game_map

val_core_X = np.vstack(val_core_rows).astype(np.float32)
val_adj_X  = np.vstack(val_adj_rows).astype(np.float32)
val_all_X  = np.hstack([val_core_X, val_adj_X])           # 18-dim for Mode C
val_y      = np.array(val_y_rows, dtype=np.int8)
val_elig   = np.array(val_eligible, dtype=np.bool_)
n_val      = len(val_y)
n_hr       = int(val_y.sum())
print(f'  2026 validation: {n_val} rows, {n_hr} HRs ({n_hr/n_val:.1%})')
print(f'  Core features: {val_core_X.shape[1]}, Adj features: {val_adj_X.shape[1]}, '
      f'All features: {val_all_X.shape[1]}')

# ── 3b. Optionally build historical training data ─────────────────────────────
train_X = None
train_y = None

if USE_HISTORICAL_TRAINING:
    print('\n  Loading historical training data...')
    for year in TRAIN_YEARS:
        data = build_training_data(year)
        if data['X'].shape[0] == 0:
            print(f'    WARNING: no rows for {year}')
            continue
        if train_X is None:
            train_X = data['X']
            train_y = data['y']
        else:
            train_X = np.vstack([train_X, data['X']])
            train_y = np.concatenate([train_y, data['y']])  # type: ignore[list-item]
    if train_X is not None and train_y is not None:
        nh = int(train_y.sum())
        print(f'  Historical training set: {len(train_y)} rows, {nh} HRs ({nh/len(train_y):.1%})')

# ── 4. Train logistic regression ──────────────────────────────────────────────

def _sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

print()
print('='*68)
print('  MODEL TRAINING')
print('='*68)

warnings.filterwarnings('ignore', category=UserWarning)   # sklearn convergence

# ── Mode A: Historical core features + post-hoc adj offsets ───────────────────
mode_a_probs = None
mode_a_auc   = 0.0
mode_a_ll    = 9.0
model_a  = None
scaler_a = None

if train_X is not None and train_y is not None and len(train_y) > 100:
    print(f'\n  [Mode A] TRAIN on {"+".join(str(y) for y in TRAIN_YEARS)} '
          f'core features ({train_X.shape[1]}-dim) -> +adj offsets on 2026')
    scaler_a = StandardScaler()
    X_tr = scaler_a.fit_transform(train_X)
    X_vl_core = scaler_a.transform(val_core_X)

    model_a = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 20).tolist(), cv=5, penalty='l2',
        scoring='neg_log_loss', max_iter=2000, random_state=42,
    )
    model_a.fit(X_tr, train_y)
    print(f'    Best C = {model_a.C_[0]:.4f}')

    # Base logit from core model + adjustment features as logit offsets
    base_logit = model_a.decision_function(X_vl_core)
    adj_offset = val_adj_X.sum(axis=1)
    mode_a_probs = _sigmoid(base_logit + adj_offset)

    mode_a_ll  = log_loss(val_y, mode_a_probs)
    mode_a_auc = roc_auc_score(val_y, mode_a_probs)
    base_probs = model_a.predict_proba(X_vl_core)[:, 1]
    base_auc   = roc_auc_score(val_y, base_probs)
    print(f'    Core-only AUC: {base_auc:.4f}  -> +adj AUC: {mode_a_auc:.4f}  '
          f'LL: {mode_a_ll:.4f}')

# ── Mode C: 2026-only leave-one-date-out CV with ALL features (LogReg) ────────
print(f'\n  [Mode C] Leave-one-date-out CV on 2026 — '
      f'LogReg, ALL {val_all_X.shape[1]} features')
cv_probs_c = np.zeros(n_val, dtype=np.float64)
for date_str in TEST_DATES:
    test_idx  = np.array(val_date_idx[date_str])
    train_idx = np.array([i for i in range(n_val) if i not in set(test_idx)])
    if len(train_idx) < 50 or val_y[train_idx].sum() < 2:
        cv_probs_c[test_idx] = n_hr / n_val
        continue
    sc = StandardScaler()
    Xtr = sc.fit_transform(val_all_X[train_idx])
    Xte = sc.transform(val_all_X[test_idx])
    m = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 15).tolist(), cv=3, penalty='l2',
        scoring='neg_log_loss', max_iter=2000, random_state=42,
    )
    m.fit(Xtr, val_y[train_idx])
    cv_probs_c[test_idx] = m.predict_proba(Xte)[:, 1]

mode_c_ll  = log_loss(val_y, cv_probs_c)
mode_c_auc = roc_auc_score(val_y, cv_probs_c) if val_y.sum() > 0 else 0.0
print(f'    CV AUC: {mode_c_auc:.4f}  LL: {mode_c_ll:.4f}')

# Fit final Mode C model on ALL 2026 data (for coefficient reporting + picks)
scaler_c = StandardScaler()
X_all_scaled = scaler_c.fit_transform(val_all_X)
model_c = LogisticRegressionCV(
    Cs=np.logspace(-3, 2, 15).tolist(), cv=5, penalty='l2',
    scoring='neg_log_loss', max_iter=2000, random_state=42,
)
model_c.fit(X_all_scaled, val_y)
print(f'    Final model C = {model_c.C_[0]:.4f}')
mode_c_full_probs = model_c.predict_proba(X_all_scaled)[:, 1]

# ── Mode D: 2026-only leave-one-date-out CV with GBM ─────────────────────────
print(f'\n  [Mode D] Leave-one-date-out CV on 2026 — '
      f'GBM, ALL {val_all_X.shape[1]} features')
cv_probs_d = np.zeros(n_val, dtype=np.float64)
for date_str in TEST_DATES:
    test_idx  = np.array(val_date_idx[date_str])
    train_idx = np.array([i for i in range(n_val) if i not in set(test_idx)])
    if len(train_idx) < 50 or val_y[train_idx].sum() < 2:
        cv_probs_d[test_idx] = n_hr / n_val
        continue
    m_gbm = HistGradientBoostingClassifier(
        max_iter=120, max_leaf_nodes=8, max_depth=3,
        learning_rate=0.02, min_samples_leaf=150,
        l2_regularization=5.0,
        class_weight='balanced', random_state=42,
    )
    m_gbm.fit(val_all_X[train_idx], val_y[train_idx])
    cv_probs_d[test_idx] = m_gbm.predict_proba(val_all_X[test_idx])[:, 1]

mode_d_ll  = log_loss(val_y, cv_probs_d)
mode_d_auc = roc_auc_score(val_y, cv_probs_d) if val_y.sum() > 0 else 0.0
print(f'    CV AUC: {mode_d_auc:.4f}  LL: {mode_d_ll:.4f}')

# Fit final GBM on ALL 2026 data for pick selection
model_d = HistGradientBoostingClassifier(
    max_iter=120, max_leaf_nodes=8, max_depth=3,
    learning_rate=0.02, min_samples_leaf=150,
    l2_regularization=5.0,
    class_weight='balanced', random_state=42,
)
model_d.fit(val_all_X, val_y)
mode_d_full_probs = model_d.predict_proba(val_all_X)[:, 1]

# ── Mode E: Ensemble (weighted average of LogReg + GBM CV probs) ──────────────
ENSEMBLE_LR_WEIGHT = 0.75  # favor the better-calibrated LogReg
ENSEMBLE_GBM_WEIGHT = 1.0 - ENSEMBLE_LR_WEIGHT

# Search for best ensemble weight
best_ew_auc = 0.0
best_ew = 0.75
for ew in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]:
    _ep = ew * cv_probs_c + (1 - ew) * cv_probs_d
    _ea = roc_auc_score(val_y, _ep) if val_y.sum() > 0 else 0.0
    if _ea > best_ew_auc:
        best_ew_auc = _ea
        best_ew = ew
ENSEMBLE_LR_WEIGHT = best_ew
ENSEMBLE_GBM_WEIGHT = 1.0 - best_ew

cv_probs_e = ENSEMBLE_LR_WEIGHT * cv_probs_c + ENSEMBLE_GBM_WEIGHT * cv_probs_d
mode_e_ll  = log_loss(val_y, cv_probs_e)
mode_e_auc = roc_auc_score(val_y, cv_probs_e) if val_y.sum() > 0 else 0.0
mode_e_full_probs = ENSEMBLE_LR_WEIGHT * mode_c_full_probs + ENSEMBLE_GBM_WEIGHT * mode_d_full_probs
print(f'\n  [Mode E] Ensemble ({ENSEMBLE_LR_WEIGHT:.0%} LR + {ENSEMBLE_GBM_WEIGHT:.0%} GBM):')
print(f'    CV AUC: {mode_e_auc:.4f}  LL: {mode_e_ll:.4f}')

# ── Select best mode ─────────────────────────────────────────────────────────
print()
print(f'  MODE COMPARISON:')
if mode_a_probs is not None:
    print(f'    Mode A (hist core + adj):  AUC={mode_a_auc:.4f}  LL={mode_a_ll:.4f}')
print(f'    Mode C (LogReg 2026 CV):   AUC={mode_c_auc:.4f}  LL={mode_c_ll:.4f}')
print(f'    Mode D (GBM 2026 CV):      AUC={mode_d_auc:.4f}  LL={mode_d_ll:.4f}')
print(f'    Mode E (Ensemble CV):      AUC={mode_e_auc:.4f}  LL={mode_e_ll:.4f}')

# Pick mode with best AUC
candidates = [
    ('C', mode_c_auc, mode_c_ll, cv_probs_c, mode_c_full_probs),
    ('D', mode_d_auc, mode_d_ll, cv_probs_d, mode_d_full_probs),
    ('E', mode_e_auc, mode_e_ll, cv_probs_e, mode_e_full_probs),
]
if mode_a_probs is not None:
    candidates.append(('A', mode_a_auc, mode_a_ll, mode_a_probs, mode_a_probs))
candidates.sort(key=lambda x: -x[1])  # best AUC first
best_mode, _, _, eval_probs, val_probs = candidates[0]

mode_labels = {
    'A': 'historical + adjustment',
    'C': 'LogReg 2026 all-features',
    'D': 'GBM 2026 all-features',
    'E': f'Ensemble ({ENSEMBLE_LR_WEIGHT:.0%} LR + {ENSEMBLE_GBM_WEIGHT:.0%} GBM)',
}
print(f'  >> Using Mode {best_mode} ({mode_labels[best_mode]})')

# Set model/scaler for coefficient reporting (LogReg modes) or feature importance (GBM)
if best_mode in ('A',):
    model  = model_a; scaler = scaler_a; active_features = CORE_FEATURE_NAMES
elif best_mode in ('C', 'E'):
    model  = model_c; scaler = scaler_c; active_features = ALL_FEATURE_NAMES
else:  # D
    model  = model_d; scaler = None;     active_features = ALL_FEATURE_NAMES

# ── 5. Model diagnostics ──────────────────────────────────────────────────────
print()
print('='*68)
print(f'  MODEL DIAGNOSTICS  (Mode {best_mode})')
print('='*68)

# LogReg coefficients
print(f"\n  LogReg coefficients (Mode C):")
coefs = model_c.coef_[0]
intercept = model_c.intercept_[0]
order = np.argsort(-np.abs(coefs))
print(f"  {'Feature':<16} {'Coef':>8}  {'|Coef|':>8}  Direction")
print(f"  {'-'*52}")
for i in order:
    direction = '+ HR' if coefs[i] > 0 else '- HR'
    print(f"  {ALL_FEATURE_NAMES[i]:<16} {coefs[i]:>8.4f}  {abs(coefs[i]):>8.4f}  {direction}")
print(f"  {'intercept':<16} {intercept:>8.4f}")

# GBM feature importances
try:
    print(f"\n  GBM feature importance (Mode D):")
    importances = model_d.feature_importances_
    imp_order = np.argsort(-importances)
    print(f"  {'Feature':<16} {'Importance':>10}")
    print(f"  {'-'*28}")
    for i in imp_order:
        if importances[i] > 0:
            print(f"  {ALL_FEATURE_NAMES[i]:<16} {importances[i]:>10.4f}")
except AttributeError:
    print(f"\n  (GBM feature importances not available in this sklearn version)")

# ── 6. Evaluation metrics ─────────────────────────────────────────────────────
print()
print('='*68)
print(f'  EVALUATION METRICS  (Mode {best_mode}, 2026 validation)')
print('='*68)

ll    = log_loss(val_y, eval_probs)
auc   = roc_auc_score(val_y, eval_probs) if val_y.sum() > 0 else 0.0
brier = brier_score_loss(val_y, eval_probs)
print(f'  Log-loss:     {ll:.4f}')
print(f'  AUC-ROC:      {auc:.4f}')
print(f'  Brier score:  {brier:.6f}')
print(f'  HR base rate: {val_y.mean():.3%} ({int(val_y.sum())}/{len(val_y)})')

# Calibration
print()
print('  Calibration (predicted probability bins):')
print(f"  {'Bin':<12} {'Count':>6} {'Pred%':>8} {'Actual%':>8}")
print(f"  {'-'*38}")
bins = [0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.20, 1.0]
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (eval_probs >= lo) & (eval_probs < hi)
    if mask.sum() == 0:
        continue
    pred_avg = eval_probs[mask].mean()
    act_avg  = val_y[mask].mean()
    print(f"  [{lo:.2f},{hi:.2f})  {int(mask.sum()):>6} {pred_avg:>7.2%} {act_avg:>7.2%}")

# ── 7. Pick selection grid ─────────────────────────────────────────────────────
pick_probs = val_probs

# ── 7a. Per-game grid (original approach) ─────────────────────────────────────
print()
print('='*68)
print(f'  PER-GAME PICK GRID  (skip × take, {len(TEST_DATES)} dates)')
print('='*68)

best_prec   = -1.0
best_skip   = 1
best_take   = 5
grid_rows   = []

for skip in PICK_SKIP_OPTIONS:
    for take in PICK_TAKE_OPTIONS:
        total_hits  = 0
        total_picks = 0
        for date_str in TEST_DATES:
            dc      = date_cache[date_str]
            actual  = dc['actual']
            for pk, row_indices in val_game_idx[date_str].items():
                elig = [(i, pick_probs[i]) for i in row_indices
                        if val_elig[i]]
                elig.sort(key=lambda x: -x[1])
                picks = elig[skip:skip + take]
                for idx, prob in picks:
                    bid = val_meta[idx]['batter_id']
                    if actual.get(bid, {}).get('hrs', 0) > 0:
                        total_hits += 1
                    total_picks += 1
        prec = total_hits / total_picks if total_picks > 0 else 0.0
        grid_rows.append({'skip': skip, 'take': take,
                          'hits': total_hits, 'picks': total_picks,
                          'prec': prec})
        if prec > best_prec:
            best_prec = prec
            best_skip = skip
            best_take = take

grid_rows.sort(key=lambda x: -x['prec'])
print(f"  {'Skip':>5} {'Take':>5} {'Picks':>7} {'Hits':>6} {'Prec':>7}")
print(f"  {'-'*35}")
for r in grid_rows:
    marker = ' <-- BEST' if r['skip'] == best_skip and r['take'] == best_take else ''
    print(f"  {r['skip']:>5} {r['take']:>5} {r['picks']:>7} {r['hits']:>6}"
          f" {r['prec']:>6.1%}{marker}")

# ── 7b. Global ranking grid (rank ALL batters for a date, pick top-N) ─────────
print()
print('='*68)
print(f'  GLOBAL RANKING GRID  (top-N across all games, {len(TEST_DATES)} dates)')
print('='*68)

GLOBAL_TOP_OPTIONS = [10, 15, 20, 25, 30, 40, 50]
best_glob_prec = -1.0
best_glob_top  = 20
glob_rows = []

for top_n in GLOBAL_TOP_OPTIONS:
    total_hits  = 0
    total_picks = 0
    for date_str in TEST_DATES:
        dc     = date_cache[date_str]
        actual = dc['actual']
        # Gather ALL eligible batters across all games for this date
        day_elig = []
        for pk, row_indices in val_game_idx[date_str].items():
            for i in row_indices:
                if val_elig[i]:
                    day_elig.append((i, pick_probs[i]))
        day_elig.sort(key=lambda x: -x[1])
        picks = day_elig[:top_n]
        for idx, prob in picks:
            bid = val_meta[idx]['batter_id']
            if actual.get(bid, {}).get('hrs', 0) > 0:
                total_hits += 1
            total_picks += 1
    prec = total_hits / total_picks if total_picks > 0 else 0.0
    glob_rows.append({'top_n': top_n, 'hits': total_hits,
                      'picks': total_picks, 'prec': prec})
    if prec > best_glob_prec:
        best_glob_prec = prec
        best_glob_top  = top_n

glob_rows.sort(key=lambda x: -x['prec'])
print(f"  {'Top-N':>6} {'Picks':>7} {'Hits':>6} {'Prec':>7}")
print(f"  {'-'*30}")
for r in glob_rows:
    marker = ' <-- BEST' if r['top_n'] == best_glob_top else ''
    print(f"  {r['top_n']:>6} {r['picks']:>7} {r['hits']:>6} {r['prec']:>6.1%}{marker}")

# Decide which pick strategy is better
use_global = best_glob_prec > best_prec
if use_global:
    print(f'\n  Global top-{best_glob_top} ({best_glob_prec:.1%}) beats '
          f'per-game skip={best_skip}/take={best_take} ({best_prec:.1%})')
else:
    print(f'\n  Per-game skip={best_skip}/take={best_take} ({best_prec:.1%}) beats '
          f'global top-{best_glob_top} ({best_glob_prec:.1%})')

# ── 8. Per-date results ───────────────────────────────────────────────────────
print()
print('='*68)
if use_global:
    print(f'  PER-DATE RESULTS  (global top-{best_glob_top})')
else:
    print(f'  PER-DATE RESULTS  (per-game skip={best_skip}, take={best_take})')
print('='*68)
print(f"  {'Date':<12} {'Picks':>6} {'Hits':>6} {'Prec':>7}  {'Actual HRs':>10}  {'Log-loss':>10}")
print(f"  {'-'*60}")

best_date_picks = {}
for date_str in TEST_DATES:
    dc      = date_cache[date_str]
    actual  = dc['actual']
    n_act   = len(actual)

    day_picks = []
    if use_global:
        # Global ranking for this date
        day_elig = []
        for pk, row_indices in val_game_idx[date_str].items():
            for i in row_indices:
                if val_elig[i]:
                    day_elig.append((i, pick_probs[i]))
        day_elig.sort(key=lambda x: -x[1])
        for idx, prob in day_elig[:best_glob_top]:
            m = val_meta[idx]
            hit = actual.get(m['batter_id'], {}).get('hrs', 0) > 0
            day_picks.append({
                'batter_id': m['batter_id'], 'name': m['name'],
                'hr_prob': float(round(float(prob), 4)), 'game_pk': m['game_pk'],
                'home_team': m['home_team'], 'away_team': m['away_team'],
                'hit': hit,
            })
    else:
        # Per-game pick selection
        for pk, row_indices in val_game_idx[date_str].items():
            elig = [(i, pick_probs[i]) for i in row_indices if val_elig[i]]
            elig.sort(key=lambda x: -x[1])
            for idx, prob in elig[best_skip:best_skip + best_take]:
                m = val_meta[idx]
                hit = actual.get(m['batter_id'], {}).get('hrs', 0) > 0
                day_picks.append({
                    'batter_id': m['batter_id'], 'name': m['name'],
                    'hr_prob': float(round(float(prob), 4)), 'game_pk': m['game_pk'],
                    'home_team': m['home_team'], 'away_team': m['away_team'],
                    'hit': hit,
                })

    best_date_picks[date_str] = day_picks
    n_hits = sum(1 for p in day_picks if p['hit'])
    prec   = n_hits / len(day_picks) if day_picks else 0.0

    didx = val_date_idx[date_str]
    d_ll = log_loss(val_y[didx], eval_probs[didx]) if didx else 0.0

    print(f"  {date_str:<12} {len(day_picks):>6} {n_hits:>6} {prec:>6.0%}  {n_act:>10}  {d_ll:>10.4f}")

# ── 9. Per-game picks for last date ───────────────────────────────────────────
print()
last_date  = TEST_DATES[-1]
last_picks = best_date_picks[last_date]
pick_label = f'global top-{best_glob_top}' if use_global else f'skip={best_skip}, take={best_take}'
print(f'  Per-game picks — {last_date} ({pick_label}):')
print(f"  {'Matchup':<34} {'Player':<24} {'P(HR)':>6}  Hit?")
print(f"  {'-'*72}")
for p in sorted(last_picks, key=lambda x: (x['game_pk'], -x['hr_prob'])):
    matchup = f"{p['away_team']} @ {p['home_team']}"
    print(f"  {matchup:<34} {p['name']:<24} {p['hr_prob']:>5.1%}  "
          f"{'YES' if p['hit'] else '---'}")

# ── 10. Save model artifacts ──────────────────────────────────────────────────
model_artifact = {
    'model_logreg':  model_c,
    'model_gbm':     model_d,
    'scaler':        scaler_c,
    'feature_names': ALL_FEATURE_NAMES,
    'best_mode':     best_mode,
    'pick_strategy': 'global' if use_global else 'per_game',
    'best_skip':     best_skip,
    'best_take':     best_take,
    'best_glob_top': best_glob_top,
    'train_years':   TRAIN_YEARS if train_X is not None else [],
    'val_dates':     TEST_DATES,
    'metrics':       {'log_loss': ll, 'auc': auc, 'brier': brier},
    'ensemble_lr_weight': ENSEMBLE_LR_WEIGHT if best_mode == 'E' else 1.0,
}
save_cache('hr_logreg_model', model_artifact)
print(f'\n  Model saved to hr_logreg_model.pkl  (Mode {best_mode}, '
      f'{"global" if use_global else "per-game"} picks)')
