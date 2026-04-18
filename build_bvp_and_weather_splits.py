"""
Build bvp_career.pkl and weather_splits.pkl from MLB play-by-play data.

Processes 2023-2026 regular-season games:
  - bvp_career:      {batter_id: {pitcher_id: {'pa': N, 'hr': N}}}
  - weather_splits:   {batter_id: {bucket: {'pa': N, 'hr': N}}}
    buckets: temp_cold (<55°F), temp_mild, temp_hot (>75°F),
             wind_out (>8 mph out), wind_calm, wind_in (>8 mph in)

Uses per-season checkpoints so restarts skip completed seasons.
"""

import os, sys, pickle, time, math, requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Full STADIUMS dict (need 'cf' for wind bucket calc)
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

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})
from requests.adapters import HTTPAdapter
_adapter = HTTPAdapter(pool_connections=20, pool_maxsize=50)
session.mount('https://', _adapter)
session.mount('http://', _adapter)

# Event types that are NOT plate appearances (baserunning / misc actions)
NON_PA_EVENTS = {
    'stolen_base_2b', 'stolen_base_3b', 'stolen_base_home',
    'caught_stealing_2b', 'caught_stealing_3b', 'caught_stealing_home',
    'pickoff_1b', 'pickoff_2b', 'pickoff_3b',
    'pickoff_caught_stealing_2b', 'pickoff_caught_stealing_3b',
    'pickoff_caught_stealing_home',
    'wild_pitch', 'passed_ball', 'balk',
    'other_advance', 'runner_double_play', 'game_advisory',
    'ejection', 'pitching_substitution', 'defensive_substitution',
    'offensive_substitution', 'defensive_switch',
}


def get_season_game_pks(season):
    """Get all completed regular-season game PKs for a season."""
    if season < 2026:
        start = f'{season}-03-20'
        end = f'{season}-10-05'
    else:
        start = '2026-03-26'
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    r = session.get('https://statsapi.mlb.com/api/v1/schedule',
                     params={'sportId': 1, 'startDate': start, 'endDate': end,
                             'gameType': 'R'}, timeout=30)
    r.raise_for_status()
    game_pks = []
    for date_entry in r.json().get('dates', []):
        for g in date_entry.get('games', []):
            if g.get('status', {}).get('abstractGameState', '') == 'Final':
                game_pks.append(g['gamePk'])
    return game_pks


def fetch_play_by_play(game_pk):
    """Fetch play-by-play and extract (batter_id, pitcher_id, is_hr) for each PA."""
    try:
        r = session.get(
            f'https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay',
            timeout=20)
        r.raise_for_status()
        plays = r.json().get('allPlays', [])
        results = []
        for play in plays:
            if not play.get('about', {}).get('isComplete', False):
                continue
            event_type = play.get('result', {}).get('eventType', '')
            if not event_type or event_type in NON_PA_EVENTS:
                continue
            batter = play.get('matchup', {}).get('batter', {}).get('id')
            pitcher = play.get('matchup', {}).get('pitcher', {}).get('id')
            if not batter or not pitcher:
                continue
            is_hr = (event_type == 'home_run')
            results.append((batter, pitcher, is_hr))
        return game_pk, results, None
    except Exception as e:
        return game_pk, [], str(e)


def process_season(season, workers=15):
    """Process one season's play-by-play. Returns (bvp_dict, game_batter_dict)."""
    ckpt_path = os.path.join(_SCRIPT_DIR, f'_pbp_ckpt_{season}.pkl')
    if os.path.exists(ckpt_path):
        print(f'  {season}: loading checkpoint...')
        with open(ckpt_path, 'rb') as f:
            return pickle.load(f)

    pks = get_season_game_pks(season)
    print(f'  {season}: {len(pks)} games to process')

    # bvp: {(batter, pitcher): {'pa': N, 'hr': N}}
    # game_batter: {(game_pk, batter): {'pa': N, 'hr': N}}
    bvp = {}
    game_batter = {}
    errors = 0
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_play_by_play, pk): pk for pk in pks}
        for future in as_completed(futures):
            game_pk, plays, err = future.result()
            if err:
                errors += 1
            for batter, pitcher, is_hr in plays:
                # BvP accumulation
                bp_key = (batter, pitcher)
                if bp_key not in bvp:
                    bvp[bp_key] = {'pa': 0, 'hr': 0}
                bvp[bp_key]['pa'] += 1
                if is_hr:
                    bvp[bp_key]['hr'] += 1

                # Per-game batter accumulation (for weather splits)
                gb_key = (game_pk, batter)
                if gb_key not in game_batter:
                    game_batter[gb_key] = {'pa': 0, 'hr': 0}
                game_batter[gb_key]['pa'] += 1
                if is_hr:
                    game_batter[gb_key]['hr'] += 1

            done += 1
            if done % 250 == 0:
                print(f'    {done}/{len(pks)} games ({errors} errors)')

    if errors:
        print(f'  {season}: {errors} games had errors')

    data = {'bvp': bvp, 'game_batter': game_batter}
    with open(ckpt_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  {season}: done — {len(bvp)} matchups, checkpoint saved')
    return data


def build_weather_splits(all_game_batter, weather_data):
    """Build weather_splits from per-game batter stats + weather data."""
    splits = {}  # {batter_id: {bucket: {'pa': N, 'hr': N}}}

    used_games = 0
    for (game_pk, batter_id), stats in all_game_batter.items():
        wx = weather_data.get(game_pk)
        if not wx:
            continue
        home_team = wx.get('home_team', '')
        stadium = STADIUMS.get(home_team, {})
        if stadium.get('indoor', True):
            continue

        used_games += 1

        # Temp bucket
        temp = wx['temp_f']
        if temp < 55:
            tbucket = 'temp_cold'
        elif temp > 75:
            tbucket = 'temp_hot'
        else:
            tbucket = 'temp_mild'

        # Wind bucket (effective wind toward CF)
        cf_dir = stadium.get('cf', 350)
        opp = (cf_dir + 180) % 360
        ang = abs(((wx['wind_dir'] - opp) % 360) - 180)
        eff_wind = wx['wind_mph'] * math.cos(math.radians(ang))
        if eff_wind > 8:
            wbucket = 'wind_out'
        elif eff_wind < -8:
            wbucket = 'wind_in'
        else:
            wbucket = 'wind_calm'

        # Add to both temp and wind buckets
        if batter_id not in splits:
            splits[batter_id] = {}
        for bucket in (tbucket, wbucket):
            if bucket not in splits[batter_id]:
                splits[batter_id][bucket] = {'pa': 0, 'hr': 0}
            splits[batter_id][bucket]['pa'] += stats['pa']
            splits[batter_id][bucket]['hr'] += stats['hr']

    print(f'  Used {used_games} game-batter rows with weather data')
    return splits


def main():
    print('=' * 60)
    print('  BUILDING BvP CAREER + WEATHER SPLITS CACHES')
    print('=' * 60)

    # ── 1. Load weather data ──
    print('\nLoading weather data...')
    weather_data = {}
    for year in [2023, 2024, 2025, 2026]:
        path = os.path.join(_SCRIPT_DIR, f'game_weather_{year}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                d = pickle.load(f)
            weather_data.update(d)
            print(f'  game_weather_{year}: {len(d)} games')
        else:
            print(f'  game_weather_{year}: NOT FOUND')
    print(f'  Total weather entries: {len(weather_data)}')

    # ── 2. Process play-by-play per season ──
    print('\nProcessing play-by-play data...')
    all_bvp = {}         # (batter, pitcher) -> {pa, hr}
    all_game_batter = {} # (game_pk, batter) -> {pa, hr}

    for season in [2023, 2024, 2025, 2026]:
        data = process_season(season)
        # Merge into global accumulators
        for key, stats in data['bvp'].items():
            if key not in all_bvp:
                all_bvp[key] = {'pa': 0, 'hr': 0}
            all_bvp[key]['pa'] += stats['pa']
            all_bvp[key]['hr'] += stats['hr']
        for key, stats in data['game_batter'].items():
            if key not in all_game_batter:
                all_game_batter[key] = {'pa': 0, 'hr': 0}
            all_game_batter[key]['pa'] += stats['pa']
            all_game_batter[key]['hr'] += stats['hr']

    # ── 3. Build BvP career cache ──
    print('\nBuilding bvp_career.pkl...')
    bvp_career = {}
    for (batter, pitcher), stats in all_bvp.items():
        if batter not in bvp_career:
            bvp_career[batter] = {}
        bvp_career[batter][pitcher] = stats

    total_matchups = sum(len(v) for v in bvp_career.values())
    total_hr = sum(s['hr'] for s in all_bvp.values())
    print(f'  {len(bvp_career)} batters, {total_matchups} matchups, {total_hr} total HR')

    out = os.path.join(_SCRIPT_DIR, 'bvp_career.pkl')
    with open(out, 'wb') as f:
        pickle.dump(bvp_career, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  Saved → bvp_career.pkl')

    # ── 4. Build weather splits cache ──
    print('\nBuilding weather_splits.pkl...')
    ws = build_weather_splits(all_game_batter, weather_data)
    print(f'  {len(ws)} batters with weather split data')

    out = os.path.join(_SCRIPT_DIR, 'weather_splits.pkl')
    with open(out, 'wb') as f:
        pickle.dump(ws, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  Saved → weather_splits.pkl')

    # ── 5. Cleanup checkpoints ──
    for season in [2023, 2024, 2025, 2026]:
        ckpt = os.path.join(_SCRIPT_DIR, f'_pbp_ckpt_{season}.pkl')
        if os.path.exists(ckpt):
            os.remove(ckpt)
    print('\nCheckpoints cleaned up. Done!')


if __name__ == '__main__':
    main()
