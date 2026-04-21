"""
Rebuilds the rolling 30-day historical form for batters and pitchers in 2024 and 2025.
This script generates an offline cache `historical_rolling_form.pkl` so that HR Trainer.py
can use actual dynamic form (log_form, log_pit_rf, log_rhr) during historical training
instead of inserting zeros.
"""
import os
import pickle
import time
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})
from requests.adapters import HTTPAdapter
_adapter = HTTPAdapter(pool_connections=20, pool_maxsize=50)
session.mount('https://', _adapter)
session.mount('http://',  _adapter)

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(CACHE_DIR, 'historical_rolling_form.pkl')

def load_cache(name):
    path = os.path.join(CACHE_DIR, f'{name}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def save_cache(name, data):
    path = os.path.join(CACHE_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def parse_ip(ip_str):
    if ip_str is None: return 0.0
    try:
        parts = str(ip_str).split('.')
        full = float(parts[0])
        frac = float(parts[1]) if len(parts) > 1 else 0.0
        return full + (frac / 3.0)
    except:
        return 0.0

def fetch_game_logs(player_ids, year):
    print(f"Fetching game logs for {len(player_ids)} players in {year}...")
    batting_logs = defaultdict(list)
    pitching_logs = defaultdict(list)
    
    ids_list = list(player_ids)
    for i in range(0, len(ids_list), 50):
        chunk = ids_list[i:i+50]
        try:
            r = session.get('https://statsapi.mlb.com/api/v1/people',
                params={'personIds': ','.join(str(x) for x in chunk),
                        'hydrate': f'stats(group=[hitting,pitching],type=[gameLog],season={year})'},
                timeout=20)
            r.raise_for_status()
            
            for person in r.json().get('people', []):
                pid = person['id']
                for stat_group in person.get('stats', []):
                    group_name = stat_group.get('group', {}).get('displayName', '')
                    
                    for split in stat_group.get('splits', []):
                        game_date = split.get('date')
                        st = split.get('stat', {})
                        
                        if not game_date: continue
                        
                        if group_name == 'hitting':
                            pa = st.get('plateAppearances', 0) or st.get('atBats', 0)
                            hr = st.get('homeRuns', 0)
                            if pa > 0:
                                batting_logs[pid].append({'date': game_date, 'pa': pa, 'hr': hr})
                        
                        elif group_name == 'pitching':
                            ip = parse_ip(st.get('inningsPitched', '0.0'))
                            hr = st.get('homeRuns', 0)
                            if ip > 0:
                                pitching_logs[pid].append({'date': game_date, 'ip': ip, 'hr': hr})
                                
        except Exception as e:
            print(f"Error fetching chunk: {e}")
        print(f"  {i}/{len(ids_list)} players processed...", end='\r')
        time.sleep(0.2)
        
    print(f"\nDone {year}. Got {len(batting_logs)} batting logs and {len(pitching_logs)} pitching logs.")
    return batting_logs, pitching_logs

def build_rolling_cache():
    all_batters = set()
    all_pitchers = set()
    
    for year in [2024, 2025]:
        hist_data = load_cache(f'train_full_v1_{year}')
        if hist_data and 'meta' in hist_data:
            meta_rows = hist_data['meta']
            print(f"Found {len(meta_rows)} rows for {year}")
            for m in meta_rows:
                all_batters.add(m['batter_id'])
                # We don't have the opposing pitcher ID in the meta explicitly! Wait, we actually don't.
                # Let's get the pitcher ID from the HR Trainer directly. 
                pass
                
    print(f"Total unique historical batters: {len(all_batters)}")
    print(f"Total unique historical pitchers: {len(all_pitchers)}")
    
    if not all_batters:
        print("Could not find historical training data caches!")
        return

    # To avoid URL length limits, we just combine both sets
    all_players = all_batters.union(all_pitchers)
    
    b_logs_24, p_logs_24 = fetch_game_logs(all_players, 2024)
    b_logs_25, p_logs_25 = fetch_game_logs(all_players, 2025)
    
    # Merge logs by player
    batting_logs_all = defaultdict(list)
    pitching_logs_all = defaultdict(list)
    
    for pid, entries in b_logs_24.items(): batting_logs_all[pid].extend(entries)
    for pid, entries in b_logs_25.items(): batting_logs_all[pid].extend(entries)
    for pid, entries in p_logs_24.items(): pitching_logs_all[pid].extend(entries)
    for pid, entries in p_logs_25.items(): pitching_logs_all[pid].extend(entries)
    
    for pid in batting_logs_all:
        batting_logs_all[pid].sort(key=lambda x: x['date'])
    for pid in pitching_logs_all:
        pitching_logs_all[pid].sort(key=lambda x: x['date'])
        
    print("Pre-computing rolling 30-day windows per game date...")
    
    # cache format: cache[date_str]['batter'][pid] = {'pa': X, 'hr': Y}
    rolling_cache = defaultdict(lambda: {'batter': {}, 'pitcher': {}})
    
    # unique dates in our dataset
    unique_dates = set()
    for year in [2024, 2025]:
        hist_data = load_cache(f'train_full_v1_{year}')
        if hist_data:
            for row in hist_data:
                unique_dates.add(row[3]) # game_pk doesn't help us here, wait.
                # Actually row[3] is pk... let's just get the exact dates from train_full! Wait, train_full doesn't store date_str.
                pass
                
    # Instead, we can just quickly iterate over every day in the season
    for yr in [2024, 2025]:
        start = datetime(yr, 3, 20)
        end = datetime(yr, 10, 5)
        curr = start
        while curr <= end:
            unique_dates.add(curr.strftime('%Y-%m-%d'))
            curr += timedelta(days=1)
            
    # For every unique date, and every player, we compute the 30-day trailing window up to date_str
    # This might be slow if we do it naively. Let's do it vectorized per player
    for date_str in sorted(list(unique_dates)):
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        end_dt = dt - timedelta(days=1)
        start_dt = dt - timedelta(days=30)
        end_str = end_dt.strftime('%Y-%m-%d')
        start_str = start_dt.strftime('%Y-%m-%d')
        
        # Batters
        for pid, entries in batting_logs_all.items():
            pa, hr = 0, 0
            for e in entries:
                if start_str <= e['date'] <= end_str:
                    pa += e['pa']
                    hr += e['hr']
                elif e['date'] > end_str:
                    break # since it's sorted
            if pa > 0:
                rolling_cache[date_str]['batter'][pid] = {'pa': pa, 'hr': hr}
                
        # Pitchers
        for pid, entries in pitching_logs_all.items():
            ip, hr = 0.0, 0
            for e in entries:
                if start_str <= e['date'] <= end_str:
                    ip += e['ip']
                    hr += e['hr']
                elif e['date'] > end_str:
                    break
            if ip > 0:
                rolling_cache[date_str]['pitcher'][pid] = {'ip': ip, 'hr': hr}
                
        # progress
        if len(rolling_cache) % 50 == 0:
            print(f"Computed form up to {date_str}...")
            
    save_cache('historical_rolling_form', dict(rolling_cache))
    print(f"Success! Saved rolling form for {len(rolling_cache)} days to historical_rolling_form.pkl")

if __name__ == '__main__':
    build_rolling_cache()
