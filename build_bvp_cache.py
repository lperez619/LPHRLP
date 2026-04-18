"""
Build bvp_career.pkl from MLB Stats API career vsPlayer splits.
Expected format: {batter_id: {pitcher_id: {'pa': N, 'hr': N}}}

Collects all batter IDs from the cached date_*.pkl files, then
fetches career vsPlayer splits for each batter.
"""
import os, pickle, time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

# 1. Collect all batter IDs from cached date files
all_batters = set()
date_files = sorted(f for f in os.listdir(SCRIPT_DIR) if f.startswith('date_') and f.endswith('.pkl'))
print(f'Found {len(date_files)} date cache files')

for fname in date_files:
    path = os.path.join(SCRIPT_DIR, fname)
    with open(path, 'rb') as f:
        dc = pickle.load(f)
    if isinstance(dc, dict) and 'all_bat' in dc:
        all_batters |= dc['all_bat']

print(f'Total unique batters: {len(all_batters)}')

# 2. Fetch career vsPlayer splits for each batter
def fetch_bvp_for_batter(bid):
    """Return {pitcher_id: {'pa': N, 'hr': N}} for this batter's career."""
    url = f'https://statsapi.mlb.com/api/v1/people/{bid}/stats'
    params = {
        'stats': 'vsPlayer',
        'group': 'hitting',
        'gameType': 'R',  # regular season only
    }
    try:
        r = session.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        matchups = {}
        for split_group in data.get('stats', []):
            for split in split_group.get('splits', []):
                pitcher = split.get('player', {})
                pid = pitcher.get('id')
                if not pid:
                    continue
                stat = split.get('stat', {})
                pa = stat.get('plateAppearances', 0) or stat.get('atBats', 0)
                hr = stat.get('homeRuns', 0)
                if pa > 0:
                    matchups[pid] = {'pa': pa, 'hr': hr}
        return bid, matchups
    except Exception as e:
        return bid, {}

bvp_career = {}
done = 0
total = len(all_batters)
batch_size = 10  # concurrent requests

batter_list = list(all_batters)
for i in range(0, total, batch_size):
    batch = batter_list[i:i+batch_size]
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = {pool.submit(fetch_bvp_for_batter, bid): bid for bid in batch}
        for fut in as_completed(futures):
            bid, matchups = fut.result()
            if matchups:
                bvp_career[bid] = matchups
            done += 1
    if done % 50 == 0 or done == total:
        print(f'  {done}/{total} batters fetched ... {len(bvp_career)} with matchup data')
    time.sleep(0.3)  # be polite to the API

# 3. Save
out_path = os.path.join(SCRIPT_DIR, 'bvp_career.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(bvp_career, f, protocol=pickle.HIGHEST_PROTOCOL)

n_pairs = sum(len(v) for v in bvp_career.values())
print(f'\nDone! Saved bvp_career.pkl')
print(f'  {len(bvp_career)} batters, {n_pairs:,} batter-pitcher pairs')
