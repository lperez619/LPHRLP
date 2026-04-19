import requests, json

# Test zone data availability
# Pitcher zone data (via pitching group)  
pitchers = [543037, 477132, 669302, 621111, 675911]
for pid in pitchers:
    r = requests.get(
        f'https://statsapi.mlb.com/api/v1/people/{pid}/stats',
        params={'stats': 'hotColdZones', 'group': 'pitching', 'season': 2025},
        timeout=15)
    data = r.json()
    stats = data.get('stats', [])
    if stats and stats[0].get('splits'):
        print(f'Pitcher {pid}: {len(stats[0]["splits"])} zone stat types')
        for s in stats[0]['splits']:
            print(f'  - {s["stat"]["name"]}: {len(s["stat"]["zones"])} zones')
        break
    else:
        print(f'Pitcher {pid}: no zone data')

# Batter zone data
print()
batters = [660271, 592450, 665489, 545361]  # Ohtani, Judge, Guerrero, Trout
for bid in batters:
    r = requests.get(
        f'https://statsapi.mlb.com/api/v1/people/{bid}/stats',
        params={'stats': 'hotColdZones', 'group': 'hitting', 'season': 2025},
        timeout=15)
    data = r.json()
    stats = data.get('stats', [])
    if stats and stats[0].get('splits'):
        print(f'Batter {bid}: {len(stats[0]["splits"])} zone stat types')
        for s in stats[0]['splits']:
            name = s['stat']['name']
            zones = s['stat']['zones']
            vals = {z['zone']: z['value'] for z in zones}
            print(f'  - {name}: zones {sorted(vals.keys())}')
        break
    else:
        print(f'Batter {bid}: no zone data')

# Check if we can get pitcher zone data via the hitting group (allowed vs pitchers)
print('\n--- Pitcher allowed zones ---')
r = requests.get(
    f'https://statsapi.mlb.com/api/v1/people/675911/stats',
    params={'stats': 'hotColdZones', 'group': 'hitting', 'season': 2025},
    timeout=15)
data = r.json()
stats = data.get('stats', [])
print(f'Pitcher as hitter: {len(stats)} stat groups, splits={len(stats[0]["splits"]) if stats else 0}')

# Try statcast search for zone data
print('\n--- Statcast search zone data ---')
r = requests.get(
    'https://baseballsavant.mlb.com/statcast_search/csv',
    params={
        'all': 'true',
        'player_type': 'pitcher',
        'player_id': '675911',
        'type': 'details',
        'hfSea': '2025|',
        'min_results': 0,
    },
    headers={'User-Agent': 'Mozilla/5.0'},
    timeout=30)
lines = r.text.strip().split('\n')
if len(lines) > 1:
    print(f'Statcast search: {len(lines)-1} pitches')
    header = lines[0]
    # Find zone-related columns
    cols = [c.strip('"') for c in header.split(',')]
    zone_cols = [c for c in cols if 'zone' in c.lower() or 'plate_x' in c.lower() or 'plate_z' in c.lower()]
    print(f'Zone columns: {zone_cols}')
else:
    print(f'Statcast search: {r.text[:500]}')
