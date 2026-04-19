"""Diagnose today's picks vs actual HRs — deep analysis."""
import sys, os, json, pickle, math
import numpy as np
_trainer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HR Trainer.py')
_trainer_globals = {'__name__': '_hr_trainer_lib', '__file__': _trainer_path}
with open(_trainer_path, 'r', encoding='utf-8') as f:
    src = f.read()
pos = src.find('\nTEST_DATES')
exec(compile(src[:pos], _trainer_path, 'exec'), _trainer_globals)

get_actual_hrs = _trainer_globals['get_actual_hrs']
load_cache = _trainer_globals['load_cache']

DATE = '2026-04-18'
actual = get_actual_hrs(DATE)
actual_pids = set(actual.keys())
actual_names = {info.get('name', '').lower(): pid for pid, info in actual.items()}

print(f'Total HR hitters on {DATE}: {len(actual)}')
print(f'Total HRs: {sum(v.get("hrs", 0) for v in actual.values())}')

# Now re-run scoring to get full rankings
print('\n=== RE-RUNNING FULL SCORING ===')
# Import everything needed
for name in ['get_games', 'get_game_time_slot', 'get_roster', 'fetch_raw_hitting',
             'fetch_stats_bulk', 'fetch_weather', 'fetch_spring_stats',
             'fetch_statcast_power', 'fetch_bat_tracking', 'fetch_pitch_arsenal',
             'fetch_hitting_splits', 'combine_hitting', 'build_park_table',
             'blend_pitcher_stats', '_precompute_row', 'row_to_core_features',
             'row_to_adj_features', 'save_cache',
             'PITCHER_SEASON', 'SEASON_WEIGHT_CONFIGS', 'PF_BLEND_CONFIGS',
             'STADIUMS', 'MIN_PA', 'fetch_batter_zones', 'fetch_pitcher_zones']:
    globals()[name] = _trainer_globals[name]

from datetime import datetime, timedelta

artifact = load_cache('hr_logreg_model')
model_logreg = artifact.get('model_logreg', artifact.get('model'))
model_gbm = artifact.get('model_gbm')
scaler = artifact['scaler']
best_mode = artifact['best_mode']

# Load shared data
barrel = load_cache(f'barrel_{PITCHER_SEASON}')
_bc = load_cache(f'bat_tracking_{PITCHER_SEASON}')
bat_t = _bc['bat_t']
bvp = load_cache(f'arsenal_batter_{PITCHER_SEASON}')
pars = load_cache(f'arsenal_pitcher_{PITCHER_SEASON}')
spring = fetch_spring_stats()
bvp_career = load_cache('bvp_career') or {}
wx_splits = load_cache('weather_splits') or {}
splits_global = load_cache('splits_2025') or {}
batter_zones = load_cache('batter_zones_2025') or {}
pitcher_zones = load_cache('pitcher_zones_2025') or {}

games = get_games(DATE)
batter_map = {}
for g in games:
    pk = g['game_pk']
    hi = g['home_lineup_ids'] or get_roster(g['home_team_id'])
    ai = g['away_lineup_ids'] or get_roster(g['away_team_id'])
    batter_map[pk] = {
        'home': hi, 'away': ai,
        'home_team': g['home_team'], 'away_team': g['away_team'],
        'time_slot': get_game_time_slot(g['game_date']),
    }

all_bat = {bid for v in batter_map.values() for s in ('home','away') for bid in v[s]}
all_pit = {g[k] for g in games for k in ('home_pitcher_id','away_pitcher_id') if g[k]}

prev_date = (datetime.strptime(DATE, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as pool:
    fut_25 = pool.submit(fetch_raw_hitting, all_bat, 2025)
    fut_26 = pool.submit(fetch_raw_hitting, all_bat, 2026)
    fut_pt = pool.submit(fetch_stats_bulk, all_pit, 'pitching', PITCHER_SEASON)
    fut_wx = pool.submit(fetch_weather, games, DATE)
    fut_pac = pool.submit(get_actual_hrs, prev_date)
    raw_25 = fut_25.result(); raw_26 = fut_26.result()
    pit_stats = fut_pt.result(); weather = fut_wx.result()
    prev_actual = fut_pac.result()

prev_hr = set(prev_actual.keys())
sw_label, sw_weights = SEASON_WEIGHT_CONFIGS[0]
pf_label, pf_blend = PF_BLEND_CONFIGS[0]
park_tbl = build_park_table(pf_blend)
raw_by_season = {}
for yr, raw in [(2025, raw_25), (2026, raw_26)]:
    if sw_weights.get(yr, 0) > 0:
        raw_by_season[yr] = raw
batter_stats = combine_hitting(raw_by_season, sw_weights)
pit_blended = blend_pitcher_stats(pit_stats, {})

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

rows = []
for g in games:
    pk = g['game_pk']
    info = batter_map[pk]
    for side, opp_id, is_home in [('away', g['home_pitcher_id'], False),
                                   ('home', g['away_pitcher_id'], True)]:
        lineup = info[side]
        for li, bid in enumerate(lineup):
            row = _precompute_row(
                bid, opp_id, info['home_team'], pk,
                info['time_slot'], is_home, info['away_team'],
                batter_stats, pit_blended, barrel, park_tbl, bat_t,
                bvp, pars, splits_global,
                {}, {}, bvp_career, weather, spring, wx_splits,
                batter_zones, pitcher_zones,
            )
            if row is not None:
                lpos = li / 8.0 if len(lineup) > 1 else 0.5
                core = row_to_core_features(row)
                adj = row_to_adj_features(row)
                adj[-1] = lpos
                rows.append({
                    'bid': bid, 'pk': pk, 'name': row[22],
                    'core': core, 'adj': adj,
                    'home_team': row[3], 'away_team': row[2],
                    'lineup_pos': li + 1,
                    'eligible': bid not in prev_hr,
                    'hit_hr': bid in actual_pids,
                    'opp_pitcher': opp_id,
                })

all_X = np.vstack([np.concatenate([r['core'], r['adj']]) for r in rows]).astype(np.float32)
if best_mode == 'E' and model_gbm:
    probs_lr = model_logreg.predict_proba(scaler.transform(all_X))[:, 1]
    probs_gbm = model_gbm.predict_proba(all_X)[:, 1]
    probs = 0.5 * probs_lr + 0.5 * probs_gbm
else:
    probs = model_logreg.predict_proba(scaler.transform(all_X))[:, 1]

for i, r in enumerate(rows):
    r['prob'] = float(probs[i])
    r['prob_lr'] = float(probs_lr[i]) if best_mode == 'E' else r['prob']
    r['prob_gbm'] = float(probs_gbm[i]) if best_mode == 'E' else 0

# Sort all by probability
ranked = sorted(range(len(rows)), key=lambda i: -rows[i]['prob'])

print(f'\n=== FULL GLOBAL RANKING (top 50) ===')
print(f'  {"Rk":>3} {"Player":<24} {"P(HR)":>6} {"LR":>6} {"GBM":>6} {"Elig":>4} {"HIT?":>4} {"Matchup":<35}')
for rank, idx in enumerate(ranked[:50], 1):
    r = rows[idx]
    matchup = f'{r["away_team"]} @ {r["home_team"]}'
    elig = 'yes' if r['eligible'] else 'SKIP'
    hit = 'YES' if r['hit_hr'] else ''
    print(f'  {rank:>3} {r["name"]:<24} {r["prob"]:>5.1%} {r["prob_lr"]:>5.1%} {r["prob_gbm"]:>5.1%} {elig:>4} {hit:>4} {matchup}')

print(f'\n=== ACTUAL HR HITTERS - WHERE THEY RANKED ===')
hr_hitter_ranks = []
for rank, idx in enumerate(ranked, 1):
    r = rows[idx]
    if r['hit_hr']:
        hr_hitter_ranks.append((rank, r))
        
for rank, r in hr_hitter_ranks:
    elig = 'yes' if r['eligible'] else 'SKIP'
    print(f'  Rank {rank:>3}  {r["name"]:<24} P(HR)={r["prob"]:.1%} Elig={elig}')

# Distribution analysis
print(f'\n=== DISTRIBUTION OF HR HITTERS ===')
in_top10 = sum(1 for rk, _ in hr_hitter_ranks if rk <= 10)
in_top20 = sum(1 for rk, _ in hr_hitter_ranks if rk <= 20)
in_top30 = sum(1 for rk, _ in hr_hitter_ranks if rk <= 30)
in_top50 = sum(1 for rk, _ in hr_hitter_ranks if rk <= 50)
in_top100 = sum(1 for rk, _ in hr_hitter_ranks if rk <= 100)
total_hr = len(hr_hitter_ranks)
print(f'  In top 10:  {in_top10}/{total_hr}')
print(f'  In top 20:  {in_top20}/{total_hr}')
print(f'  In top 30:  {in_top30}/{total_hr}')
print(f'  In top 50:  {in_top50}/{total_hr}')
print(f'  In top 100: {in_top100}/{total_hr}')

# Check SKIP (prev-day HR) analysis
print(f'\n=== PREV-DAY HR SKIP ANALYSIS ===')
skipped_hits = [(rk, r) for rk, r in hr_hitter_ranks if not r['eligible']]
print(f'  HR hitters we skipped due to prev-day HR: {len(skipped_hits)}')
for rk, r in skipped_hits:
    print(f'    Rank {rk:>3} {r["name"]:<24} P(HR)={r["prob"]:.1%}')

# Feature analysis for top picks vs actual HR hitters
print(f'\n=== FEATURE COMPARISON: Our Top 10 vs Actual HR Hitters ===')
from itertools import islice
feat_names = _trainer_globals['ALL_FEATURE_NAMES']
print(f'  Feature names: {feat_names}')
print()

top10_indices = ranked[:10]
hr_indices = [idx for rank, idx in enumerate(ranked) if rows[idx]['hit_hr']]

for label, indices in [('OUR TOP 10', top10_indices), ('ACTUAL HR HITTERS (avg)', hr_indices[:20])]:
    feats = np.mean([np.concatenate([rows[i]['core'], rows[i]['adj']]) for i in indices], axis=0)
    print(f'  {label}:')
    for j, fn in enumerate(feat_names):
        print(f'    {fn:<16} {feats[j]:>8.4f}')
    print()

# Check if backtest validation dates had similar patterns
print('=== BACKTEST COMPARISON ===')
print('Checking if April 18 pattern is typical...')
# Count avg HR hitters per date in backtest
test_dates_hrs = {}
for d in ['2026-03-27','2026-03-28','2026-03-29','2026-03-30','2026-03-31',
          '2026-04-01','2026-04-02','2026-04-03','2026-04-04','2026-04-05',
          '2026-04-06','2026-04-07','2026-04-08','2026-04-09','2026-04-10',
          '2026-04-11','2026-04-12','2026-04-13','2026-04-14','2026-04-15']:
    dc = load_cache(f'date_{d}')
    if dc and 'actual' in dc:
        test_dates_hrs[d] = len(dc['actual'])
        
print(f'  HR hitters per date in backtest:')
for d, n in sorted(test_dates_hrs.items()):
    print(f'    {d}: {n}')
print(f'  Today ({DATE}): {len(actual)} (some games still in progress)')
avg_bt = np.mean(list(test_dates_hrs.values()))
print(f'  Backtest avg: {avg_bt:.1f}')
