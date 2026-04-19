"""
HR Picks for Today — uses the trained ensemble model from HR Trainer.py
to generate ranked HR predictions for today's MLB games.

Usage: python hr_picks_today.py [YYYY-MM-DD]
  If no date given, defaults to today.
"""
import sys, os, math, time, pickle, json, warnings
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Bootstrap: import everything from HR Trainer without running its main block
# We do this by reading the file and exec'ing only up to the main section.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_trainer_path = os.path.join(_SCRIPT_DIR, 'HR Trainer.py')

# Build a restricted globals dict, then exec the definitions portion
_trainer_globals = {'__name__': '_hr_trainer_lib', '__file__': _trainer_path}
with open(_trainer_path, 'r', encoding='utf-8') as f:
    _full_source = f.read()

# Find the "# -- main" marker and only exec up to it
_main_marker = '\nTEST_DATES'
_alt_markers = ['\n# All completed regular-season dates']
_cut_pos = len(_full_source)
for marker in [_main_marker] + _alt_markers:
    pos = _full_source.find(marker)
    if pos != -1 and pos < _cut_pos:
        _cut_pos = pos

_lib_source = _full_source[:_cut_pos]
exec(compile(_lib_source, _trainer_path, 'exec'), _trainer_globals)

# Pull out everything we need
load_cache          = _trainer_globals['load_cache']
save_cache          = _trainer_globals['save_cache']
get_games           = _trainer_globals['get_games']
get_game_time_slot  = _trainer_globals['get_game_time_slot']
get_roster          = _trainer_globals['get_roster']
fetch_raw_hitting   = _trainer_globals['fetch_raw_hitting']
fetch_stats_bulk    = _trainer_globals['fetch_stats_bulk']
fetch_weather       = _trainer_globals['fetch_weather']
get_actual_hrs      = _trainer_globals['get_actual_hrs']
fetch_spring_stats  = _trainer_globals['fetch_spring_stats']
fetch_statcast_power = _trainer_globals['fetch_statcast_power']
fetch_bat_tracking  = _trainer_globals['fetch_bat_tracking']
fetch_pitch_arsenal = _trainer_globals['fetch_pitch_arsenal']
fetch_hitting_splits = _trainer_globals['fetch_hitting_splits']
combine_hitting     = _trainer_globals['combine_hitting']
build_park_table    = _trainer_globals['build_park_table']
blend_pitcher_stats = _trainer_globals['blend_pitcher_stats']
_precompute_row     = _trainer_globals['_precompute_row']
row_to_core_features = _trainer_globals['row_to_core_features']
row_to_adj_features  = _trainer_globals['row_to_adj_features']
row_to_all_features  = _trainer_globals['row_to_all_features']
CORE_FEATURE_NAMES  = _trainer_globals['CORE_FEATURE_NAMES']
ADJ_FEATURE_NAMES   = _trainer_globals['ADJ_FEATURE_NAMES']
ALL_FEATURE_NAMES   = _trainer_globals['ALL_FEATURE_NAMES']
PITCHER_SEASON      = _trainer_globals['PITCHER_SEASON']
SEASON_WEIGHT_CONFIGS = _trainer_globals['SEASON_WEIGHT_CONFIGS']
PF_BLEND_CONFIGS    = _trainer_globals['PF_BLEND_CONFIGS']
BAT_SPEED_MEAN      = _trainer_globals.get('BAT_SPEED_MEAN', 71.0)
BLAST_MEAN_ACT      = _trainer_globals.get('BLAST_MEAN_ACT', 71.0)
STADIUMS            = _trainer_globals.get('STADIUMS', {})
MIN_PA              = _trainer_globals['MIN_PA']

warnings.filterwarnings('ignore')

# ── Parse date argument ────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    TARGET_DATE = sys.argv[1]
else:
    TARGET_DATE = datetime.now().strftime('%Y-%m-%d')

print('=' * 68)
print(f'  HR PICKS FOR {TARGET_DATE}')
print('=' * 68)

# ── 1. Load trained model ─────────────────────────────────────────────────────
print('\n  Loading model...')
artifact = load_cache('hr_logreg_model')
if artifact is None:
    print('  ERROR: No trained model found (hr_logreg_model.pkl).')
    print('  Run "HR Trainer.py" first to train the model.')
    sys.exit(1)

best_mode      = artifact['best_mode']
model_logreg   = artifact.get('model_logreg', artifact.get('model'))
model_gbm      = artifact.get('model_gbm')
scaler         = artifact['scaler']
feature_names  = artifact['feature_names']
pick_strategy  = artifact.get('pick_strategy', 'global')
best_glob_top  = artifact.get('best_glob_top', 10)
best_skip      = artifact.get('best_skip', 0)
best_take      = artifact.get('best_take', 3)
ensemble_lr_w  = artifact.get('ensemble_lr_weight', 0.5)
calibrator     = artifact.get('calibrator')

print(f'  Model mode: {best_mode}')
print(f'  Pick strategy: {pick_strategy}')
print(f'  Calibration: {"yes" if calibrator else "no"}')
if pick_strategy == 'global':
    print(f'  Global top-{best_glob_top}')
else:
    print(f'  Per-game skip={best_skip}, take={best_take}')

# ── 2. Load season-level shared data ──────────────────────────────────────────
print('\n  Loading season-level data...')
barrel = load_cache(f'barrel_{PITCHER_SEASON}')
if barrel is None:
    barrel = fetch_statcast_power(PITCHER_SEASON)
    save_cache(f'barrel_{PITCHER_SEASON}', barrel)
print(f'  Power data: {len(barrel)} players')

_bat_cache = load_cache(f'bat_tracking_{PITCHER_SEASON}')
if _bat_cache is None:
    bat_t = fetch_bat_tracking(PITCHER_SEASON)
    save_cache(f'bat_tracking_{PITCHER_SEASON}',
               {'bat_t': bat_t, 'speed_mean': BAT_SPEED_MEAN, 'blast_mean': BLAST_MEAN_ACT})
else:
    bat_t = _bat_cache['bat_t']
print(f'  Bat tracking: {len(bat_t)} players')

bvp = load_cache(f'arsenal_batter_{PITCHER_SEASON}')
if bvp is None:
    bvp = fetch_pitch_arsenal('batter', PITCHER_SEASON)
    save_cache(f'arsenal_batter_{PITCHER_SEASON}', bvp)
print(f'  Batter arsenal: {len(bvp)} batters')

pars = load_cache(f'arsenal_pitcher_{PITCHER_SEASON}')
if pars is None:
    pars = fetch_pitch_arsenal('pitcher', PITCHER_SEASON)
    save_cache(f'arsenal_pitcher_{PITCHER_SEASON}', pars)
print(f'  Pitcher arsenal: {len(pars)} pitchers')

spring = fetch_spring_stats()

bvp_career_global = load_cache('bvp_career') or {}
print(f'  BvP career: {len(bvp_career_global)} batters')

weather_splits_global = load_cache('weather_splits') or {}
print(f'  Weather splits: {len(weather_splits_global)} batters')

fetch_batter_zones   = _trainer_globals['fetch_batter_zones']
fetch_pitcher_zones  = _trainer_globals['fetch_pitcher_zones']
compute_zone_matchup = _trainer_globals['compute_zone_matchup']

# ── 3. Fetch today's games ────────────────────────────────────────────────────
print(f'\n  Fetching games for {TARGET_DATE}...')
games = get_games(TARGET_DATE)
if not games:
    print('  No games found for this date.')
    sys.exit(0)

slots = {}
for g in games:
    s = get_game_time_slot(g['game_date'])
    slots[s] = slots.get(s, 0) + 1
print(f'  {len(games)} games ({slots.get("day",0)} day / {slots.get("prime",0)} prime / {slots.get("late",0)} late)')

# Build batter map from lineups
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

all_bat = {bid for v in batter_map.values() for s in ('home', 'away') for bid in v[s]}
all_pit = {g[k] for g in games for k in ('home_pitcher_id', 'away_pitcher_id') if g[k]}
print(f'  Batters: {len(all_bat)}  Pitchers: {len(all_pit)}')

# ── 4. Fetch today's stats ────────────────────────────────────────────────────
print('\n  Fetching stats...')
prev_date = (datetime.strptime(TARGET_DATE, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')

with ThreadPoolExecutor(max_workers=4) as pool:
    fut_25  = pool.submit(fetch_raw_hitting, all_bat, 2025)
    fut_26  = pool.submit(fetch_raw_hitting, all_bat, 2026)
    fut_pt  = pool.submit(fetch_stats_bulk, all_pit, 'pitching', PITCHER_SEASON)
    fut_wx  = pool.submit(fetch_weather, games, TARGET_DATE)
    fut_pac = pool.submit(get_actual_hrs, prev_date)
    raw_25      = fut_25.result()
    raw_26      = fut_26.result()
    pit_stats   = fut_pt.result()
    weather     = fut_wx.result()
    prev_actual = fut_pac.result()

prev_hr_hitters = set(prev_actual.keys())
print(f'  2025 hitting: {len(raw_25)}  2026 hitting: {len(raw_26)}')
print(f'  Pitchers: {sum(1 for v in pit_stats.values() if v.get("stat"))}')
outdoor = sum(1 for g in games if not STADIUMS.get(g['home_team'], {}).get('indoor', True))
print(f'  Weather: {len(weather)}/{outdoor} outdoor')
print(f'  Prev-day HR hitters: {len(prev_hr_hitters)}')

# ── 5. Build feature vectors ──────────────────────────────────────────────────
print('\n  Building feature vectors...')
sw_label, sw_weights = SEASON_WEIGHT_CONFIGS[0]
pf_label, pf_blend = PF_BLEND_CONFIGS[0]
park_tbl = build_park_table(pf_blend)

raw_by_season = {}
for yr, raw in [(2025, raw_25), (2026, raw_26)]:
    if sw_weights.get(yr, 0) > 0:
        raw_by_season[yr] = raw
batter_stats = combine_hitting(raw_by_season, sw_weights)

pit_blended = blend_pitcher_stats(pit_stats, {})

# Fetch zone data for today's batters and pitchers
print('  Loading zone data...')
batter_zones_global = load_cache('batter_zones_2025')
if batter_zones_global is None:
    batter_zones_global = fetch_batter_zones(all_bat)
    save_cache('batter_zones_2025', batter_zones_global)
else:
    missing = all_bat - set(batter_zones_global.keys())
    if missing:
        new_zones = fetch_batter_zones(missing)
        batter_zones_global.update(new_zones)
        save_cache('batter_zones_2025', batter_zones_global)
print(f'  Batter zones: {len(batter_zones_global)} players')

pitcher_zones_global = load_cache('pitcher_zones_2025')
if pitcher_zones_global is None:
    pitcher_zones_global = fetch_pitcher_zones(all_pit)
    save_cache('pitcher_zones_2025', pitcher_zones_global)
else:
    missing = all_pit - set(pitcher_zones_global.keys())
    if missing:
        new_zones = fetch_pitcher_zones(missing)
        pitcher_zones_global.update(new_zones)
        save_cache('pitcher_zones_2025', pitcher_zones_global)
print(f'  Pitcher zones: {len(pitcher_zones_global)} players')

# Fetch splits for today's batters
splits_global = load_cache('splits_2025')
if splits_global is None:
    print('  Fetching hitting splits...')
    splits_global = fetch_hitting_splits(all_bat, 2025)
    save_cache('splits_2025', splits_global)
else:
    missing = all_bat - set(splits_global.keys())
    if missing:
        print(f'  Fetching {len(missing)} new splits...')
        new_splits = fetch_hitting_splits(missing, 2025)
        splits_global.update(new_splits)
        save_cache('splits_2025', splits_global)

recent_bat = {}
pitcher_recent = {}

# Diagnostics: track dropped batters
_dropped = []
_all_lineup_bids = set()

rows = []      # (row_tuple, lineup_pos, game_pk, side_info)
game_rows = {} # {game_pk: [row_indices]}

for g in games:
    pk = g['game_pk']
    info = batter_map[pk]
    game_rows[pk] = []
    for side, opp_id, is_home in [
        ('away', g['home_pitcher_id'], False),
        ('home', g['away_pitcher_id'], True),
    ]:
        lineup = info[side]
        for lineup_idx, bid in enumerate(lineup):
            _all_lineup_bids.add(bid)
            row = _precompute_row(
                bid, opp_id, info['home_team'], pk,
                info['time_slot'], is_home, info['away_team'],
                batter_stats, pit_blended, barrel, park_tbl, bat_t,
                bvp, pars, splits_global,
                recent_bat, pitcher_recent, bvp_career_global,
                weather, spring, weather_splits_global,
                batter_zones_global, pitcher_zones_global,
            )
            if row is not None:
                lpos = lineup_idx / 8.0 if len(lineup) > 1 else 0.5
                core_feat = row_to_core_features(row)
                adj_feat  = row_to_adj_features(row)
                adj_feat[-1] = lpos
                idx = len(rows)
                is_eligible = True   # all batters eligible (no prev-day HR skip)
                rows.append({
                    'row': row, 'core': core_feat, 'adj': adj_feat,
                    'bid': bid, 'pk': pk, 'name': row[22],
                    'home_team': row[3], 'away_team': row[2],
                    'lineup_pos': lineup_idx + 1,
                    'eligible': is_eligible,
                })
                game_rows[pk].append(idx)
            else:
                # Figure out why this batter was dropped
                reason = 'not in batter_stats (< MIN_PA)'
                bs = batter_stats.get(bid)
                if bs and bs.get('stat'):
                    pa = bs['stat'].get('plateAppearances', 0)
                    reason = f'PA={pa} < MIN_PA={MIN_PA}' if pa < MIN_PA else 'unknown'
                team = info['home_team'] if is_home else info['away_team']
                _dropped.append((bid, team, lineup_idx + 1, reason))

print(f'  {len(rows)} eligible batter-game rows')
if _dropped:
    # Try to get names for dropped batters from the raw hitting data
    _all_raw = {}
    for yr_raw in [raw_25, raw_26]:
        for pid, d in yr_raw.items():
            _all_raw[pid] = d.get('name', f'ID:{pid}')
    print(f'  {len(_dropped)} batters DROPPED from lineups:')
    for bid, team, lpos, reason in _dropped:
        name = _all_raw.get(bid, f'ID:{bid}')
        print(f'    {name:<24} {team:<5}  lineup #{lpos}  -- {reason}')

if not rows:
    print('  No eligible rows to score.')
    sys.exit(0)

# Stack into feature matrices
all_X = np.vstack([
    np.concatenate([r['core'], r['adj']]) for r in rows
]).astype(np.float32)

# ── 6. Score with model ───────────────────────────────────────────────────────
print('\n  Scoring...')

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

if best_mode == 'E' and model_gbm is not None:
    # Ensemble: weighted LogReg + GBM
    probs_lr = model_logreg.predict_proba(scaler.transform(all_X))[:, 1]
    probs_gbm = model_gbm.predict_proba(all_X)[:, 1]
    probs = ensemble_lr_w * probs_lr + (1 - ensemble_lr_w) * probs_gbm
elif best_mode == 'D' and model_gbm is not None:
    probs = model_gbm.predict_proba(all_X)[:, 1]
elif best_mode == 'A':
    core_X = np.vstack([r['core'] for r in rows]).astype(np.float32)
    adj_X  = np.vstack([r['adj'] for r in rows]).astype(np.float32)
    base_logit = model_logreg.decision_function(scaler.transform(core_X))
    adj_offset = adj_X.sum(axis=1)
    probs = _sigmoid(base_logit + adj_offset)
else:  # Mode C
    probs = model_logreg.predict_proba(scaler.transform(all_X))[:, 1]

# Attach probabilities
for i, r in enumerate(rows):
    r['prob'] = float(probs[i])

# ── 7. Generate picks ─────────────────────────────────────────────────────────
print()
print('=' * 68)
print(f'  HR PICKS FOR {TARGET_DATE}  (Mode {best_mode})')
print('=' * 68)

if pick_strategy == 'global':
    # Global: rank all eligible batters, take top-N
    eligible = [(i, rows[i]['prob']) for i in range(len(rows)) if rows[i]['eligible']]
    eligible.sort(key=lambda x: -x[1])
    picks = eligible[:best_glob_top]
    print(f'\n  Top {best_glob_top} picks (global ranking):')
else:
    # Per-game: from each game, skip top-N, take next-M
    picks = []
    for pk, idxs in game_rows.items():
        elig = [(i, rows[i]['prob']) for i in idxs if rows[i]['eligible']]
        elig.sort(key=lambda x: -x[1])
        picks.extend(elig[best_skip:best_skip + best_take])
    picks.sort(key=lambda x: -x[1])
    print(f'\n  {len(picks)} picks (per-game skip={best_skip}, take={best_take}):')

# Display picks
print()
print(f"  {'#':>3}  {'Player':<24} {'Matchup':<30} {'Pos':>3} {'P(HR)':>7}")
print(f"  {'-' * 72}")
for rank, (idx, prob) in enumerate(picks, 1):
    r = rows[idx]
    matchup = f"{r['away_team']} @ {r['home_team']}"
    print(f"  {rank:>3}  {r['name']:<24} {matchup:<30} {r['lineup_pos']:>3} {prob:>6.1%}")

# ── 8. Full rankings per game ─────────────────────────────────────────────────
print()
print('=' * 68)
print('  FULL RANKINGS BY GAME')
print('=' * 68)

for g in games:
    pk = g['game_pk']
    idxs = game_rows.get(pk, [])
    if not idxs:
        continue
    away = g['away_team']; home = g['home_team']
    ts = get_game_time_slot(g['game_date'])

    # Sort by probability
    ranked = sorted(idxs, key=lambda i: -rows[i]['prob'])

    print(f"\n  {away} @ {home}  ({ts})")
    print(f"  {'#':>3}  {'Player':<24} {'P(HR)':>7}  {'Pos':>3}")
    print(f"  {'-' * 44}")
    for rank, i in enumerate(ranked, 1):
        r = rows[i]
        print(f"  {rank:>3}  {r['name']:<24} {r['prob']:>6.1%}  {r['lineup_pos']:>3}")

# ── 9. Summary ─────────────────────────────────────────────────────────────────
print()
print('=' * 68)
n_picked = len(picks)
avg_prob = np.mean([rows[i]['prob'] for i, _ in picks]) if picks else 0
print(f'  {n_picked} picks | avg P(HR) = {avg_prob:.1%} | '
      f'expected HRs = {sum(rows[i]["prob"] for i, _ in picks):.1f}')
print('=' * 68)
