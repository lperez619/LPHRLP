import os
import json

def apply_odds_patches():
    with open('HR_Trainer_Odds.py', 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Insert load_day_odds
    if 'def load_day_odds(' not in text:
        old_cache = "def save_cache(key, data):\n    path = os.path.join(CACHE_DIR, f'{key}.pkl')\n    with open(path, 'wb') as f:\n        pickle.dump(data, f)"
        
        new_cache = old_cache + "\n\ndef load_day_odds(date_str):\n    import os, json\n    path = os.path.join('odds_cache', f'{date_str}.json')\n    if os.path.exists(path):\n        try:\n            with open(path, 'r', encoding='utf-8') as f:\n                return json.load(f)\n        except Exception:\n            pass\n    return {}"
        text = text.replace(old_cache, new_cache)

    # 2. _precompute_row signature
    old_sig = "savant_day=None, team_pitch=None,\n                    batter_zones=None, pitcher_zones=None):"
    new_sig = "savant_day=None, team_pitch=None,\n                    batter_zones=None, pitcher_zones=None, day_odds=None):"
    if new_sig not in text: text = text.replace(old_sig, new_sig)
    
    # 3. _precompute_row logic
    old_logic = "    zone_f     = compute_zone_matchup(bid, opp_id,\n                                      batter_zones or {}, pitcher_zones or {})\n\n    return (bid, pk, away_team, home_team,"
    new_logic = "    zone_f     = compute_zone_matchup(bid, opp_id,\n                                      batter_zones or {}, pitcher_zones or {})\n\n    b_name = b.get('name', '')\n    b_odds = day_odds.get(b_name, 0.05) if day_odds else 0.05\n\n    return (bid, pk, away_team, home_team,"
    if 'b_odds = day_odds.get' not in text: text = text.replace(old_logic, new_logic)

    # 4. _precompute_row return tuple
    old_ret = "            pit_rf, ha_f, raw_bvp,\n            b['name'], wx_hist_f, dn_bat_f, zone_f, t_vuln)"
    new_ret = "            pit_rf, ha_f, raw_bvp,\n            b['name'], wx_hist_f, dn_bat_f, zone_f, t_vuln, b_odds)"
    if 'b_odds)' not in text: text = text.replace(old_ret, new_ret)

    # 5. unpacked features (both core and adj)
    old_unpack = "     pit_rf, ha_f, raw_bvp,\n     name, wx_hist_f, dn_bat_f, zone_f, t_vuln) = row"
    new_unpack = "     pit_rf, ha_f, raw_bvp,\n     name, wx_hist_f, dn_bat_f, zone_f, t_vuln, b_odds) = row"
    text = text.replace(old_unpack, new_unpack)

    # 6. array append
    old_arr = "        1.0 if norm_rv is not None else 0.0,\n        _safe_log(ha_f),\n        _safe_log(dn_bat_f),\n    ], dtype=np.float32)"
    new_arr = "        1.0 if norm_rv is not None else 0.0,\n        _safe_log(ha_f),\n        _safe_log(dn_bat_f),\n        _safe_log(b_odds),\n    ], dtype=np.float32)"
    if '_safe_log(b_odds)' not in text: text = text.replace(old_arr, new_arr)

    # 7. passing day odds mapping in 2026 iteration
    old_call1 = "                    batter_zones_global, pitcher_zones_global,\n                )\n                if row is not None:"
    new_call1 = "                    batter_zones_global, pitcher_zones_global, day_odds=load_day_odds(date_str)\n                )\n                if row is not None:"
    if 'day_odds=load_day_odds(date_str)' not in text: text = text.replace(old_call1, new_call1)
    
    # 8. passing day odds mapping in build_training_data iteration
    old_call2 = "                        batter_zones=b_zones, pitcher_zones=p_zones,\n                  )\n                  if row is not None:"
    new_call2 = "                        batter_zones=b_zones, pitcher_zones=p_zones, day_odds=load_day_odds(g_date_str)\n                  )\n                  if row is not None:"
    if 'day_odds=load_day_odds(g_date_str)' not in text: text = text.replace(old_call2, new_call2)

    with open('HR_Trainer_Odds.py', 'w', encoding='utf-8') as f:
        f.write(text)
    
    print('Patched successfully')

if __name__ == '__main__':
    apply_odds_patches()
