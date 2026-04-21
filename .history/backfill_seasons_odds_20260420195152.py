import os
import json
import time
import glob
import requests
from datetime import datetime, timedelta
import pickle

API_KEY = 'a06894b1d1bc73b219d9d0af26c2b5f4'
SPORT = 'baseball_mlb'

def get_events(dt_str):
    url = f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/events'
    r = requests.get(url, params={'apiKey': API_KEY, 'date': dt_str})
    if r.status_code == 200:
        return r.json().get('data', [])
    return []

def get_event_odds(event_id, dt_str):
    url = f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/events/{event_id}/odds'
    r = requests.get(url, params={
        'apiKey': API_KEY, 'date': dt_str, 'regions': 'us', 'markets': 'batter_home_runs'
    })
    if r.status_code == 200:
        d = r.json()
        return d.get('data', {}).get('bookmakers', [])
    return []

def extract_odds(bms):
    player_probs = {}
    for bm in bms:
        for market in bm.get('markets', []):
            if market['key'] == 'batter_home_runs':
                for outcome in market.get('outcomes', []):
                    player = outcome['description']
                    name = outcome['name']
                    if name.lower() == 'over':
                        price = outcome['price']
                        prob = 1.0 / price if price > 0 else 0
                        if player not in player_probs:
                            player_probs[player] = []
                        player_probs[player].append(prob)

    avg_probs = {}
    for player, probs in player_probs.items():
        avg_probs[player] = sum(probs) / len(probs)
    return avg_probs

def main():
    if not os.path.exists('odds_cache'):
        os.makedirs('odds_cache')

    # Build dates from March 20 to Oct 5 for both years
    season_dates = set()
    for year in [2024, 2025]:
        d1 = datetime(year, 3, 20)
        d2 = datetime(year, 10, 5)
        d = d1
        while d <= d2:
            season_dates.add(d.strftime('%Y-%m-%d'))
            d += timedelta(days=1)

    sorted_dates = sorted(list(season_dates))
    print(f"Found {len(sorted_dates)} total historical dates.")

    for date_str in sorted_dates:
        out_file = f"odds_cache/{date_str}.json"
        if os.path.exists(out_file):
            continue

        print(f"\\n--- Backfilling {date_str} ---")
        dt_start_of_day = f"{date_str}T14:00:00Z"
        events = get_events(dt_start_of_day)
        print(f"Found {len(events)} events")
        
        day_odds = {}
        for ev in events:
            dt_com = datetime.strptime(ev['commence_time'], '%Y-%m-%dT%H:%M:%SZ')
            dt_odds = dt_com - timedelta(hours=1)
            odds_str = dt_odds.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            bms = get_event_odds(ev['id'], odds_str)
            if bms:
                game_odds = extract_odds(bms)
                for p, pprob in game_odds.items():
                    if p not in day_odds:
                        day_odds[p] = pprob
            
            time.sleep(0.5) # API limit
            
        with open(out_file, 'w') as f:
            json.dump(day_odds, f, indent=2)

if __name__ == '__main__':
    pass #main()
