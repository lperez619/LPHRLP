import os
import json
import time
import requests
from datetime import datetime, timedelta

API_KEY = 'a06894b1d1bc73b219d9d0af26c2b5f4'
SPORT = 'baseball_mlb'
TEST_DATES = [
    '2026-03-27','2026-03-28','2026-03-29','2026-03-30',
    '2026-03-31','2026-04-01','2026-04-02','2026-04-03',
    '2026-04-04','2026-04-05','2026-04-06','2026-04-07',
    '2026-04-08','2026-04-09','2026-04-10','2026-04-11',
    '2026-04-12','2026-04-13','2026-04-14','2026-04-15',
    '2026-04-16','2026-04-17','2026-04-18','2026-04-19',
]

def get_events(dt_str):
    url = f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/events'
    r = requests.get(url, params={'apiKey': API_KEY, 'date': dt_str})
    if r.status_code == 200:
        return r.json().get('data', [])
    print(f"Error fetching events for {dt_str}: {r.status_code} - {r.text}")
    return []

def get_event_odds(event_id, dt_str):
    url = f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/events/{event_id}/odds'
    r = requests.get(url, params={
        'apiKey': API_KEY,
        'date': dt_str,
        'regions': 'us',
        'markets': 'batter_home_runs'
    })
    if r.status_code == 200:
        d = r.json()
        return d.get('data', {}).get('bookmakers', [])
    print(f"Error fetching odds for {event_id} at {dt_str}: {r.status_code} - {r.text}")
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
                        # Convert american or decimal to implied probability
                        # By default The Odds API returns decimal odds (e.g. 3.50 -> 1/3.5 = 28.5%)
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

    for date_str in TEST_DATES[:1]: # Start with just the first day for testing
        print(f"\\n--- Backfilling {date_str} ---")
        out_file = f"odds_cache/{date_str}.json"
        
        dt_start_of_day = f"{date_str}T14:00:00Z"
        events = get_events(dt_start_of_day)
        print(f"Found {len(events)} events around midday {dt_start_of_day}")
        
        day_odds = {}
        
        for ev in events:
            # We want odds from 1 hour before commence_time
            dt_com = datetime.strptime(ev['commence_time'], '%Y-%m-%dT%H:%M:%SZ')
            dt_odds = dt_com - timedelta(hours=1)
            odds_str = dt_odds.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            print(f"  {ev['home_team']} vs {ev['away_team']} | Commence: {ev['commence_time']} | Query: {odds_str}")
            
            bms = get_event_odds(ev['id'], odds_str)
            if bms:
                game_odds = extract_odds(bms)
                print(f"    -> Extracted odds for {len(game_odds)} players")
                for p, pprob in game_odds.items():
                    if p not in day_odds:
                        day_odds[p] = pprob
            
            time.sleep(1) # API rate limit safety
            
        with open(out_file, 'w') as f:
            json.dump(day_odds, f, indent=2)
            
        print(f"Saved {len(day_odds)} player odds blocks for {date_str} to {out_file}")

if __name__ == '__main__':
    main()
