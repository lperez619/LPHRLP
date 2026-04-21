import sys, os, requests, json
from datetime import datetime

API_KEY = 'e1018e0128aaebebfc93aa09a48c2d9a'
SPORT = 'baseball_mlb'
CACHE_DIR = 'odds_cache'

def fetch_live_events():
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/events'
    r = requests.get(url, params={'apiKey': API_KEY})
    return r.json() if r.status_code == 200 else []

def fetch_event_odds(event_id):
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds'
    r = requests.get(url, params={
        'apiKey': API_KEY, 'regions': 'us', 'markets': 'batter_home_runs', 'oddsFormat': 'decimal'
    })
    return r.json() if r.status_code == 200 else {}

def fetch_live_odds():
    TARGET_DATE = datetime.now().strftime('%Y-%m-%d')
    if len(sys.argv) > 1:
        TARGET_DATE = sys.argv[1]

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{TARGET_DATE}.json")
    history_path = os.path.join(CACHE_DIR, f"{TARGET_DATE}_history.json")
    movement_path = os.path.join(CACHE_DIR, f"{TARGET_DATE}_movement.json")

    # Load existing history if it exists
    history = {}
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

    print(f"Fetching upcoming live events...")
    events = fetch_live_events()
    player_probs = {}
    
    for event in events:
        event_id = event['id']
        odds_data = fetch_event_odds(event_id)
        
        bms = odds_data.get('bookmakers', [])
        for bm in bms:
            for market in bm.get('markets', []):
                if market['key'] == 'batter_home_runs':
                    for outcome in market.get('outcomes', []):
                        player = outcome.get('description', '')
                        name = outcome.get('name', '')
                        if player and name.lower() == 'over':
                            price = outcome.get('price', 0)
                            if price > 0:
                                player_probs.setdefault(player, []).append(1.0 / price)

    avg_probs = {}
    current_time = datetime.now().isoformat()
    movement_deltas = {}

    for player, probs in player_probs.items():
        if len(probs) > 0:
            p_name = player.lower().replace("'", "").replace(".", "")
            current_prob = round(sum(probs) / len(probs), 4)
            avg_probs[p_name] = current_prob
            
            # Update history
            history.setdefault(p_name, []).append({'time': current_time, 'prob': current_prob})
            
            # Calculate movement (Current Prob - Opening Prob)
            opening_prob = history[p_name][0]['prob']
            movement_deltas[p_name] = round(current_prob - opening_prob, 4)

    # 1. Save standard current odds (for basic compatibility)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(avg_probs, f, indent=2)
        
    # 2. Save full time-series history
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    # 3. Save calculated movement deltas
    with open(movement_path, 'w', encoding='utf-8') as f:
        json.dump(movement_deltas, f, indent=2)
        
    print(f"Successfully processed {len(avg_probs)} players.")
    print(f"Odds snapshot saved. Line movement tracking updated.")

if __name__ == '__main__':
    fetch_live_odds()
