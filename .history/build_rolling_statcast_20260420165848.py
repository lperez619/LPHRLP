import os
import pickle
import time
import requests
import pandas as pd
import io
import json
import numpy as np
from datetime import datetime, timedelta

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORICAL_FILE = os.path.join(CACHE_DIR, 'historical_rolling_form.pkl')
SAVANT_CACHE_FILE = os.path.join(CACHE_DIR, 'savant_rolling_cache.pkl')

def build_rolling_savant_cache():
    if not os.path.exists(HISTORICAL_FILE):
        print("Required historical_rolling_form.pkl not found. Please run build_historical_form.py first.")
        return

    with open(HISTORICAL_FILE, 'rb') as f:
        hist_cache = pickle.load(f)
    dates = sorted(list(hist_cache.keys()))
    
    print(f"Found {len(dates)} game dates. Building trailing 30-day Savant cache...")
    
    savant_cache = {}
    if os.path.exists(SAVANT_CACHE_FILE):
        with open(SAVANT_CACHE_FILE, 'rb') as f:
            savant_cache = pickle.load(f)
            
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    # Process from newest to oldest to skip already downloaded
    for i, date_str in enumerate(dates):
        if date_str in savant_cache:
            continue
            
        print(f"[{i+1}/{len(dates)}] Fetching Savant stats for 30 days ending before {date_str}...")
        
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        end_date = (date_obj - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (date_obj - timedelta(days=30)).strftime('%Y-%m-%d')
        
        url = f"https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfGT=R%7C&game_date_gt={start_date}&game_date_lt={end_date}&player_type=batter&group_by=name&min_pitches=0&min_results=0"
        
        try:
            r = session.get(url, timeout=15)
            df = pd.read_csv(io.StringIO(r.text))
            
            day_cache = {}
            if 'player_id' in df.columns:
                for _, row in df.iterrows():
                    pid = int(row['player_id'])
                    
                    ev = row['launch_speed'] if 'launch_speed' in row and not pd.isna(row['launch_speed']) else 0.0
                    barrels = row['barrels_total'] if 'barrels_total' in row and not pd.isna(row['barrels_total']) else 0.0
                    pa = row['pa'] if 'pa' in row and not pd.isna(row['pa']) else 0
                    hh_pct = row['hardhit_percent'] if 'hardhit_percent' in row and not pd.isna(row['hardhit_percent']) else 0.0
                    bip = row['bip'] if 'bip' in row and not pd.isna(row['bip']) else 0
                    
                    day_cache[pid] = {
                        'ev': float(ev),
                        'brl': float(barrels),
                        'pa': int(pa),
                        'hh_pct': float(hh_pct),
                        'bip': int(bip)
                    }
                    
            savant_cache[date_str] = day_cache
            
            if (i + 1) % 10 == 0:
                with open(SAVANT_CACHE_FILE, 'wb') as f:
                    pickle.dump(savant_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error fetching {date_str}: {e}")
            time.sleep(2)
            
    with open(SAVANT_CACHE_FILE, 'wb') as f:
        pickle.dump(savant_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Savant rolling cache built successfully!")

if __name__ == '__main__':
    build_rolling_savant_cache()
