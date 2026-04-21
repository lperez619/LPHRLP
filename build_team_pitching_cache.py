import os
import pickle
import requests
import pandas as pd
from datetime import datetime, timedelta

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM_PITCHING_CACHE_FILE = os.path.join(CACHE_DIR, 'team_pitching_cache.pkl')

def build_team_pitching_cache():
    print("Building Team Pitching cache...")
    
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    # 1. Get 30 MLB Teams
    r = session.get('https://statsapi.mlb.com/api/v1/teams?sportId=1')
    teams = {t['id']: t['name'] for t in r.json()['teams']}
    
    cache = {} # year -> team_name -> date_str -> {'ip', 'hr'}
    
    for year in [2024, 2025]:
        print(f"Fetching {year}...")
        cache[year] = {}
        for tid, tname in teams.items():
            cache[year][tname] = {}
            url = f"https://statsapi.mlb.com/api/v1/teams/{tid}/stats?stats=gameLog&group=pitching&season={year}"
            
            log_r = session.get(url).json()
            game_logs = []
            
            if 'stats' in log_r and len(log_r['stats']) > 0:
                for split in log_r['stats'][0].get('splits', []):
                    date_str = split['date']
                    hr = split['stat']['homeRuns']
                    ip_str = split['stat']['inningsPitched']
                    
                    try:
                        parts = str(ip_str).split('.')
                        full = float(parts[0])
                        frac = float(parts[1]) if len(parts) > 1 else 0.0
                        ip = full + (frac / 3.0)
                    except:
                        ip = 0.0
                        
                    game_logs.append({
                        'date': datetime.strptime(date_str, '%Y-%m-%d'),
                        'hr': hr,
                        'ip': ip
                    })
            
            if not game_logs:
                continue
            
            df = pd.DataFrame(game_logs)
            df = df.sort_values(by='date').reset_index(drop=True)
            df = df.groupby('date').sum().reset_index() # In case of doubleheaders
            
            # 2. Iterate all possible days in the season and compute 30-day trailing
            start_date = df['date'].min()
            end_date = df['date'].max()
            
            if pd.isna(start_date) or pd.isna(end_date):
                continue
                
            curr_date = start_date
            while curr_date <= end_date + timedelta(days=5):
                date_str = curr_date.strftime('%Y-%m-%d')
                
                # Trailing 30 days EXCLUDING curr_date
                mask = (df['date'] >= curr_date - timedelta(days=30)) & (df['date'] < curr_date)
                rolling_hr = df.loc[mask, 'hr'].sum()
                rolling_ip = df.loc[mask, 'ip'].sum()
                
                cache[year][tname][date_str] = {
                    'hr': float(rolling_hr),
                    'ip': float(rolling_ip)
                }
                
                curr_date += timedelta(days=1)
                
    with open(TEAM_PITCHING_CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Team Pitching cache built successfully!")

if __name__ == '__main__':
    build_team_pitching_cache()
