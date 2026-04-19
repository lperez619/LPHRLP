import requests, time

# Quick check: count how many batters per game the lineup hydration returns
print('--- Lineup sizes per game on April 18 ---')
url = 'https://statsapi.mlb.com/api/v1/schedule?date=2026-04-18&sportId=1&hydrate=lineups,probablePitcher,team'
r = requests.get(url, timeout=15)
sched = r.json()

for d in sched.get('dates', []):
    for g in d.get('games', []):
        away = g['teams']['away']['team']['name']
        home = g['teams']['home']['team']['name']
        lin = g.get('lineups', {})
        hl = [p['id'] for p in lin.get('homePlayers', [])]
        al = [p['id'] for p in lin.get('awayPlayers', [])]
        print(f"  {away:<30} ({len(al)}) @ {home:<30} ({len(hl)})  total={len(al)+len(hl)}")
