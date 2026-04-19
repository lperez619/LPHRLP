import requests, time

names = ['Bo Naylor', 'Kyle Tucker', 'Masyn Winn', 'Felix Reyes',
         'Leody Taveras', 'Jacob Wilson', 'Andrew Benintendi', 'Carson Kelly']

ids = {}
for name in names:
    url = 'https://statsapi.mlb.com/api/v1/people/search?names=' + name.replace(' ', '%20') + '&sportIds=1&active=true'
    r = requests.get(url, timeout=10)
    data = r.json()
    rows = data.get('people', [])
    if rows:
        p = rows[0]
        pid = p['id']
        ids[name] = pid
        team = p.get('currentTeam', {}).get('name', '?')
        print(f"{name:<22} ID:{pid}  team:{team}")
    else:
        print(f"{name:<22} NOT FOUND in active MLB")
    time.sleep(0.3)

# Now check today's games and lineups
print('\n--- Checking April 18 games ---')
url = 'https://statsapi.mlb.com/api/v1/schedule?date=2026-04-18&sportId=1&hydrate=lineups,probablePitcher'
r = requests.get(url, timeout=15)
sched = r.json()

pid_to_name = {v: k for k, v in ids.items()}
target_ids = set(ids.values())

for d in sched.get('dates', []):
    for g in d.get('games', []):
        away = g['teams']['away']['team']['name']
        home = g['teams']['home']['team']['name']
        # Get lineup IDs
        away_lineup = [p['id'] for p in g.get('lineups', {}).get('awayPlayers', [])]
        home_lineup = [p['id'] for p in g.get('lineups', {}).get('homePlayers', [])]
        found_in_game = target_ids & (set(away_lineup) | set(home_lineup))
        if found_in_game:
            for pid in found_in_game:
                side = 'away' if pid in away_lineup else 'home'
                pos = (away_lineup if side == 'away' else home_lineup).index(pid) + 1
                print(f"  FOUND: {pid_to_name.get(pid, pid):<22} in {away} @ {home} ({side} #{pos})")

        # Also check rosters
        for side_key, side_name, lineup in [('away', away, away_lineup), ('home', home, home_lineup)]:
            roster_ids = set()
            try:
                tid = g['teams'][side_key]['team']['id']
                rurl = f'https://statsapi.mlb.com/api/v1/teams/{tid}/roster/active'
                rr = requests.get(rurl, timeout=10)
                for p in rr.json().get('roster', []):
                    roster_ids.add(p['person']['id'])
            except:
                pass
            on_roster = target_ids & roster_ids
            not_in_lineup = on_roster - set(lineup)
            for pid in not_in_lineup:
                print(f"  ON ROSTER but NOT in lineup: {pid_to_name.get(pid, pid):<22} {side_name}")
            time.sleep(0.2)

# Check if any target player's team is NOT playing today
print('\n--- Checking if players are on teams not playing ---')
for name, pid in ids.items():
    url = f'https://statsapi.mlb.com/api/v1/people/{pid}?hydrate=currentTeam'
    r = requests.get(url, timeout=10)
    data = r.json()
    people = data.get('people', [])
    if people:
        team = people[0].get('currentTeam', {}).get('name', 'UNKNOWN')
        print(f"  {name:<22} -> {team}")
    time.sleep(0.2)
