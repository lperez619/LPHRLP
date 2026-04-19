import requests, time

names = ['Bo Naylor', 'Kyle Tucker', 'Masyn Winn', 'Felix Reyes',
         'Leody Taveras', 'Jacob Wilson', 'Andrew Benintendi', 'Carson Kelly']

for name in names:
    url = 'https://statsapi.mlb.com/api/v1/people/search?names=' + name.replace(' ', '%20') + '&sportIds=1&active=true'
    r = requests.get(url, timeout=10)
    data = r.json()
    rows = data.get('people', [])
    if rows:
        p = rows[0]
        team = p.get('currentTeam', {}).get('name', '?')
        print(f"{name:<22} ID:{p['id']}  team:{team}")
    else:
        print(f"{name:<22} NOT FOUND in active MLB")
    time.sleep(0.3)
