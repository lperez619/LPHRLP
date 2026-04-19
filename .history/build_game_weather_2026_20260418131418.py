"""
Build game_weather_2026.pkl from MLB schedule + Open-Meteo archive API.

Format: {game_pk: {'temp_f': float, 'wind_mph': float, 'wind_dir': int, 'home_team': str}}
Only outdoor stadiums are included (matches existing 2023-2025 files).
"""

import os, pickle, time, requests
from datetime import datetime, timedelta

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STADIUMS = {
    'Arizona Diamondbacks': {'lat':33.4455,'lon':-112.0667,'indoor':True},
    'Atlanta Braves':       {'lat':33.8908,'lon': -84.4678,'indoor':False},
    'Baltimore Orioles':    {'lat':39.2839,'lon': -76.6218,'indoor':False},
    'Boston Red Sox':       {'lat':42.3467,'lon': -71.0972,'indoor':False},
    'Chicago Cubs':         {'lat':41.9484,'lon': -87.6553,'indoor':False},
    'Chicago White Sox':    {'lat':41.8299,'lon': -87.6338,'indoor':False},
    'Cincinnati Reds':      {'lat':39.0975,'lon': -84.5072,'indoor':False},
    'Cleveland Guardians':  {'lat':41.4962,'lon': -81.6852,'indoor':False},
    'Colorado Rockies':     {'lat':39.7560,'lon':-104.9942,'indoor':False},
    'Detroit Tigers':       {'lat':42.3390,'lon': -83.0485,'indoor':False},
    'Houston Astros':       {'lat':29.7572,'lon': -95.3558,'indoor':True},
    'Kansas City Royals':   {'lat':39.0517,'lon': -94.4803,'indoor':False},
    'Los Angeles Angels':   {'lat':33.8003,'lon':-117.8827,'indoor':False},
    'Los Angeles Dodgers':  {'lat':34.0739,'lon':-118.2400,'indoor':False},
    'Miami Marlins':        {'lat':25.7781,'lon': -80.2196,'indoor':True},
    'Milwaukee Brewers':    {'lat':43.0280,'lon': -87.9712,'indoor':True},
    'Minnesota Twins':      {'lat':44.9817,'lon': -93.2775,'indoor':False},
    'New York Mets':        {'lat':40.7571,'lon': -73.8458,'indoor':False},
    'New York Yankees':     {'lat':40.8296,'lon': -73.9262,'indoor':False},
    'Oakland Athletics':    {'lat':37.7516,'lon':-122.2005,'indoor':False},
    'Athletics':            {'lat':38.5905,'lon':-121.4997,'indoor':False},
    'Philadelphia Phillies':{'lat':39.9061,'lon': -75.1665,'indoor':False},
    'Pittsburgh Pirates':   {'lat':40.4469,'lon': -80.0058,'indoor':False},
    'San Diego Padres':     {'lat':32.7076,'lon':-117.1570,'indoor':False},
    'San Francisco Giants': {'lat':37.7786,'lon':-122.3893,'indoor':False},
    'Seattle Mariners':     {'lat':47.5914,'lon':-122.3323,'indoor':True},
    'St. Louis Cardinals':  {'lat':38.6226,'lon': -90.1928,'indoor':False},
    'Tampa Bay Rays':       {'lat':27.7683,'lon': -82.6534,'indoor':True},
    'Texas Rangers':        {'lat':32.7474,'lon': -97.0832,'indoor':True},
    'Toronto Blue Jays':    {'lat':43.6414,'lon': -79.3894,'indoor':True},
    'Washington Nationals': {'lat':38.8730,'lon': -77.0074,'indoor':False},
}

session = requests.Session()


def get_all_2026_games():
    """Fetch all 2026 regular-season games that have already been played."""
    start = '2026-03-26'
    # Only fetch up to yesterday (archive API needs completed days)
    end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f'Fetching 2026 schedule: {start} to {end}')

    games = []
    r = session.get('https://statsapi.mlb.com/api/v1/schedule',
                     params={'sportId': 1, 'startDate': start, 'endDate': end,
                             'gameType': 'R'},
                     timeout=30)
    r.raise_for_status()
    for date_entry in r.json().get('dates', []):
        for g in date_entry.get('games', []):
            status = g.get('status', {}).get('abstractGameState', '')
            if status != 'Final':
                continue
            home = g.get('teams', {}).get('home', {}).get('team', {}).get('name', '')
            games.append({
                'game_pk': g['gamePk'],
                'home_team': home,
                'game_date': g.get('gameDate', ''),
                'date_str': date_entry['date'],
            })
    print(f'  Found {len(games)} completed regular-season games')
    return games


def fetch_weather_for_games(games):
    """Fetch weather for all outdoor games, batching by date + coordinates."""
    results = {}
    # Group games by date
    by_date = {}
    for g in games:
        by_date.setdefault(g['date_str'], []).append(g)

    coord_cache = {}  # (date, lat, lon) -> hourly data
    skipped_indoor = 0

    for date_str in sorted(by_date):
        date_games = by_date[date_str]
        for g in date_games:
            stadium = STADIUMS.get(g['home_team'], {})
            if stadium.get('indoor', True):
                skipped_indoor += 1
                continue

            lat, lon = stadium['lat'], stadium['lon']
            cache_key = (date_str, lat, lon)

            if cache_key not in coord_cache:
                try:
                    r = session.get('https://archive-api.open-meteo.com/v1/archive',
                                    params={
                                        'latitude': lat, 'longitude': lon,
                                        'hourly': 'temperature_2m,wind_speed_10m,wind_direction_10m',
                                        'temperature_unit': 'fahrenheit',
                                        'wind_speed_unit': 'mph',
                                        'timezone': 'UTC',
                                        'start_date': date_str,
                                        'end_date': date_str,
                                    }, timeout=15)
                    r.raise_for_status()
                    h = r.json().get('hourly', {})
                    coord_cache[cache_key] = {
                        'temps': h.get('temperature_2m', []),
                        'winds': h.get('wind_speed_10m', []),
                        'dirs':  h.get('wind_direction_10m', []),
                    }
                    time.sleep(0.15)  # rate limit
                except Exception as e:
                    print(f'  Weather error {g["home_team"]} {date_str}: {e}')
                    coord_cache[cache_key] = {}

            hourly = coord_cache.get(cache_key, {})
            if hourly and hourly.get('temps'):
                # Use game start hour from game_date (UTC), fallback to 23 (7 PM EDT)
                try:
                    dt = datetime.fromisoformat(g['game_date'].replace('Z', '+00:00'))
                    idx = min(dt.hour, 23)
                except Exception:
                    idx = 23

                results[g['game_pk']] = {
                    'temp_f':    hourly['temps'][idx],
                    'wind_mph':  hourly['winds'][idx],
                    'wind_dir':  int(hourly['dirs'][idx]),
                    'home_team': g['home_team'],
                }

        print(f'  {date_str}: {len([g for g in date_games if g["game_pk"] in results])} outdoor games fetched')

    print(f'\nTotal: {len(results)} outdoor games with weather')
    print(f'Skipped {skipped_indoor} indoor games')
    return results


def main():
    games = get_all_2026_games()
    if not games:
        print('No completed 2026 games found!')
        return

    weather = fetch_weather_for_games(games)
    out_path = os.path.join(_SCRIPT_DIR, 'game_weather_2026.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(weather, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nSaved → {out_path} ({len(weather)} entries)')


if __name__ == '__main__':
    main()
