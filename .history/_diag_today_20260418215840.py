"""Diagnose today's picks vs actual HRs."""
import sys, os
_trainer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HR Trainer.py')
_trainer_globals = {'__name__': '_hr_trainer_lib', '__file__': _trainer_path}
with open(_trainer_path, 'r', encoding='utf-8') as f:
    src = f.read()
pos = src.find('\nTEST_DATES')
exec(compile(src[:pos], _trainer_path, 'exec'), _trainer_globals)

get_actual_hrs = _trainer_globals['get_actual_hrs']
load_cache = _trainer_globals['load_cache']

DATE = '2026-04-18'
actual = get_actual_hrs(DATE)
print(f'Total HR hitters on {DATE}: {len(actual)}')
total_hrs = sum(v.get('hrs', 0) for v in actual.values())
print(f'Total HRs: {total_hrs}')
print()

for pid, info in sorted(actual.items(), key=lambda x: -x[1].get('hrs', 0)):
    name = info.get('name', str(pid))
    hrs = info.get('hrs', 0)
    print(f'  {name:<28} {hrs} HR')

# Our picks from today
picks = [
    'Matt Olson', 'Fernando Tatis Jr.', 'Gunnar Henderson', 'Nolan Schanuel',
    'Yandy Diaz', 'Aaron Judge', 'Yordan Alvarez', 'Kyle Schwarber',
    'Jordan Walker', 'Lourdes Gurriel Jr.',
]
pick_names_lower = set(p.lower() for p in picks)
actual_names_lower = {info.get('name', '').lower(): info for info in actual.values()}

print()
print('=== PICK ANALYSIS ===')
for p in picks:
    hit = p.lower() in actual_names_lower
    mark = 'HIT' if hit else 'MISS'
    print(f'  {mark:4}  {p}')

# Check where actual HR hitters ranked in our model
print()
print('=== WHERE DID ACTUAL HR HITTERS RANK? ===')
# Load today's model output by re-running scoring mentally
# Actually let's check if any of our top picks' games are still in progress
print()
print('=== GAME STATUS CHECK ===')
session = _trainer_globals['session']
r = session.get('https://statsapi.mlb.com/api/v1/schedule/games/',
    params={'sportId': 1, 'date': DATE, 'hydrate': 'linescore'},
    timeout=15)
r.raise_for_status()
for de in r.json().get('dates', []):
    for g in de.get('games', []):
        status = g.get('status', {}).get('detailedState', '?')
        home = g['teams']['home']['team']['name']
        away = g['teams']['away']['team']['name']
        ls = g.get('linescore', {})
        inn = ls.get('currentInning', '?')
        half = ls.get('inningHalf', '')
        print(f'  {away:<28} @ {home:<28} {status:<20} Inn: {half} {inn}')
