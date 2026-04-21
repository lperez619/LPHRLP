import requests
import pandas as pd
import io
import json
import time

def build_rolling_savant_cache(year):
    cache = {} # str(date) -> dict of {player_id: {metrics}}
    # Find all distinct dates we care about from our dataset
    # Wait, we can just find all dates where games were played!
    
    print(f"Building Savant cache for {year}...")
    pass

build_rolling_savant_cache(2024)
