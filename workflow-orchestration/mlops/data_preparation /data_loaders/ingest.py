import io
import pandas as pd
import requests
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from typing import Dict
import os
from pathlib import Path
from datetime import datetime


@data_loader
def load_data_from_api(*args, **kwargs):
    """
    Download Chicago Taxi data from API for a given year and month
    and return it as a DataFrame.
    
    kwargs expects:
      - year: int
      - month: int
    """
    year = kwargs.get('year', 2023)
    month = kwargs.get('month', 1)

    # Build output path
    output_dir = os.path.expanduser("~/OneDrive/Data/ChicagoTaxi")
    os.makedirs(output_dir, exist_ok=True)

    start_date = f"{year}-{month:02d}-01T00:00:00"
    end_date = (
        f"{year + 1}-01-01T00:00:00" if month == 12
        else f"{year}-{month + 1:02d}-01T00:00:00"
    )

    url = "https://data.cityofchicago.org/resource/wrvz-psew.json"
    where_clause = f"trip_start_timestamp >= '{start_date}' AND trip_start_timestamp < '{end_date}'"

    limit = 100000
    offset = 0
    all_data = []

    while True:
        params = {
            "$where": where_clause,
            "$limit": limit,
            "$offset": offset
        }

        response = requests.get(url, params=params, timeout=300)
        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        all_data.extend(batch)
        offset += limit
        print(f"Fetched {offset} records...")

    df = pd.DataFrame(all_data)

    # Save a local copy as backup (optional)
    output_path = os.path.join(output_dir, f"chicago_taxi_{year}_{month:02d}.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")

    return df