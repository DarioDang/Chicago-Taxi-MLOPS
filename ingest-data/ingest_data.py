#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from pathlib import Path
import argparse
import requests

# Command line arguments
parser = argparse.ArgumentParser(description="Download Chicago Taxi data and save to local path.")

# Arguments for year and month
parser.add_argument("--year", type=int, required=True, help="Year to download (e.g., 2023)")
parser.add_argument("--month", type=int, required=True, help="Month to download (1-12)")

args = parser.parse_args()

# Set the path to save the data
local_drive_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset"))
os.makedirs(local_drive_path, exist_ok=True)
print(f"Saving to folder: {local_drive_path}")

# Function to fetch and save Chicago Taxi data
def fetch_chicago_taxi_data(year: int, month: int, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_date = f"{year}-{month:02d}-01T00:00:00"
    if month == 12:
        end_date = f"{year + 1}-01-01T00:00:00"
    else:
        end_date = f"{year}-{month + 1:02d}-01T00:00:00"

    url = "https://data.cityofchicago.org/resource/wrvz-psew.json"
    where_clause = (
        f"trip_start_timestamp >= '{start_date}' AND trip_start_timestamp < '{end_date}'"
    )

    limit = 100000
    offset = 0
    all_data = []

    while True:
        params = {
            "$where": where_clause,
            "$limit": limit,
            "$offset": offset
        }

        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        all_data.extend(batch)
        offset += limit
        print(f"Fetched {offset} records...")

    df = pd.DataFrame(all_data)
    filename = os.path.join(output_dir, f"chicago_taxi_{year}_{month:02d}.parquet")
    df.to_parquet(filename, index=False)
    print(f"Saved {len(df)} records to {filename}")
    return filename

# Main execution
fetch_chicago_taxi_data(args.year, args.month, local_drive_path)

# bash 
# python ingest_data.py --year 2023 --month 3