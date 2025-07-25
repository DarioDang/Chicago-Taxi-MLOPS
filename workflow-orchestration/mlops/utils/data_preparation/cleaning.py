import pandas as pd

def clean_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    df["trip_start_timestamp"] = pd.to_datetime(df["trip_start_timestamp"], errors="coerce")
    df["trip_end_timestamp"] = pd.to_datetime(df["trip_end_timestamp"], errors="coerce")

    df["trip_seconds"] = pd.to_numeric(df["trip_seconds"], errors="coerce")
    df["duration_minutes"] = df["trip_seconds"] / 60

    df["trip_miles"] = pd.to_numeric(df["trip_miles"], errors="coerce")
    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")

    df = df.dropna(subset=["duration_minutes", "trip_miles"])
    df = df[df["duration_minutes"] > 0]

    return df