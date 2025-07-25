import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Extract time features
    df["hour"] = df["trip_start_timestamp"].dt.hour
    df["day_of_week"] = df["trip_start_timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"] >= 5

    df["fare"] = pd.to_numeric(df["fare"], errors="coerce")
    df["trip_total"] = pd.to_numeric(df["trip_total"], errors="coerce")

    # Combine features
    df["PU_DO"] = df["pickup_community_area"].fillna("NA").astype(str) + "_" + df["dropoff_community_area"].fillna("NA").astype(str)

    # Filter invalid rows before computing derived features
    df = df[df["trip_miles"] > 0]
    df = df[df["duration_minutes"] > 0]

    # Compute derived features
    df["fare_per_mile"] = df["fare"] / df["trip_miles"]
    df["trip_speed"] = df["trip_miles"] / (df["duration_minutes"] / 60)

    # Handle infinite values
    df["fare_per_mile"] = df["fare_per_mile"].replace([float("inf"), -float("inf")], pd.NA)
    df["trip_speed"] = df["trip_speed"].replace([float("inf"), -float("inf")], pd.NA)

    return df.dropna(subset=["fare_per_mile", "trip_speed"])