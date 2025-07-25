# Import library 
import datetime
import time
import random
import warnings
import logging 
import uuid
import pytz
import os
import pandas as pd
import io
import psycopg
import joblib
from prefect import task, flow
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

SEND_TIMEOUT = 10
rand = random.Random()

categorical_variables = ['PU_DO']
numerical_variables = ['trip_miles', 'is_weekend', 'fare_per_mile', 'hour', 'day_of_week']

@task(name="Prepare DataFrame")
# Define clean function for taxi data
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

@task(name="Feature Engineering")
# define function to engineer features
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["trip_start_timestamp"].dt.hour
    df["day_of_week"] = df["trip_start_timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"] >= 5

    df["fare"] = pd.to_numeric(df["fare"], errors="coerce")
    df["trip_total"] = pd.to_numeric(df["trip_total"], errors="coerce")

    df["PU_DO"] = df["pickup_community_area"].fillna("NA").astype(str) + "_" + df["dropoff_community_area"].fillna("NA").astype(str)

    df = df[df["trip_miles"] > 0]
    df = df[df["duration_minutes"] > 0]

    df["fare_per_mile"] = df["fare"] / df["trip_miles"]
    df["trip_speed"] = df["trip_miles"] / (df["duration_minutes"] / 60)

    df["fare_per_mile"] = df["fare_per_mile"].replace([float("inf"), -float("inf")], pd.NA)
    df["trip_speed"] = df["trip_speed"].replace([float("inf"), -float("inf")], pd.NA)

    df = df.dropna(subset=["fare_per_mile", "trip_speed"])

    return df


# Create the table statement
create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics (
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float
)
"""

reference_data = pd.read_parquet("data/reference.parquet")

def load_model_and_vectorizer(model_path, dv_path):
    """
    Load a trained ML model and its corresponding DictVectorizer from local paths.
    """
    model = joblib.load(model_path)
    dv = joblib.load(dv_path)
    
    return model, dv

# Define the paths to the model and DictVectorizer
model_path = 'model/model.pkl'
dv_path = 'model/dict_vectorizer.bin'

# Load the model and DictVectorizer
model, dv = load_model_and_vectorizer(model_path, dv_path)


# Read the whole dataset 
local_path = os.path.abspath(os.path.join(os.getcwd(), "..", "Dataset"))
feb_file_path = os.path.join(local_path, "chicago_taxi_2023_02.parquet")
raw_data  = pd.read_parquet(feb_file_path)

# Applied the data preprocessing
raw_data = clean_taxi_data(raw_data)
raw_data = engineer_features(raw_data)

begin = datetime.datetime(2023, 2, 1, 0, 0, 0)


column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=numerical_variables,
    categorical_features=categorical_variables,
    target=None
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

@task(name="Prepare PostgreSQL Database")
# Def a function to stored database
def prep_db():
    "Write a function to configure the database"
    with psycopg.connect(" host=localhost port =5433 user=postgres password =root", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname = 'postgres_db'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE postgres_db")
        with psycopg.connect("host=localhost  port =5433 dbname =postgres_db user=postgres password =root") as conn:
            conn.execute(create_table_statement)


# Def function to calculate dummy metrics
@task(name="Calculate Metrics PostgreSQL")
def calculate_metrics_postgresql(i):
    current_data = raw_data[(raw_data.trip_start_timestamp >= (begin + datetime.timedelta(i))) &
                            (raw_data.trip_start_timestamp < (begin + datetime.timedelta(i + 1)))]

    if current_data.empty:
        logging.info(f"Day {i}: no data, skipped.")
        return

    current_data.fillna(0, inplace=True)
    dicts = current_data[numerical_variables + categorical_variables].to_dict(orient='records')
    X = dv.transform(dicts)
    current_data['prediction'] = model.predict(X)
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    result = report.as_dict()
    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

    with psycopg.connect("host=localhost port=5433 dbname=postgres_db user=postgres password=root", autocommit=True) as conn:
        with conn.cursor() as curr:
            curr.execute(
                "INSERT INTO dummy_metrics (timestamp, prediction_drift, num_drifted_columns, share_missing_values) "
                "VALUES (%s, %s, %s, %s)",
                (begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values)
            )

    logging.info(f"Day {i}: data sent")


@flow(name="Batch Monitoring Backfill")
# Write a main function to insert timestamp values into the database
def batch_monitoring_backfill():
    prep_db()
    for i in range(31):
        calculate_metrics_postgresql(i)


if __name__ == '__main__':
    batch_monitoring_backfill()