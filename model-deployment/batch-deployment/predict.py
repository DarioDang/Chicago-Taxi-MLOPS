from flask import Flask
from prefect import flow, task
import pandas as pd
import mlflow
from mlflow import artifacts
import pickle
import psycopg
import uuid
import random
import os
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from mlflow.tracking import MlflowClient

# Config
MODEL_NAME = "randomforest-reg-v2"
STAGE = "Production"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
DB_PARAMS = {
    "dbname": "batch_db",
    "user": "postgres",
    "password": "root",
    "host": "localhost",
    "port": "5432"
}

app = Flask(__name__)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


# Hypothetical frequency distribution from real data
pickup_areas = [8, 32, 76, 6, 28, 67, 71]
pickup_weights = [0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1]  # must sum to 1

dropoff_areas = [8, 32, 76, 6, 28, 67, 71]
dropoff_weights = [0.3, 0.25, 0.1, 0.1, 0.1, 0.1, 0.05]

@task(name="random_hour")
def random_hour():
    return random.choices(
        population=[7, 8, 9, 12, 17, 18, 22, 2],  # common hours
        weights=[0.15, 0.15, 0.1, 0.1, 0.2, 0.15, 0.1, 0.05],  # morning, lunch, evening
        k=1
    )[0]


@task(name="biased_timestamp")
def biased_timestamp():
    now = datetime.now()
    hour = random_hour()
    ts = now.replace(hour=hour, minute=random.randint(0, 59), second=0, microsecond=0)
    return ts.isoformat()

@task(name="generate_random_ride")
def generate_biased_ride():
    pickup = random.choices(pickup_areas, weights=pickup_weights, k=1)[0]
    dropoff = random.choices(dropoff_areas, weights=dropoff_weights, k=1)[0]

    trip_miles = round(random.uniform(1, 15), 2)
    fare_per_mile = random.uniform(2, 4)  # common Chicago range
    fare = round(trip_miles * fare_per_mile, 2)
    extra_fees = round(random.uniform(1, 5), 2)
    trip_total = round(fare + extra_fees, 2)

    return {
        "pickup_community_area": pickup,
        "dropoff_community_area": dropoff,
        "trip_start_timestamp": biased_timestamp(),
        "fare": fare,
        "trip_total": trip_total,
        "trip_miles": trip_miles
    }


@task(name="prepare_features")
def prepare_features(ride: dict) -> pd.DataFrame:
    df = pd.DataFrame([ride])

    if not pd.api.types.is_datetime64_any_dtype(df["trip_start_timestamp"]):
        df["trip_start_timestamp"] = pd.to_datetime(df["trip_start_timestamp"])

    df["hour"] = df["trip_start_timestamp"].dt.hour
    df["day_of_week"] = df["trip_start_timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"] >= 5

    df["fare"] = pd.to_numeric(df["fare"], errors="coerce")
    df["trip_total"] = pd.to_numeric(df["trip_total"], errors="coerce")

    df["PU_DO"] = df["pickup_community_area"].fillna("NA").astype(str) + "_" + df["dropoff_community_area"].fillna("NA").astype(str)

    df = df[df["trip_miles"] > 0]

    df["fare_per_mile"] = df["fare"] / df["trip_miles"]
    df["fare_per_mile"] = df["fare_per_mile"].replace([float("inf"), -float("inf")], pd.NA)

    selected_features = [
        'PU_DO',
        'trip_miles',
        'is_weekend',
        'fare_per_mile',
        'hour',
        'day_of_week',
    ]

    df = df.dropna(subset=selected_features)

    # Cap rare PU_DO combinations
    top_pudo = df['PU_DO'].value_counts().nlargest(1000).index
    df['PU_DO'] = df['PU_DO'].where(df['PU_DO'].isin(top_pudo), "Other")

    return df[selected_features]

@task(name="load_dict_vectorizer")
def load_dict_vectorizer(run_id):
    # Download 'preprocessing/' folder from the run's artifacts
    dv_folder_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="preprocessing"
    )

    # Build the full path to dict_vectorizer.bin inside the downloaded folder
    dv_artifact_path = os.path.join(dv_folder_path, "dict_vectorizer.bin")

    # Load and return the DictVectorizer
    with open(dv_artifact_path, "rb") as f:
        dv = pickle.load(f)
    return dv

@task(name="predict_duration")
def predict_duration(features):
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)

    run_id = client.get_latest_versions(MODEL_NAME, stages=[STAGE])[0].run_id
    dv = load_dict_vectorizer(run_id)

    X = dv.transform(features.to_dict(orient="records"))
    preds = model.predict(X)
    return round(preds[0], 2)

@task(name="insert_into_db")
def insert_into_db(data, prediction):
    trip_id = str(uuid.uuid4())
    conn = psycopg.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ride_predictions (
            id SERIAL PRIMARY KEY,
            trip_id TEXT UNIQUE,
            pickup_community_area INTEGER,
            dropoff_community_area INTEGER,
            trip_start_timestamp TIMESTAMP,
            fare FLOAT NOT NULL,
            trip_total FLOAT,
            trip_miles FLOAT,
            predicted_duration FLOAT,
            created_at TIMESTAMP
        )
    """)
    cur.execute(
        """
        INSERT INTO ride_predictions (
            trip_id,
            pickup_community_area, dropoff_community_area,
            trip_start_timestamp, fare, trip_total, trip_miles,
            predicted_duration, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            trip_id,
            data["pickup_community_area"],
            data["dropoff_community_area"],
            data["trip_start_timestamp"],
            data["fare"],
            data["trip_total"],
            data["trip_miles"],
            prediction,
            datetime.now()
        )
    )

    conn.commit()
    cur.close()
    conn.close()

@flow(name="scheduled_batch_prediction")
def scheduled_job():
    num_rides = random.randint(1, 10)
    print(f"[{datetime.now()}] Generating {num_rides} rides...")

    for i in range(num_rides):
        ride = generate_biased_ride()
        features = prepare_features(ride)

        if features.empty:
            print(f"Ride {i+1}: Skipped due to insufficient feature data")
            continue

        prediction = predict_duration(features)
        insert_into_db(ride, prediction)
        print(f"Ride {i+1}: Prediction saved: {prediction} mins")

# Schedule job
scheduler = BackgroundScheduler()
scheduler.add_job(func=lambda: scheduled_job(), trigger="interval", minutes=2, next_run_time=datetime.now())
scheduler.start()

@app.route("/")
def home():
    return "Batch prediction service running every 2 minutes."

if __name__ == "__main__":
    app.run(port=9696)