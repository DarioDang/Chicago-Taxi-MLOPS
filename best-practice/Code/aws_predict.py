# Import necessary libraries
import mlflow
import pandas as pd
import boto3
import os
import pickle
from flask import Flask, request, jsonify

# Constants
RUN_ID = '4d42b1b9f5c341c699fe72d680d49463'
S3_BUCKET = 'dario-mlflow-models-storage'
MODEL_PATH = f's3://{S3_BUCKET}/{RUN_ID}/artifacts/model'
DV_S3_KEY = f'{RUN_ID}/artifacts/preprocessing/dict_vectorizer.bin'
LOCAL_DV_PATH = 'dict_vectorizer.bin'

# Globals (used after lazy loading)
model = None
dv = None

def load_model_and_vectorizer():
    """
    Loads the model and vectorizer from S3 or local cache if not already loaded.
    Ensures this runs only once (lazy loading).
    """
    global model, dv

    if model is None:
        model = mlflow.pyfunc.load_model(MODEL_PATH)

    if dv is None:
        if not os.path.exists(LOCAL_DV_PATH):
            print(f"Downloading vectorizer from s3://{S3_BUCKET}/{DV_S3_KEY}")
            s3 = boto3.client('s3')
            s3.download_file(S3_BUCKET, DV_S3_KEY, LOCAL_DV_PATH)

        with open(LOCAL_DV_PATH, 'rb') as f_in:
            dv = pickle.load(f_in)


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

    top_pudo = df['PU_DO'].value_counts().nlargest(1000).index
    df['PU_DO'] = df['PU_DO'].where(df['PU_DO'].isin(top_pudo), "Other")

    return df[selected_features]


def predict(features: pd.DataFrame) -> float:
    """
    Predict duration using the model and vectorizer.
    """
    load_model_and_vectorizer()
    X = dv.transform(features.to_dict(orient="records"))
    prediction = model.predict(X)
    return round(prediction[0], 2)


# Flask app
app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)

    if features.empty:
        return jsonify({'error': 'Invalid input, check values.'}), 400

    duration = predict(features)

    result = {
        'duration': duration,
        'model_id': RUN_ID
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
