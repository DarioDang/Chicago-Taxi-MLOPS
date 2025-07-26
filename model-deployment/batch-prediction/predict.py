# Import necessary libraries
import mlflow
import os
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient
from mlflow import artifacts



# Set MLflow tracking URI and run ID
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# Load MLflow model
MODEL_NAME = "randomforest-reg-v2"
STAGE = "Production"
logged_model = f"models:/{MODEL_NAME}/{STAGE}"
model = mlflow.pyfunc.load_model(logged_model)

# Get the run ID of the current production model
model_version = client.get_latest_versions(name=MODEL_NAME, stages=[STAGE])[0]
RUN_ID = model_version.run_id

# Download 'preprocessing/' folder from the run's artifacts
dv_folder_path = mlflow.artifacts.download_artifacts(
    run_id=RUN_ID,
    artifact_path="preprocessing"
)

# Build the full path to dict_vectorizer.bin inside the downloaded folder
dv_artifact_path = os.path.join(dv_folder_path, "dict_vectorizer.bin")

# Load the DictVectorizer
with open(dv_artifact_path, "rb") as f:
    dv = pickle.load(f)


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


def predict(features: pd.DataFrame) -> float:
    """
    Predict duration (or any target) using loaded model.
    """
    X = dv.transform(features.to_dict(orient='records'))
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