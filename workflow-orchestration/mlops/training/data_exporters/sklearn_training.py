import os
import joblib
import boto3
from typing import Dict, Any, Tuple
import pandas as pd

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_wrapped_model_to_s3(
    training_set: Tuple,  # e.g., (X, y)
    hyperparameter_tuning: Dict[str, Any],
    *args,
    **kwargs,
):
    X, y = training_set[:2]
    wrapped_model = hyperparameter_tuning['wrapped_model']

    # ✅ Retrain the model on full dataset using best params
    X_transformed = wrapped_model.dv.transform(X.to_dict(orient="records"))
    wrapped_model.model.fit(X_transformed, y)

    # ✅ Save locally
    os.makedirs('/tmp/model', exist_ok=True)
    model_path = '/tmp/model/final_model.pkl'
    joblib.dump(wrapped_model, model_path)

    # ✅ Upload to S3
    s3 = boto3.client('s3')
    s3.upload_file(model_path, 'dario-mlflow-models-storage', 'models/final_model.pkl')

    print("✅ Final wrapped model trained on full data and saved to S3.")