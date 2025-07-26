import os
from typing import Dict, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import tempfile
from mlflow import MlflowClient
from mlflow.data import from_numpy, from_pandas
from mlflow.entities import DatasetInput, InputTag, Run
from mlflow.models import infer_signature
from mlflow.sklearn import log_model as log_model_sklearn
from mlflow.xgboost import log_model as log_model_xgboost
from sklearn.base import BaseEstimator
from mlflow.pyfunc import log_model as log_model_pyfunc



DEFAULT_DEVELOPER = os.getenv('EXPERIMENTS_DEVELOPER', 'Dario')
DEFAULT_EXPERIMENT_NAME = 'chicago-taxi-experiment'
DEFAULT_TRACKING_URI = 'sqlite:///mlflow.db'

# Set a custom artifact location (same as mlflow server config)
DEFAULT_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlartifacts")

def setup_experiment(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> Tuple[MlflowClient, str]:
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=DEFAULT_ARTIFACT_ROOT  
        )

    return client, experiment_id


def track_experiment_and_register(
    experiment_name: Optional[str] = None,
    block_uuid: Optional[str] = None,
    developer: Optional[str] = None,
    hyperparameters: Dict[str, Union[float, int, str]] = {},
    metrics: Dict[str, float] = {},
    model: Optional[Union[BaseEstimator, xgb.Booster]] = None,
    partition: Optional[str] = None,
    pipeline_uuid: Optional[str] = None,
    predictions: Optional[np.ndarray] = None,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    training_set: Optional[pd.DataFrame] = None,
    training_targets: Optional[pd.Series] = None,
    track_datasets: bool = False,
    validation_set: Optional[pd.DataFrame] = None,
    validation_targets: Optional[pd.Series] = None,
    verbosity: Union[bool, int] = False,
    dict_vectorizer: Optional[object] = None,
    **kwargs,
) -> Run:
    experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME
    tracking_uri = tracking_uri or DEFAULT_TRACKING_URI

    client, experiment_id = setup_experiment(experiment_name, tracking_uri)

    if not run_name:
        run_name = ':'.join(
            [str(s) for s in [pipeline_uuid, partition, block_uuid] if s]
        )

    run = client.create_run(experiment_id, run_name=run_name or None)
    run_id = run.info.run_id

    for key, value in [
        ('developer', developer or DEFAULT_DEVELOPER),
        ('model', model.__class__.__name__),
    ]:
        if value is not None:
            client.set_tag(run_id, key, value)

    for key, value in [
        ('block_uuid', block_uuid),
        ('partition', partition),
        ('pipeline_uuid', pipeline_uuid),
    ]:
        if value is not None:
            client.log_param(run_id, key, value)

    for key, value in hyperparameters.items():
        client.log_param(run_id, key, value)
        if verbosity:
            print(f'Logged hyperparameter {key}: {value}.')

    for key, value in metrics.items():
        client.log_metric(run_id, key, value)
        if verbosity:
            print(f'Logged metric {key}: {value}.')

    dataset_inputs = []
    if track_datasets:
        for dataset_name, dataset, tags in [
            ('dataset', training_set, dict(context='training')),
            ('targets', training_targets.to_numpy() if training_targets is not None else None, dict(context='training')),
            ('dataset', validation_set, dict(context='validation')),
            ('targets', validation_targets.to_numpy() if validation_targets is not None else None, dict(context='validation')),
            ('predictions', predictions, dict(context='training')),
        ]:
            if dataset is None:
                continue

            dataset_from = from_pandas if isinstance(dataset, pd.DataFrame) else from_numpy if isinstance(dataset, np.ndarray) else None

            if dataset_from:
                ds = dataset_from(dataset, name=dataset_name)._to_mlflow_entity()
                ds_input = DatasetInput(ds, tags=[InputTag(k, v) for k, v in tags.items()])
                dataset_inputs.append(ds_input)

        if len(dataset_inputs) >= 1:
            client.log_inputs(run_id, dataset_inputs)
    if model:
        with mlflow.start_run(run_id=run_id):
            # Save dict_vectorizer separately
            if dict_vectorizer is not None:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    vectorizer_path = os.path.join(tmp_dir, "dict_vectorizer.bin")
                    with open(vectorizer_path, "wb") as f:
                        pickle.dump(dict_vectorizer, f)
                    mlflow.log_artifact(vectorizer_path, artifact_path="preprocessing")
                    if verbosity:
                        print("Logged dict_vectorizer to preprocessing/dict_vectorizer.bin")

            # Log the model
            log_model = log_model_sklearn if isinstance(model, BaseEstimator) else log_model_xgboost
            input_example = training_set.head(1) if isinstance(training_set, pd.DataFrame) else None
            opts = dict(input_example=input_example)

            if training_set is not None and predictions is not None:
                try:
                    opts['signature'] = infer_signature(training_set, predictions)
                except Exception:
                    pass

            registered_model_name = kwargs.get("registered_model_name")

            # Register only if the model is RandomForestRegressor
            should_register = (
                registered_model_name is not None and
                model.__class__.__name__ == "RandomForestRegressor"
            )

            log_model(
                model,
                artifact_path='model',
                **({"registered_model_name": registered_model_name} if should_register else {}),
                **opts
            )

            if verbosity:
                print(f'Logged model: {model.__class__.__name__}')

            # Register and transition model stage
            if registered_model_name and model.__class__.__name__ == "RandomForestRegressor":
                client = MlflowClient()
                versions = client.get_latest_versions(registered_model_name, stages=[])
                latest_version = max([int(v.version) for v in versions], default=1)

                desired_stage = kwargs.get("register_stage", "None")  # e.g., "Production", "Staging"
                if desired_stage.lower() in ["production", "staging"]:
                    client.transition_model_version_stage(
                        name=registered_model_name,
                        version=str(latest_version),
                        stage=desired_stage.capitalize(),
                        archive_existing_versions=True
                    )
                    if verbosity:
                        print(f"Model {registered_model_name} transitioned to {desired_stage.capitalize()}")
    return run