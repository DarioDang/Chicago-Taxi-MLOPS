from typing import Callable, Dict, Tuple, Union

from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.utils.s3_logging import track_experiment_to_s3  


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def train(
    settings: Tuple[
        Dict[str, Union[bool, float, int, str]],               # hyperparameters
        csr_matrix,                                            # X (full feature matrix)
        Series,                                                # y (full target series)
        Dict[str, Union[Callable[..., BaseEstimator], str]],   # model_info
    ],
    **kwargs,
) -> Tuple[BaseEstimator, Dict[str, str]]:
    hyperparameters, X, y, model_info = settings

    model_class = model_info['cls']
    model_name = model_info.get('name', model_class.__name__)
    dv = model_info.get('dv', None)

    # Train the model on the full dataset
    model = model_class(**hyperparameters)
    model.fit(X, y)

    # Log to MLflow + Save to S3
    track_experiment_to_s3(
        model=model,
        dict_vectorizer=dv,
        hyperparameters=hyperparameters,
        training_set=X,
        training_targets=y,
        run_name=f"final_{model_name}",
        registered_model_name="randomforest-reg",   # <-- Set this only for RF
        register_stage="Production",                # <-- Promote only RF
        block_uuid=kwargs.get("block_uuid"),
        pipeline_uuid=kwargs.get("pipeline_uuid"),
        experiment_name="chicago-taxi-experiment-s3",  
        verbosity=True,
    )

    return model, model_info