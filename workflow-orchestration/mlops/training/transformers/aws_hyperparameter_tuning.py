from typing import Callable, Dict, Tuple, Union

import pandas as pd
from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.utils.models.sklearn import load_class, tune_hyperparameters
from mlops.utils.s3_logging import track_experiment_to_s3

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def hyperparameter_tuning(
    model_class_name: Union[str, Dict],
    training_set: Dict[str, list],
    *args,
    **kwargs,
) -> Tuple[
    Dict[str, Union[bool, float, int, str]],
    csr_matrix,
    Series,
    Dict[str, Union[Callable[..., BaseEstimator], str]],
]:
    # Unpack training data
    build = training_set.get("build")
    if not isinstance(build, list) or len(build) < 7:
        raise ValueError("training_set['build'] must be a list of at least 7 elements")

    X, X_train, X_val, y, y_train, y_val, dv = build[:7]

    # Clean labels
    y_train = pd.to_numeric(y_train, errors="coerce")
    y_val = pd.to_numeric(y_val, errors="coerce")
    X_train = X_train[y_train.notna()]
    y_train = y_train[y_train.notna()]
    X_val = X_val[y_val.notna()]
    y_val = y_val[y_val.notna()]

    results = []

    model_names, _ = model_class_name  # Unpack both model list and metadata

    for name in model_names:
        model_class = load_class(name)
        best_params, best_rmse = tune_hyperparameters(
            model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            max_evaluations=kwargs.get('max_evaluations', 50),
            random_state=kwargs.get('random_state', 42),
        )

        model = model_class(**best_params)
        model.fit(X_train, y_train)

        track_experiment_to_s3(
            model=model,
            dict_vectorizer=dv,
            hyperparameters=best_params,
            metrics={'rmse': best_rmse},
            training_set=X_train,
            training_targets=y_train,
            validation_set=X_val,
            validation_targets=y_val,
            pipeline_uuid=kwargs.get('pipeline_uuid'),
            block_uuid=kwargs.get('block_uuid'),
            run_name=f"tuning_{name}",
        )

        results.append((best_params, name, best_rmse))

    return results, X, y, {'dv': dv}