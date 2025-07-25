from typing import Callable, Dict, Tuple, Union

import pandas as pd
from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.utils.models.sklearn import load_class, tune_hyperparameters

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

    X, X_train, X_val, y, y_train, y_val, _ = build[:7]

    # Clean labels
    y_train = pd.to_numeric(y_train, errors="coerce")
    y_val = pd.to_numeric(y_val, errors="coerce")
    X_train = X_train[y_train.notna()]
    y_train = y_train[y_train.notna()]
    X_val = X_val[y_val.notna()]
    y_val = y_val[y_val.notna()]

    # Load model from input (not looped)
    model_class = load_class(model_class_name)

    # Tune the model
    best_params, best_rmse = tune_hyperparameters(
        model_class,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_evaluations=kwargs.get('max_evaluations', 50),
        random_state=kwargs.get('random_state', 42),
    )

    print(f"✅ {model_class_name} best RMSE: {best_rmse:.4f}")

    return best_params, X, y, {'cls': model_class, 'name': model_class_name}