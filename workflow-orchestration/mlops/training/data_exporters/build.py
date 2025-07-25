import nest_asyncio
nest_asyncio.apply()

from mlops.utils.data_preparation.encoders import encode_features
from mlops.utils.data_preparation.feature_selector import select_features

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from pandas import DataFrame, Series
from typing import Tuple
from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix


@data_exporter
def export(
    data: Tuple[DataFrame, DataFrame, DataFrame],
    *args, **kwargs
) -> Tuple[
    csr_matrix,
    csr_matrix,
    csr_matrix,
    Series,
    Series,
    Series,
    BaseEstimator,
]:
    df, df_train, df_val = data
    target = kwargs.get('target', 'duration_minutes')

    # Full dataset
    X, _, _ = encode_features(select_features(df))
    y = df[target]

    # Train/val sets
    X_train, X_val, dv = encode_features(
        select_features(df_train),
        select_features(df_val),
    )
    y_train = df_train[target]
    y_val = df_val[target]

    return X, X_train, X_val, y, y_train, y_val, dv


@test
def test_dataset(
    X: csr_matrix,
    X_train: csr_matrix,
    X_val: csr_matrix,
    y: Series,
    y_train: Series,
    y_val: Series,
    *args,
) -> None:
    # Check full dataset shape
    assert (
        X.shape[0] == 369140
    ), f"Expected 369140 rows, got {X.shape[0]}"
    assert (
        X.shape[1] == 1006
    ), f"Expected 1007 columns, got {X.shape[1]}"
    assert (
        len(y.index) == X.shape[0]
    ), f"Mismatch: X rows = {X.shape[0]}, y rows = {len(y)}"



@test
def test_training_set(
    X: csr_matrix,
    X_train: csr_matrix,
    X_val: csr_matrix,
    y: Series,
    y_train: Series,
    y_val: Series,
    *args,
) -> None:
    # Check shape of training data
    assert (
        X_train.shape[0] == 159581
    ), f"Expected 159581 training rows, got {X_train.shape[0]}"
    
    assert (
        X_train.shape[1] == 1006
    ), f"Expected 1007 training columns, got {X_train.shape[1]}"
    
    assert (
        len(y_train.index) == X_train.shape[0]
    ), f"Mismatch: X_train rows = {X_train.shape[0]}, y_train rows = {len(y_train)}"

@test
def test_validation_set(
    X: csr_matrix,
    X_train: csr_matrix,
    X_val: csr_matrix,
    y: Series,
    y_train: Series,
    y_val: Series,
    *args,
) -> None:
    # Check shape of validation data
    assert (
        X_val.shape[0] == 209559
    ), f"Expected 209559 validation rows, got {X_val.shape[0]}"
    
    assert (
        X_val.shape[1] == 1006
    ), f"Expected 1007 validation columns, got {X_val.shape[1]}"
    
    assert (
        len(y_val) == X_val.shape[0]
    ), f"Mismatch: X_val rows = {X_val.shape[0]}, y_val rows = {len(y_val)}"