import pandas as pd
from typing import Tuple


# Import utility functions (folder name is 'data_preparation')
from mlops.utils.data_preparation.cleaning import clean_taxi_data
from mlops.utils.data_preparation.feature_engineering import engineer_features
from mlops.utils.data_preparation.feature_selector import select_features
from mlops.utils.data_preparation.splitters import split_on_value

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applies data cleaning, feature engineering, feature selection, and splitting.

    Parameters (from kwargs):
    - split_on_feature: column to split on (e.g. 'trip_start_timestamp')
    - split_on_feature_value: value to split by (e.g. '2023-01-15T00:00:00')
    - target: target column name (e.g. 'duration_minutes')
    """
    # Retrieve configurable parameters
    split_on_feature = kwargs.get('split_on_feature', 'trip_start_timestamp')
    split_on_feature_value = kwargs.get('split_on_feature_value', '2023-01-15T00:00:00')
    target = kwargs.get('target', 'duration_minutes')

    # Step 1: Clean raw data
    df = clean_taxi_data(df)

    # Step 2: Feature engineering
    df = engineer_features(df)

    # Step 3: Feature selection (include target + splitting feature)
    df = select_features(df, features=[split_on_feature, target])

    # Step 4: Split into train and validation sets
    df_train, df_val = split_on_value(
        df,
        feature=split_on_feature,
        value=split_on_feature_value,
        drop_feature=True
    )

    return df, df_train, df_val