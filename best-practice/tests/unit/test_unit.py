import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pandas as pd
import pytest
from Code.aws_predict import prepare_features
from unittest.mock import patch, MagicMock
from Code.aws_predict import predict


def test_prepare_features_valid_input():
    ride = {
        "trip_start_timestamp": "2023-02-15 08:30:00",
        "pickup_community_area": "8",
        "dropoff_community_area": "32",
        "fare": 20.5,
        "trip_total": 25.0,
        "trip_miles": 5.0,
    }

    features = prepare_features(ride)

    # Check that result is a DataFrame with exactly one row
    assert isinstance(features, pd.DataFrame)
    assert len(features) == 1

    # Check required columns
    expected_columns = [
        "PU_DO",
        "trip_miles",
        "is_weekend",
        "fare_per_mile",
        "hour",
        "day_of_week",
    ]
    assert list(features.columns) == expected_columns

    # Check specific values
    assert features["PU_DO"].iloc[0] == "8_32"
    assert features["trip_miles"].iloc[0] == 5.0
    assert features["hour"].iloc[0] == 8
    assert features["day_of_week"].iloc[0] == 2  # Wednesday
    assert features["is_weekend"].iloc[0] == False
    assert round(features["fare_per_mile"].iloc[0], 2) == round(20.5 / 5.0, 2)


def test_prepare_features_missing_timestamp():
    ride = {
        "pickup_community_area": "8",
        "dropoff_community_area": "32",
        "fare": 20.5,
        "trip_total": 25.0,
        "trip_miles": 5.0,
    }

    with pytest.raises(KeyError):
        prepare_features(ride)


def test_prepare_features_zero_trip_miles():
    ride = {
        "trip_start_timestamp": "2023-02-15 08:30:00",
        "pickup_community_area": "8",
        "dropoff_community_area": "32",
        "fare": 20.5,
        "trip_total": 25.0,
        "trip_miles": 0.0,
    }

    features = prepare_features(ride)
    assert features.empty


def test_prepare_features_inf_fare():
    ride = {
        "trip_start_timestamp": "2023-02-15 08:30:00",
        "pickup_community_area": "8",
        "dropoff_community_area": "32",
        "fare": float("inf"),
        "trip_total": 25.0,
        "trip_miles": 5.0,
    }

    features = prepare_features(ride)
    assert features.empty


@patch("Code.aws_predict.model")
@patch("Code.aws_predict.dv")
@patch("Code.aws_predict.load_model_and_vectorizer")
def test_predict_returns_float(mock_loader_func, mock_dv, mock_model):
    ride = {
        "trip_start_timestamp": "2023-02-15 08:30:00",
        "pickup_community_area": "8",
        "dropoff_community_area": "32",
        "fare": 20.5,
        "trip_total": 25.0,
        "trip_miles": 5.0,
    }

    features = prepare_features(ride)

    # Mock vectorizer and model behavior
    mock_dv.transform.return_value = [[0.1, 0.2, 0.3]]
    mock_model.predict.return_value = [15.23]

    result = predict(features)

    assert isinstance(result, float)
    assert result == 15.23




