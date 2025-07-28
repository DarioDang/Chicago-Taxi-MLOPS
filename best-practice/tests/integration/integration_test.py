import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Code.aws_predict import prepare_features
import pytest
from Code.aws_predict import app
from unittest.mock import patch


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()




@patch("Code.aws_predict.model")
@patch("Code.aws_predict.dv")
@patch("Code.aws_predict.load_model_and_vectorizer")
def test_predict_endpoint_success(mock_loader_func, mock_dv, mock_model, client):
    # Arrange
    ride = {
        "trip_start_timestamp": "2023-02-15 08:30:00",
        "pickup_community_area": "8",
        "dropoff_community_area": "32",
        "fare": 20.5,
        "trip_total": 25.0,
        "trip_miles": 5.0,
    }

    # Mock behaviors
    mock_dv.transform.return_value = [[0.1, 0.2, 0.3]]
    mock_model.predict.return_value = [15.4]

    # Act
    response = client.post("/predict", json=ride)

    # Assert
    assert response.status_code == 200
    data = response.get_json()
    assert "duration" in data
    assert data["duration"] == 15.4
    assert data["model_id"] == "4d42b1b9f5c341c699fe72d680d49463"
