import requests

# Example input payload (match the schema expected by prepare_features)
test_ride = {
    "trip_start_timestamp": "2023-07-14T08:30:00",
    "pickup_community_area": 78,
    "dropoff_community_area": 52,
    "trip_miles": 15.2,
    "fare": 48.5,
    "trip_total": 50.1
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json = test_ride)
print(response.json())
