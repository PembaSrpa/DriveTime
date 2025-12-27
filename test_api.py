# test_api.py
import requests
import json

BASE_URL = "http://localhost:8000"

# Test 1: Health check
print("Testing health endpoint...")
response = requests.get(f"{BASE_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

# Test 2: Prediction
print("Testing prediction endpoint...")
test_data = {
    "distance_km": 200,
    "traffic_hours": 8.5,
    "vehicle_avg_speed": 70,
    "vehicle_type": "car",
    "road_condition": "good",
    "weather_condition": "clear"
}

response = requests.post(f"{BASE_URL}/predict", json=test_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

# Test 3: Different scenario
print("Testing with rain and poor roads...")
test_data_rain = {
    "distance_km": 200,
    "traffic_hours": 17.5,
    "vehicle_avg_speed": 60,
    "vehicle_type": "bus",
    "road_condition": "poor",
    "weather_condition": "rain"
}

response = requests.post(f"{BASE_URL}/predict", json=test_data_rain)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

print("✅ All tests completed!")
