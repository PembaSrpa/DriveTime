# generate_data.py
"""
Generates synthetic training data for the DriveTime model.
Creates realistic trip scenarios with various conditions.
"""

import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of training samples
NUM_SAMPLES = 5000

print(f"Generating {NUM_SAMPLES} training samples...")

# Define possible values for categorical features
vehicle_types = ['car', 'truck', 'motorcycle', 'suv', 'van']
road_conditions = ['excellent', 'good', 'fair', 'poor']
weather_conditions = ['clear', 'rain', 'fog', 'snow', 'cloudy']

# Generate random data
data = []

for i in range(NUM_SAMPLES):
    # Random distance between 1 and 500 km
    distance_km = round(random.uniform(1, 500), 2)

    # Traffic hours (0-23.99, where decimal represents minutes)
    # e.g., 8.5 = 8:30 AM, 17.75 = 5:45 PM
    traffic_hours = round(random.uniform(0, 23.99), 2)

    # Vehicle average speed (40-120 km/h)
    vehicle_avg_speed = round(random.uniform(40, 120), 2)

    # Random categorical features
    vehicle_type = random.choice(vehicle_types)
    road_condition = random.choice(road_conditions)
    weather_condition = random.choice(weather_conditions)

    # Calculate BASE travel time (distance / speed = hours, * 60 = minutes)
    base_time = (distance_km / vehicle_avg_speed) * 60

    # Add realistic adjustments based on conditions

    # Traffic adjustment (rush hours: 7-9 AM and 5-7 PM cause delays)
    if 7 <= traffic_hours < 9 or 17 <= traffic_hours < 19:
        traffic_multiplier = random.uniform(1.3, 1.8)  # 30-80% longer
    elif 9 <= traffic_hours < 17:
        traffic_multiplier = random.uniform(1.0, 1.2)  # Normal to slightly busy
    else:
        traffic_multiplier = random.uniform(0.8, 1.0)  # Night time, faster

    # Road condition adjustment
    road_multipliers = {
        'excellent': random.uniform(0.9, 1.0),
        'good': random.uniform(1.0, 1.1),
        'fair': random.uniform(1.1, 1.3),
        'poor': random.uniform(1.3, 1.6)
    }
    road_multiplier = road_multipliers[road_condition]

    # Weather condition adjustment
    weather_multipliers = {
        'clear': random.uniform(0.95, 1.0),
        'cloudy': random.uniform(1.0, 1.05),
        'rain': random.uniform(1.2, 1.4),
        'fog': random.uniform(1.3, 1.5),
        'snow': random.uniform(1.4, 1.8)
    }
    weather_multiplier = weather_multipliers[weather_condition]

    # Vehicle type adjustment (trucks slower, motorcycles faster in traffic)
    vehicle_multipliers = {
        'car': 1.0,
        'suv': 1.05,
        'truck': 1.2,
        'van': 1.1,
        'motorcycle': 0.85
    }
    vehicle_multiplier = vehicle_multipliers[vehicle_type]

    # Calculate ACTUAL travel time with all adjustments
    actual_time_minutes = base_time * traffic_multiplier * road_multiplier * weather_multiplier * vehicle_multiplier

    # Add small random noise (real world is unpredictable)
    actual_time_minutes *= random.uniform(0.95, 1.05)
    actual_time_minutes = round(actual_time_minutes, 2)

    # Store the record
    data.append({
        'distance_km': distance_km,
        'traffic_hours': traffic_hours,
        'vehicle_avg_speed': vehicle_avg_speed,
        'vehicle_type': vehicle_type,
        'road_condition': road_condition,
        'weather_condition': weather_condition,
        'actual_time_minutes': actual_time_minutes
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/sample_data.csv', index=False)

print(f"✅ Generated {len(df)} samples")
print(f"✅ Saved to data/sample_data.csv")
print("\nSample data preview:")
print(df.head(10))
print("\nData statistics:")
print(df.describe())
