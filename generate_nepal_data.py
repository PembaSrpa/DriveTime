# generate_nepal_data.py
"""
Generates Nepal-specific synthetic training data for DriveTime.
Uses real Nepal routes, cities, road conditions, and traffic patterns.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

NUM_SAMPLES = 5000

print(f"Generating {NUM_SAMPLES} Nepal-specific training samples...")

# Real Nepal routes with approximate distances (km)
nepal_routes = [
    # Major highways
    {'from': 'Kathmandu', 'to': 'Pokhara', 'distance': 200, 'road_type': 'highway'},
    {'from': 'Kathmandu', 'to': 'Chitwan', 'distance': 146, 'road_type': 'highway'},
    {'from': 'Kathmandu', 'to': 'Dharan', 'distance': 420, 'road_type': 'highway'},
    {'from': 'Kathmandu', 'to': 'Bhairahawa', 'distance': 275, 'road_type': 'highway'},
    {'from': 'Kathmandu', 'to': 'Birgunj', 'distance': 135, 'road_type': 'highway'},
    {'from': 'Pokhara', 'to': 'Butwal', 'distance': 125, 'road_type': 'highway'},
    {'from': 'Dharan', 'to': 'Biratnagar', 'distance': 35, 'road_type': 'highway'},

    # Hill/mountain roads
    {'from': 'Kathmandu', 'to': 'Nagarkot', 'distance': 32, 'road_type': 'hill'},
    {'from': 'Kathmandu', 'to': 'Dhulikhel', 'distance': 30, 'road_type': 'hill'},
    {'from': 'Pokhara', 'to': 'Sarangkot', 'distance': 13, 'road_type': 'hill'},
    {'from': 'Dharan', 'to': 'Dhankuta', 'distance': 42, 'road_type': 'hill'},

    # City routes
    {'from': 'Kathmandu', 'to': 'Bhaktapur', 'distance': 13, 'road_type': 'city'},
    {'from': 'Kathmandu', 'to': 'Patan', 'distance': 7, 'road_type': 'city'},
    {'from': 'Pokhara', 'to': 'Lekhnath', 'distance': 15, 'road_type': 'city'},
    {'from': 'Biratnagar', 'to': 'Itahari', 'distance': 8, 'road_type': 'city'},

    # Terai (plains) roads
    {'from': 'Birgunj', 'to': 'Hetauda', 'distance': 65, 'road_type': 'terai'},
    {'from': 'Bhairahawa', 'to': 'Butwal', 'distance': 22, 'road_type': 'terai'},
    {'from': 'Biratnagar', 'to': 'Dharan', 'distance': 35, 'road_type': 'terai'},
]

# Nepal-specific vehicle types
vehicle_types = [
    'car',           # Private cars
    'microbus',      # Common public transport
    'bus',           # Tourist/long distance buses
    'motorcycle',    # Very common in Nepal
    'scooter',       # City transport
    'jeep',          # Hill/mountain areas
    'truck',         # Cargo
]

# Nepal road conditions (realistic distribution)
road_conditions = ['excellent', 'good', 'fair', 'poor']
road_condition_weights = [0.15, 0.35, 0.35, 0.15]  # Most roads are good/fair

# Nepal weather conditions
weather_conditions = [
    'clear',         # Dry season
    'cloudy',        # Common
    'rain',          # Monsoon (June-September)
    'fog',           # Winter mornings in Terai/valleys
    'dust',          # Dry season on unpaved roads
]

# Generate data
data = []

for i in range(NUM_SAMPLES):
    # Select random route
    route = random.choice(nepal_routes)
    distance_km = route['distance']
    road_type = route['road_type']

    # Add ±10% variation to distance (different routes, detours)
    distance_km = round(distance_km * random.uniform(0.9, 1.1), 2)

    # Traffic hours (0-23.99)
    traffic_hours = round(random.uniform(0, 23.99), 2)

    # Vehicle speed based on road type and vehicle
    vehicle_type = random.choice(vehicle_types)

    # Base speeds by road type (km/h)
    if road_type == 'highway':
        base_speed = random.uniform(50, 80)
    elif road_type == 'hill':
        base_speed = random.uniform(25, 45)  # Slower in hills
    elif road_type == 'city':
        base_speed = random.uniform(15, 35)  # City traffic
    else:  # terai
        base_speed = random.uniform(45, 70)

    # Adjust speed by vehicle type
    vehicle_speed_factors = {
        'car': 1.0,
        'microbus': 0.85,
        'bus': 0.8,
        'motorcycle': 1.1,
        'scooter': 0.7,
        'jeep': 0.9,
        'truck': 0.75,
    }
    vehicle_avg_speed = round(base_speed * vehicle_speed_factors[vehicle_type], 2)

    # Road condition (weighted random)
    road_condition = random.choices(road_conditions, road_condition_weights)[0]

    # Weather (seasonal patterns)
    # Monsoon months (June-Sept) have more rain
    month = random.randint(1, 12)
    if 6 <= month <= 9:  # Monsoon
        weather_condition = random.choices(
            weather_conditions,
            weights=[0.2, 0.3, 0.4, 0.05, 0.05]
        )[0]
    elif month in [12, 1, 2]:  # Winter (fog in Terai/valleys)
        weather_condition = random.choices(
            weather_conditions,
            weights=[0.5, 0.2, 0.05, 0.2, 0.05]
        )[0]
    else:  # Dry season
        weather_condition = random.choices(
            weather_conditions,
            weights=[0.6, 0.2, 0.05, 0.05, 0.1]
        )[0]

    # Calculate BASE travel time
    base_time = (distance_km / vehicle_avg_speed) * 60

    # NEPAL-SPECIFIC ADJUSTMENTS

    # 1. Traffic adjustment (Nepal rush hours: 7-10 AM, 4-7 PM)
    if 7 <= traffic_hours < 10 or 16 <= traffic_hours < 19:
        if road_type == 'city':
            traffic_multiplier = random.uniform(1.5, 2.2)  # Heavy city traffic
        else:
            traffic_multiplier = random.uniform(1.2, 1.5)
    elif road_type == 'city' and 10 <= traffic_hours < 16:
        traffic_multiplier = random.uniform(1.1, 1.3)
    else:
        traffic_multiplier = random.uniform(0.9, 1.1)

    # 2. Festival season delays (Dashain, Tihar - assume 10% of samples)
    if random.random() < 0.1:
        traffic_multiplier *= random.uniform(1.3, 1.8)  # Festival traffic

    # 3. Road condition multiplier
    road_multipliers = {
        'excellent': random.uniform(0.95, 1.0),
        'good': random.uniform(1.0, 1.15),
        'fair': random.uniform(1.15, 1.4),
        'poor': random.uniform(1.4, 2.0)  # Potholes, unpaved
    }
    road_multiplier = road_multipliers[road_condition]

    # 4. Weather multiplier (monsoon can severely affect travel)
    weather_multipliers = {
        'clear': random.uniform(0.95, 1.0),
        'cloudy': random.uniform(1.0, 1.05),
        'rain': random.uniform(1.3, 1.7),  # Landslides possible
        'fog': random.uniform(1.2, 1.5),
        'dust': random.uniform(1.05, 1.15),
    }
    weather_multiplier = weather_multipliers[weather_condition]

    # 5. Hill/mountain road penalties
    if road_type == 'hill':
        terrain_multiplier = random.uniform(1.2, 1.5)  # Winding roads, elevation
    else:
        terrain_multiplier = 1.0

    # Calculate actual travel time
    actual_time_minutes = (base_time * traffic_multiplier * road_multiplier *
                          weather_multiplier * terrain_multiplier)

    # Add random noise
    actual_time_minutes *= random.uniform(0.95, 1.05)
    actual_time_minutes = round(actual_time_minutes, 2)

    # Store record
    data.append({
        'distance_km': distance_km,
        'traffic_hours': traffic_hours,
        'vehicle_avg_speed': vehicle_avg_speed,
        'vehicle_type': vehicle_type,
        'road_condition': road_condition,
        'weather_condition': weather_condition,
        'actual_time_minutes': actual_time_minutes,
        'route_from': route['from'],
        'route_to': route['to'],
        'road_type': road_type,
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/sample_data.csv', index=False)

print(f"✅ Generated {len(df)} Nepal-specific samples")
print(f"✅ Saved to data/sample_data.csv")
print("\n📊 Sample routes:")
print(df[['route_from', 'route_to', 'distance_km', 'actual_time_minutes']].head(10))
print("\n📊 Vehicle type distribution:")
print(df['vehicle_type'].value_counts())
print("\n📊 Road type distribution:")
print(df['road_type'].value_counts())
print("\n📊 Weather distribution:")
print(df['weather_condition'].value_counts())
print("\n📊 Travel time statistics:")
print(df['actual_time_minutes'].describe())
