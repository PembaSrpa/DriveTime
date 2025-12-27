# DriveTime - ML-Powered Vehicle Travel Estimator

Predicts vehicle travel times for Nepal routes using XGBoost machine learning.

## Features
- XGBoost regression model with 90% accuracy
- FastAPI REST API
- Nepal-specific route data
- Considers traffic, weather, road conditions

## API Endpoints

### POST /predict
Predict travel time

**Request:**
```json
{
  "distance_km": 200,
  "traffic_hours": 8.5,
  "vehicle_avg_speed": 70,
  "vehicle_type": "car",
  "road_condition": "good",
  "weather_condition": "clear"
}
```

**Response:**
```json
{
  "predicted_time_minutes": 245.67,
  "predicted_time_hours": 4.09
}
```

### GET /health
Check API status

### GET /docs
Interactive API documentation

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python app/ml/train.py
```

3. Run the server:
```bash
uvicorn app.main:app --reload
```

4. Visit http://localhost:8000/docs

## Deployment

Deployed on Railway with PostgreSQL database.

## Tech Stack
- FastAPI
- XGBoost
- PostgreSQL
- SQLAlchemy
- Scikit-Learn
- Pandas
