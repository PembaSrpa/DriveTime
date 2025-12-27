# app/models.py
"""
Database models (tables) for storing trip data.
SQLAlchemy maps these Python classes to PostgreSQL tables.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from app.database import Base


class TripData(Base):
    """
    Stores historical trip data for training the ML model.
    Each row represents one completed trip.
    """
    __tablename__ = "trip_data"

    # Primary key - unique identifier for each trip
    id = Column(Integer, primary_key=True, index=True)

    # Trip features (inputs for our model)
    distance_km = Column(Float, nullable=False)  # Distance in kilometers
    traffic_hours = Column(Float, nullable=False)  # Time of day affecting traffic (0-23.99)
    vehicle_avg_speed = Column(Float, nullable=False)  # Average speed of vehicle (km/h)
    vehicle_type = Column(String, nullable=False)  # car, truck, motorcycle, etc.
    road_condition = Column(String, nullable=False)  # excellent, good, fair, poor
    weather_condition = Column(String, nullable=False)  # clear, rain, fog, snow

    # Target variable (what we're predicting)
    actual_time_minutes = Column(Float, nullable=False)  # Actual travel time in minutes

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    """
    Logs all predictions made by the API.
    Useful for monitoring and improving the model over time.
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    # Input features
    distance_km = Column(Float, nullable=False)
    traffic_hours = Column(Float, nullable=False)
    vehicle_avg_speed = Column(Float, nullable=False)
    vehicle_type = Column(String, nullable=False)
    road_condition = Column(String, nullable=False)
    weather_condition = Column(String, nullable=False)

    # Prediction output
    predicted_time_minutes = Column(Float, nullable=False)

    # Timestamp
    predicted_at = Column(DateTime, default=datetime.utcnow)
