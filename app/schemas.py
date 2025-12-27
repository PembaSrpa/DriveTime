from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime


class PredictionRequest(BaseModel):
    """
    Request schema for travel time prediction.
    All fields are required and validated automatically.
    """
    distance_km: float = Field(
        ...,
        gt=0,
        description="Distance in kilometers (must be positive)"
    )
    traffic_hours: float = Field(
        ...,
        ge=0,
        lt=24,
        description="Time of day in 24-hour format (0-23.99, e.g., 8.5 = 8:30 AM)"
    )
    vehicle_avg_speed: float = Field(
        ...,
        gt=0,
        le=150,
        description="Vehicle average speed in km/h (1-150)"
    )
    vehicle_type: Literal['car', 'truck', 'motorcycle', 'suv', 'van', 'microbus', 'bus', 'jeep', 'scooter'] = Field(
        ...,
        description="Type of vehicle"
    )
    road_condition: Literal['excellent', 'good', 'fair', 'poor'] = Field(
        ...,
        description="Current road condition"
    )
    weather_condition: Literal['clear', 'rain', 'fog', 'snow', 'cloudy', 'dust'] = Field(
        ...,
        description="Current weather condition"
    )

    class Config:
        # Example for API documentation
        json_schema_extra = {
            "example": {
                "distance_km": 200.0,
                "traffic_hours": 8.5,
                "vehicle_avg_speed": 70.0,
                "vehicle_type": "car",
                "road_condition": "good",
                "weather_condition": "clear"
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema for travel time prediction.
    Returns predicted time and input features for reference.
    """
    predicted_time_minutes: float = Field(
        ...,
        description="Predicted travel time in minutes"
    )
    predicted_time_hours: float = Field(
        ...,
        description="Predicted travel time in hours (for convenience)"
    )
    input_features: dict = Field(
        ...,
        description="Echo of input features for verification"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_time_minutes": 245.67,
                "predicted_time_hours": 4.09,
                "input_features": {
                    "distance_km": 200.0,
                    "traffic_hours": 8.5,
                    "vehicle_avg_speed": 70.0,
                    "vehicle_type": "car",
                    "road_condition": "good",
                    "weather_condition": "clear"
                }
            }
        }


class HealthResponse(BaseModel):
    """
    Response schema for health check endpoint.
    """
    status: str = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    timestamp: datetime = Field(..., description="Current server timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2025-12-26T10:30:00Z"
            }
        }
