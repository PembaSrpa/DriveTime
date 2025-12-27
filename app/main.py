from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os

# Import our schemas and prediction service
from app.schemas import PredictionRequest, PredictionResponse, HealthResponse
from app.ml.predict import predictor


# Create FastAPI application
app = FastAPI(
    title="DriveTime API",
    description="ML-Powered Vehicle Travel Time Estimator for Nepal Routes",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc UI
)

# Add CORS middleware (allows frontend apps to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Event: Run when API starts
@app.on_event("startup")
async def startup_event():
    """
    Load ML model when the API starts.
    This runs once, not on every request.
    """
    print("🚀 Starting DriveTime API...")
    print("📦 Loading ML model...")

    try:
        predictor.load_model()
        print("✅ API is ready to serve predictions!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("⚠️  API will start but predictions will fail.")


# Event: Run when API shuts down
@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup when API shuts down.
    """
    print("👋 Shutting down DriveTime API...")


# ROOT ENDPOINT
@app.get("/")
async def root():
    """
    Root endpoint - Welcome message and available endpoints.
    """
    return {
        "message": "Welcome to DriveTime API - ML-Powered Vehicle Travel Estimator",
        "version": "1.0.0",
        "description": "Predicts vehicle travel times for Nepal routes using XGBoost",
        "endpoints": {
            "predict": "POST /predict - Get travel time prediction",
            "health": "GET /health - Check API health",
            "docs": "GET /docs - Interactive API documentation (Swagger UI)",
            "redoc": "GET /redoc - Alternative API documentation (ReDoc)"
        },
        "example_request": {
            "url": "/predict",
            "method": "POST",
            "body": {
                "distance_km": 200.0,
                "traffic_hours": 8.5,
                "vehicle_avg_speed": 70.0,
                "vehicle_type": "car",
                "road_condition": "good",
                "weather_condition": "clear"
            }
        }
    }


# HEALTH CHECK ENDPOINT
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint - Verify API and model are working.

    Returns:
        HealthResponse with status, model_loaded, and timestamp
    """
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.model_loaded,
        timestamp=datetime.utcnow()
    )


# PREDICTION ENDPOINT (Main functionality)
@app.post("/predict", response_model=PredictionResponse)
async def predict_travel_time(request: PredictionRequest):
    """
    Predict vehicle travel time based on input features.

    Args:
        request: PredictionRequest with trip details

    Returns:
        PredictionResponse with predicted travel time

    Raises:
        HTTPException: If model not loaded or prediction fails
    """

    # Check if model is loaded
    if not predictor.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please contact administrator."
        )

    try:
        # Convert Pydantic model to dictionary
        input_data = {
            "distance_km": request.distance_km,
            "traffic_hours": request.traffic_hours,
            "vehicle_avg_speed": request.vehicle_avg_speed,
            "vehicle_type": request.vehicle_type,
            "road_condition": request.road_condition,
            "weather_condition": request.weather_condition
        }

        # Make prediction using our ML model
        predicted_minutes = predictor.predict(input_data)

        # Convert to hours for convenience
        predicted_hours = round(predicted_minutes / 60, 2)

        # Return response
        return PredictionResponse(
            predicted_time_minutes=predicted_minutes,
            predicted_time_hours=predicted_hours,
            input_features=input_data
        )

    except Exception as e:
        # Log the error (in production, use proper logging)
        print(f"❌ Prediction error: {e}")

        # Return error to user
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# MODEL INFO ENDPOINT (Optional - for debugging)
@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded ML model.
    Useful for debugging and verification.

    Returns:
        Dictionary with model metadata
    """
    if not predictor.model_loaded:
        return {"error": "Model not loaded"}

    return predictor.get_model_info()


# Run the application (for local testing)
if __name__ == "__main__":
    import uvicorn

    # Get port from environment variable (Railway provides this)
    port = int(os.getenv("PORT", 8000))

    print(f"🚀 Starting server on port {port}...")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True  # Auto-reload on code changes (disable in production)
    )
