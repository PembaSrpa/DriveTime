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
    description="Stateless ML-Powered Vehicle Travel Time Estimator for Nepal Routes",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Stateless DriveTime API...")
    try:
        predictor.load_model()
        print("✅ ML Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

@app.get("/")
async def root():
    return {
        "message": "Welcome to DriveTime API (Stateless Mode)",
        "status": "Running without Database",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    # Removed DB check to prevent crashes after Supabase deletion
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.model_loaded,
        timestamp=datetime.utcnow()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_travel_time(request: PredictionRequest):
    if not predictor.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # We now include 'road_type' in the input_data for better Nepal accuracy
        input_data = request.dict()

        predicted_minutes = predictor.predict(input_data)
        predicted_hours = round(predicted_minutes / 60, 2)

        return PredictionResponse(
            predicted_time_minutes=predicted_minutes,
            predicted_time_hours=predicted_hours,
            input_features=input_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
