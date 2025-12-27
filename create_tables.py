from app.database import engine, Base
from app.models import TripData, PredictionLog

# Create all tables
Base.metadata.create_all(bind=engine)

print("✅ Database tables created successfully!")
print("Tables created: trip_data, prediction_logs")
