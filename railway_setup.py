import os
import sys

print("🚂 Railway Setup Script")
print("=" * 50)

# Check if DATABASE_URL exists (means we're on Railway)
if not os.getenv("DATABASE_URL"):
    print("⚠️  DATABASE_URL not found.")
    print("This script is meant for Railway deployment.")
    print("For local setup, run:")
    print("  1. python create_tables.py")
    print("  2. python seed_database.py")
    sys.exit(1)

print("✅ DATABASE_URL found - Running on Railway")

# Import after checking DATABASE_URL
from app.database import engine, Base, SessionLocal
from app.models import TripData, PredictionLog
import pandas as pd
from datetime import datetime

try:
    # Step 1: Create tables
    print("\n📋 Step 1: Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created: trip_data, prediction_logs")

    # Step 2: Check if data exists
    print("\n📊 Step 2: Checking existing data...")
    db = SessionLocal()
    existing_count = db.query(TripData).count()

    if existing_count > 0:
        print(f"✅ Database already has {existing_count} records. Skipping seed.")
    else:
        print("📂 No data found. Seeding database...")

        # Load and insert data
        df = pd.read_csv('data/sample_data.csv')
        print(f"✅ Loaded {len(df)} records from CSV")

        # Insert in batches
        batch_size = 100
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            trip_records = []

            for _, row in batch.iterrows():
                trip = TripData(
                    distance_km=float(row['distance_km']),
                    traffic_hours=float(row['traffic_hours']),
                    vehicle_avg_speed=float(row['vehicle_avg_speed']),
                    vehicle_type=str(row['vehicle_type']),
                    road_condition=str(row['road_condition']),
                    weather_condition=str(row['weather_condition']),
                    actual_time_minutes=float(row['actual_time_minutes']),
                    created_at=datetime.utcnow()
                )
                trip_records.append(trip)

            db.bulk_save_objects(trip_records)
            db.commit()
            print(f"  Inserted {min(i+batch_size, len(df))}/{len(df)} records...")

        final_count = db.query(TripData).count()
        print(f"✅ Seeded {final_count} records successfully!")

    db.close()
    print("\n🎉 Railway setup complete!")

except Exception as e:
    print(f"\n❌ Setup failed: {e}")
    sys.exit(1)
