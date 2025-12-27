import pandas as pd
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine
from app.models import TripData, Base
from datetime import datetime

print("🌱 Starting database seeding...")

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Create database session
db = SessionLocal()

try:
    # Check if data already exists
    existing_count = db.query(TripData).count()

    if existing_count > 0:
        print(f"⚠️  Database already has {existing_count} records.")
        response = input("Do you want to clear and reload? (yes/no): ")

        if response.lower() == 'yes':
            print("🗑️  Clearing existing data...")
            db.query(TripData).delete()
            db.commit()
            print("✅ Data cleared.")
        else:
            print("❌ Seeding cancelled.")
            exit()

    # Load CSV data
    print("📂 Loading data from CSV...")
    df = pd.read_csv('data/sample_data.csv')
    print(f"✅ Loaded {len(df)} records from CSV")

    # Insert data in batches (faster than one by one)
    print("💾 Inserting data into database...")
    batch_size = 100
    total_inserted = 0

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]

        # Create TripData objects
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

        # Bulk insert
        db.bulk_save_objects(trip_records)
        db.commit()

        total_inserted += len(trip_records)
        print(f"  Inserted {total_inserted}/{len(df)} records...")

    print(f"\n✅ Successfully seeded {total_inserted} records into trip_data table!")

    # Verify
    final_count = db.query(TripData).count()
    print(f"📊 Total records in database: {final_count}")

    # Show sample records
    print("\n🔍 Sample records from database:")
    sample_records = db.query(TripData).limit(5).all()
    for record in sample_records:
        print(f"  ID {record.id}: {record.distance_km}km, {record.vehicle_type}, {record.actual_time_minutes:.2f}min")

except Exception as e:
    print(f"❌ Error seeding database: {e}")
    db.rollback()
    raise e

finally:
    db.close()

print("\n🎉 Database seeding complete!")
