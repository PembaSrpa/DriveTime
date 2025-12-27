from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database URL from environment variable
# Railway will automatically provide this as DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Create SQLAlchemy engine
# This manages the connection pool to PostgreSQL
engine = create_engine(DATABASE_URL)

# SessionLocal is a factory for creating database sessions
# Each session represents a "conversation" with the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all our database models
# All table models will inherit from this
Base = declarative_base()

# Dependency function for FastAPI routes
# This ensures each request gets its own database session
def get_db():
    db = SessionLocal()
    try:
        yield db  # Provides the session to the route
    finally:
        db.close()  # Always closes the session when done
