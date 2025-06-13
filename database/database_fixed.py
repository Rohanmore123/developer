from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
import json
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy.engine.url import URL

# Helper function to parse AWS Secrets Manager values
def parse_secret(value):
    if value and value.startswith('{'):
        try:
            # Try to parse as JSON
            return json.loads(value)
        except json.JSONDecodeError:
            # If it's not valid JSON, return the original value
            return value
    return value

# Try to get database credentials from environment variables
# First try lowercase variables (used in local development)
postgres_username = os.getenv("postgres_username")
postgres_password = os.getenv("postgres_password")
postgres_host = os.getenv("postgres_host")
postgres_port = os.getenv("postgres_port")
postgres_database = os.getenv("postgres_database")

# If any of the lowercase variables are missing, try uppercase variables (used in AWS)
if not postgres_username:
    postgres_username = os.getenv("POSTGRES_USERNAME")
if not postgres_password:
    postgres_password = os.getenv("POSTGRES_PASSWORD")
if not postgres_host:
    postgres_host = os.getenv("POSTGRES_HOST")
if not postgres_port:
    postgres_port = os.getenv("POSTGRES_PORT")
if not postgres_database:
    postgres_database = os.getenv("POSTGRES_DATABASE")

# Parse values from AWS Secrets Manager if needed
postgres_username = parse_secret(postgres_username)
postgres_password = parse_secret(postgres_password)
postgres_host = parse_secret(postgres_host)
postgres_port = parse_secret(postgres_port)
postgres_database = parse_secret(postgres_database)

# For debugging
print(f"Database connection parameters:")
print(f"Username: {postgres_username}")
print(f"Password: {'*****' if postgres_password else 'None'}")
print(f"Host: {postgres_host}")
print(f"Port: {postgres_port}")
print(f"Database: {postgres_database}")

# Fallback values for testing
if not postgres_username or not postgres_password or not postgres_host or not postgres_port or not postgres_database:
    print("Using fallback database connection parameters for testing")
    # Load from .env file if available
    from dotenv import load_dotenv
    load_dotenv()

    # Try to get from environment again
    postgres_username = os.getenv("POSTGRES_USERNAME", "postgres")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "db_password")
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = os.getenv("POSTGRES_PORT", 5432)
    postgres_database = os.getenv("POSTGRES_DATABASE", "postgres")

try:
    DATABASE_URL = URL.create(
        drivername="postgresql+psycopg2",
        username=postgres_username,
        password=postgres_password,
        host=postgres_host,
        port=postgres_port,
        database=postgres_database
    )
    print(f"Database URL created successfully")
except Exception as e:
    print(f"Error creating database URL: {e}")
    # Fallback to a direct connection string
    DATABASE_URL = f"postgresql://{postgres_username}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_database}"
    print(f"Using fallback connection string")

# Fallback to environment variable if URL creation fails
if not DATABASE_URL:
    DATABASE_URL = os.getenv("DATABASE_URL")
    print(f"Using fallback DATABASE_URL from environment: {DATABASE_URL}")

# Create engine and session
try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    print("Database engine created successfully")
except Exception as e:
    print(f"Error creating database engine: {e}")
    # Set dummy values for engine and SessionLocal to prevent import errors
    engine = None
    SessionLocal = None
    Base = declarative_base()

# Dependency for getting DB session
def get_db():
    if not engine:
        print("Warning: Database engine not initialized")
        return None

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
