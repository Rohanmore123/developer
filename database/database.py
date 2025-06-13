from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy.engine.url import URL
# DATABASE_URL = "postgresql://username:password@database-url/db_name"

postgres_username = os.getenv("postgres_username")
postgres_password = os.getenv("postgres_password")
postgres_host = os.getenv("postgres_host")
postgres_port = os.getenv("postgres_port")
postgres_database = os.getenv("postgres_database")



DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=postgres_username,
    password=postgres_password,
    host=postgres_host,
    port=postgres_port,
    database=postgres_database
)

# print(f"Database URL: {DATABASE_URL}")

# Fallback to environment variable if URL creation fails
if not DATABASE_URL:
    DATABASE_URL = os.getenv("DATABASE_URL")
    print(f"Using fallback DATABASE_URL from environment: {DATABASE_URL}")

# DATABASE_URL = "postgresql://postgres:Raje%4012345@localhost:5432/Prasha_Health"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

#  Dependency for getting DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
