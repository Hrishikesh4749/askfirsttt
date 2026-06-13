"""
database.py
------------
Sets up the SQLite database engine, session factory, and declarative base
used by all SQLAlchemy models in this project.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite database file (created automatically in the backend/ directory)
DATABASE_URL = "sqlite:///./memory_chat.db"

# `check_same_thread=False` is required for SQLite when used with FastAPI,
# since FastAPI can access the DB from multiple threads.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a database session and ensures it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
