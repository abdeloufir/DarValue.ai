"""
Database connection and session management for DarValue.ai
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from decouple import config
from typing import Generator
import logging

from .models import Base

# Database configuration
DATABASE_URL = config(
    'DATABASE_URL', 
    default='postgresql://postgres:password@localhost:5432/darvalue_db'
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=config('DATABASE_ECHO', default=False, cast=bool)
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

logger = logging.getLogger(__name__)


def create_database():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session
    Use this in FastAPI dependencies or other contexts
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get a database session for direct use
    Remember to close the session when done
    """
    return SessionLocal()


class DatabaseManager:
    """Database manager class for handling connections and operations"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def create_tables(self):
        """Create all tables"""
        create_database()
    
    def drop_tables(self):
        """Drop all tables - USE WITH CAUTION"""
        Base.metadata.drop_all(bind=engine)
        logger.warning("All database tables have been dropped")
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return SessionLocal()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()