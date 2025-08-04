"""Database connection and session management."""

import os
from typing import Optional, Any, Dict
from contextlib import contextmanager
import logging

try:
    from sqlalchemy import create_engine, MetaData
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import StaticPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions for persistent storage."""
    
    def __init__(self, database_url: Optional[str] = None):
        if not SQLALCHEMY_AVAILABLE:
            logger.warning("SQLAlchemy not available. Database features disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.database_url = database_url or self._get_default_database_url()
        self.engine = None
        self.SessionLocal = None
        self._setup_database()
    
    def _get_default_database_url(self) -> str:
        """Get default database URL from environment or use SQLite."""
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            return database_url
        
        # Default to SQLite in current directory
        return "sqlite:///moe_debugger.db"
    
    def _setup_database(self):
        """Setup database engine and session factory."""
        try:
            # Configure engine based on database type
            if self.database_url.startswith("sqlite"):
                # SQLite specific configuration
                self.engine = create_engine(
                    self.database_url,
                    poolclass=StaticPool,
                    connect_args={
                        "check_same_thread": False,  # Allow multiple threads
                        "timeout": 20  # Connection timeout
                    },
                    echo=os.getenv("DEBUG", "false").lower() == "true"
                )
            else:
                # PostgreSQL, MySQL, etc.
                self.engine = create_engine(
                    self.database_url,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    echo=os.getenv("DEBUG", "false").lower() == "true"
                )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info(f"Database connection established: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            self.enabled = False
    
    def create_tables(self):
        """Create all database tables."""
        if not self.enabled:
            return
        
        try:
            from .models import Base
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
    
    def drop_tables(self):
        """Drop all database tables."""
        if not self.enabled:
            return
        
        try:
            from .models import Base
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
    
    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup."""
        if not self.enabled:
            yield None
            return
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session_factory(self):
        """Get the session factory for dependency injection."""
        return self.SessionLocal if self.enabled else None
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        if not self.enabled:
            return {
                "status": "disabled",
                "message": "Database not available"
            }
        
        try:
            with self.get_session() as session:
                # Simple query to test connection
                session.execute("SELECT 1")
                return {
                    "status": "healthy",
                    "database_url": self.database_url.split("@")[-1] if "@" in self.database_url else self.database_url,
                    "engine": str(self.engine.dialect.name)
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()