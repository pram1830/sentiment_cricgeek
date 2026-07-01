"""
Database connection and session management
"""

import os
from typing import Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool

from models import Base


ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()


def _load_database_url() -> str:
    database_url = os.getenv("DATABASE_URL", "sqlite:///./cricgeek_dev.db")
    if ENVIRONMENT == "production" and database_url.startswith("sqlite"):
        raise RuntimeError("Production storage must use PostgreSQL or MySQL. Set DATABASE_URL.")
    return database_url


# Database configuration. PostgreSQL and MySQL are supported for production;
# SQLite remains available only for local development and tests.
DATABASE_URL = _load_database_url()

# Engine configuration
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"timeout": 20, "check_same_thread": False},
        echo=False,
    )
else:
    connect_args = {}
    if DATABASE_URL.startswith("postgresql"):
        connect_args["connect_timeout"] = 10
    elif DATABASE_URL.startswith("mysql"):
        connect_args["charset"] = "utf8mb4"

    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
        connect_args=connect_args,
    )

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Session:
    """Get database session (dependency for FastAPI)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def drop_db():
    """Drop all tables (DANGEROUS - for testing only)"""
    Base.metadata.drop_all(bind=engine)


def seed_db():
    """Seed database with sample data (for development)"""
    from models import User, WriterProfile, Community, TopicEnum, CommunityVisibilityEnum
    
    db = SessionLocal()
    try:
        # Check if already seeded
        if db.query(User).count() > 0:
            print("Database already seeded")
            return
        
        # Create sample users
        user1 = User(
            username="cricket_analyst",
            email="analyst@cricgeek.local",
            password_hash=User.hash_password("password123"),
            email_verified=True,
            profile_bio="Cricket statistics enthusiast"
        )
        
        user2 = User(
            username="emotional_fan",
            email="fan@cricgeek.local",
            password_hash=User.hash_password("password123"),
            email_verified=True,
            profile_bio="Passionate cricket fan"
        )
        
        db.add(user1)
        db.add(user2)
        db.flush()
        
        # Create writer profiles
        profile1 = WriterProfile(
            user_id=user1.id,
            writing_style="analytical",
            primary_topics=["batting", "strategy"],
            total_submissions=5,
        )
        
        profile2 = WriterProfile(
            user_id=user2.id,
            writing_style="emotional",
            primary_topics=["general", "team_performance"],
            total_submissions=2,
        )
        
        db.add(profile1)
        db.add(profile2)
        db.flush()
        
        # Create communities
        community1 = Community(
            name="Analytical Cricketers",
            slug="analytical-cricketers",
            description="For writers who focus on stats and strategy",
            primary_topic=TopicEnum.STRATEGY,
            creator_id=user1.id,
            visibility=CommunityVisibilityEnum.PUBLIC,
        )
        
        community2 = Community(
            name="Fan Discussions",
            slug="fan-discussions",
            description="General cricket discussions from passionate fans",
            primary_topic=TopicEnum.GENERAL,
            creator_id=user2.id,
            visibility=CommunityVisibilityEnum.PUBLIC,
        )
        
        db.add(community1)
        db.add(community2)
        
        db.commit()
        print("Database seeded successfully")
        
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
    finally:
        db.close()


# Event listeners for connection management
@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Configure per-connection database settings."""
    if DATABASE_URL.startswith("postgresql"):
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("SET statement_timeout = '30s'")
        finally:
            cursor.close()


if __name__ == "__main__":
    # CLI commands for database management
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "init":
            print("Initializing database...")
            init_db()
            print("Database initialized")
        
        elif command == "drop":
            print("WARNING: Dropping all tables...")
            drop_db()
            print("Database dropped")
        
        elif command == "seed":
            print("Seeding database...")
            init_db()
            seed_db()
        
        elif command == "reset":
            print("Resetting database...")
            drop_db()
            init_db()
            seed_db()
            print("Database reset and seeded")
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: init, drop, seed, reset")
    
    else:
        print("Database utility")
        print("Usage: python database.py [init|drop|seed|reset]")
