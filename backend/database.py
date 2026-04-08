"""
SQLite database setup using SQLAlchemy ORM.
Stores all processed messages, predictions, and actions.
"""

import os
from datetime import datetime
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "support_triage.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class ProcessedMessage(Base):
    """Stores every processed customer message with its classification result."""
    __tablename__ = "processed_messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    message_id = Column(String(100), index=True, nullable=True)
    customer_name = Column(String(200), nullable=True)
    customer_email = Column(String(200), nullable=True)
    channel = Column(String(50), nullable=True)
    original_text = Column(Text, nullable=False)
    predicted_intent = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    action_taken = Column(String(50), nullable=False)  # Auto-Reply / Escalated / Discarded
    auto_reply_text = Column(Text, nullable=True)
    processed_at = Column(DateTime, default=datetime.utcnow)
    message_timestamp = Column(DateTime, nullable=True)


class ModelMetrics(Base):
    """Stores model performance metrics for dashboard display."""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_id = Column(String(100), nullable=False)
    accuracy = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    num_samples = Column(Integer, nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency injection for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
