"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────────────────────

class MessageInput(BaseModel):
    """Single message for classification."""
    id: str | None = Field(None, description="Unique message identifier")
    customer_name: str | None = Field(None, description="Customer name")
    email: str | None = Field(None, description="Customer email")
    text: str = Field(..., description="Message text to classify")
    timestamp: str | None = Field(None, description="Message timestamp (ISO format)")
    channel: str | None = Field(None, description="Message channel (email, chat, social_media)")


class BatchInput(BaseModel):
    """Batch of messages for processing."""
    messages: list[MessageInput]


# ── Response Models ──────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """Classification result for a single message."""
    message_id: str | None
    customer_name: str | None
    original_text: str
    predicted_intent: str
    confidence_score: float
    action_taken: str  # "Auto-Reply" | "Escalated" | "Discarded"
    auto_reply_text: str | None
    processed_at: datetime


class BatchResult(BaseModel):
    """Batch processing result summary."""
    total_processed: int
    auto_replied: int
    escalated: int
    discarded: int
    results: list[PredictionResult]


class StatsResponse(BaseModel):
    """Dashboard statistics."""
    total_messages: int
    auto_replied: int
    escalated: int
    discarded: int
    avg_confidence: float
    intent_distribution: dict[str, int]
    escalation_rate: float
    auto_reply_rate: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    database_connected: bool
    version: str
