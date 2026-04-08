"""
FastAPI backend for the Smart Post-Purchase Support Triage System.
Serves prediction endpoints and message history/statistics.
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import ProcessedMessage, get_db, init_db
from backend.classifier import classifier
from backend.models import (
    BatchInput,
    BatchResult,
    HealthResponse,
    MessageInput,
    PredictionResult,
    StatsResponse,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_db()
    print("🗄️  Database initialized.")
    yield
    print("👋 Shutting down...")


# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Post-Purchase Support Triage System",
    description="AI-powered customer intent classification and triage for e-commerce support.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=classifier.is_loaded(),
        database_connected=True,
        version="1.0.0",
    )


# ── Single Prediction ───────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict_single(message: MessageInput, db: Session = Depends(get_db)):
    """Classify a single customer message and store the result."""
    if not classifier.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    result = classifier.predict(message.text)
    now = datetime.utcnow()

    # Parse message timestamp
    msg_ts = None
    if message.timestamp:
        try:
            msg_ts = datetime.fromisoformat(message.timestamp.replace("Z", "+00:00"))
        except ValueError:
            msg_ts = None

    # Store in database
    db_record = ProcessedMessage(
        message_id=message.id,
        customer_name=message.customer_name,
        customer_email=message.email,
        channel=message.channel,
        original_text=message.text,
        predicted_intent=result["predicted_intent"],
        confidence_score=result["confidence_score"],
        action_taken=result["action_taken"],
        auto_reply_text=result["auto_reply_text"],
        processed_at=now,
        message_timestamp=msg_ts,
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    # Append to outbound_replies.json if auto-replied or escalated
    if result["action_taken"] in ("Auto-Reply", "Escalated") and result["auto_reply_text"]:
        classifier.append_outbound_reply({
            "message_id": message.id,
            "customer_name": message.customer_name,
            "customer_email": message.email,
            "intent": result["predicted_intent"],
            "action": result["action_taken"],
            "reply_text": result["auto_reply_text"],
            "generated_at": now.isoformat(),
        })

    return PredictionResult(
        message_id=message.id,
        customer_name=message.customer_name,
        original_text=message.text,
        predicted_intent=result["predicted_intent"],
        confidence_score=result["confidence_score"],
        action_taken=result["action_taken"],
        auto_reply_text=result["auto_reply_text"],
        processed_at=now,
    )


# ── Batch Prediction ────────────────────────────────────────────────────────

@app.post("/batch-predict", response_model=BatchResult, tags=["Prediction"])
def predict_batch(batch: BatchInput, db: Session = Depends(get_db)):
    """Process a batch of customer messages."""
    if not classifier.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    results = []
    auto_replied = 0
    escalated = 0
    discarded = 0

    for message in batch.messages:
        result = classifier.predict(message.text)
        now = datetime.utcnow()

        msg_ts = None
        if message.timestamp:
            try:
                msg_ts = datetime.fromisoformat(message.timestamp.replace("Z", "+00:00"))
            except ValueError:
                msg_ts = None

        db_record = ProcessedMessage(
            message_id=message.id,
            customer_name=message.customer_name,
            customer_email=message.email,
            channel=message.channel,
            original_text=message.text,
            predicted_intent=result["predicted_intent"],
            confidence_score=result["confidence_score"],
            action_taken=result["action_taken"],
            auto_reply_text=result["auto_reply_text"],
            processed_at=now,
            message_timestamp=msg_ts,
        )
        db.add(db_record)

        if result["action_taken"] in ("Auto-Reply", "Escalated") and result["auto_reply_text"]:
            classifier.append_outbound_reply({
                "message_id": message.id,
                "customer_name": message.customer_name,
                "customer_email": message.email,
                "intent": result["predicted_intent"],
                "action": result["action_taken"],
                "reply_text": result["auto_reply_text"],
                "generated_at": now.isoformat(),
            })

        # Count actions
        if result["action_taken"] == "Auto-Reply":
            auto_replied += 1
        elif result["action_taken"] == "Escalated":
            escalated += 1
        elif result["action_taken"] == "Discarded":
            discarded += 1

        results.append(PredictionResult(
            message_id=message.id,
            customer_name=message.customer_name,
            original_text=message.text,
            predicted_intent=result["predicted_intent"],
            confidence_score=result["confidence_score"],
            action_taken=result["action_taken"],
            auto_reply_text=result["auto_reply_text"],
            processed_at=now,
        ))

    db.commit()

    return BatchResult(
        total_processed=len(results),
        auto_replied=auto_replied,
        escalated=escalated,
        discarded=discarded,
        results=results,
    )


# ── File Upload for Batch Processing ────────────────────────────────────────

@app.post("/upload-traffic", response_model=BatchResult, tags=["Prediction"])
async def upload_traffic_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a simulated_traffic.json file for batch processing."""
    if not classifier.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are accepted.")

    try:
        content = await file.read()
        messages_data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")

    # Convert to MessageInput list
    messages = []
    for m in messages_data:
        messages.append(MessageInput(
            id=m.get("id"),
            customer_name=m.get("customer_name"),
            email=m.get("email"),
            text=m["text"],
            timestamp=m.get("timestamp"),
            channel=m.get("channel"),
        ))

    batch = BatchInput(messages=messages)
    return predict_batch(batch, db)


# ── Message History ──────────────────────────────────────────────────────────

@app.get("/messages", tags=["Data"])
def get_messages(
    intent: str | None = None,
    action: str | None = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Retrieve processed messages with optional filters."""
    query = db.query(ProcessedMessage)

    if intent:
        query = query.filter(ProcessedMessage.predicted_intent == intent)
    if action:
        query = query.filter(ProcessedMessage.action_taken == action)

    total = query.count()
    messages = (
        query.order_by(ProcessedMessage.processed_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "total": total,
        "messages": [
            {
                "id": m.id,
                "message_id": m.message_id,
                "customer_name": m.customer_name,
                "customer_email": m.customer_email,
                "channel": m.channel,
                "original_text": m.original_text,
                "predicted_intent": m.predicted_intent,
                "confidence_score": m.confidence_score,
                "action_taken": m.action_taken,
                "auto_reply_text": m.auto_reply_text,
                "processed_at": m.processed_at.isoformat() if m.processed_at else None,
            }
            for m in messages
        ],
    }


# ── Dashboard Statistics ────────────────────────────────────────────────────

@app.get("/stats", response_model=StatsResponse, tags=["Data"])
def get_stats(db: Session = Depends(get_db)):
    """Get aggregated statistics for the dashboard."""
    total = db.query(func.count(ProcessedMessage.id)).scalar() or 0
    auto_replied = (
        db.query(func.count(ProcessedMessage.id))
        .filter(ProcessedMessage.action_taken == "Auto-Reply")
        .scalar() or 0
    )
    escalated_count = (
        db.query(func.count(ProcessedMessage.id))
        .filter(ProcessedMessage.action_taken == "Escalated")
        .scalar() or 0
    )
    discarded_count = (
        db.query(func.count(ProcessedMessage.id))
        .filter(ProcessedMessage.action_taken == "Discarded")
        .scalar() or 0
    )
    avg_conf = db.query(func.avg(ProcessedMessage.confidence_score)).scalar() or 0.0

    # Intent distribution
    intent_counts = (
        db.query(ProcessedMessage.predicted_intent, func.count(ProcessedMessage.id))
        .group_by(ProcessedMessage.predicted_intent)
        .all()
    )
    intent_distribution = {intent: count for intent, count in intent_counts}

    escalation_rate = (escalated_count / total * 100) if total > 0 else 0.0
    auto_reply_rate = (auto_replied / total * 100) if total > 0 else 0.0

    return StatsResponse(
        total_messages=total,
        auto_replied=auto_replied,
        escalated=escalated_count,
        discarded=discarded_count,
        avg_confidence=round(avg_conf, 4),
        intent_distribution=intent_distribution,
        escalation_rate=round(escalation_rate, 2),
        auto_reply_rate=round(auto_reply_rate, 2),
    )
