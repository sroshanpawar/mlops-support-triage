"""
Tests for the FastAPI backend endpoints.
"""

import json
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app
from backend.database import init_db


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    init_db()
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "database_connected" in data
        assert "version" in data

    def test_health_status_healthy(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestPredictionEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_returns_result(self, client):
        """Test single message prediction (requires trained model)."""
        response = client.post(
            "/predict",
            json={
                "text": "Where is my order?",
                "customer_name": "Test User",
                "channel": "test",
            },
        )
        # If model is not loaded, expect 503
        if response.status_code == 503:
            pytest.skip("Model not trained yet — skipping prediction test.")
        assert response.status_code == 200
        data = response.json()
        assert "predicted_intent" in data
        assert "confidence_score" in data
        assert "action_taken" in data
        assert data["action_taken"] in ("Auto-Reply", "Escalated", "Discarded")

    def test_predict_requires_text(self, client):
        """Test that 'text' field is required."""
        response = client.post("/predict", json={"customer_name": "Test"})
        assert response.status_code == 422  # Validation error


class TestBatchPredictionEndpoint:
    """Tests for the /batch-predict endpoint."""

    def test_batch_predict(self, client):
        """Test batch message processing."""
        messages = [
            {"text": "Where is my order?", "id": "t1"},
            {"text": "I want a refund", "id": "t2"},
            {"text": "How much does this cost?", "id": "t3"},
        ]
        response = client.post("/batch-predict", json={"messages": messages})
        if response.status_code == 503:
            pytest.skip("Model not trained yet.")
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 3
        assert len(data["results"]) == 3

    def test_batch_predict_empty(self, client):
        """Test empty batch processing."""
        response = client.post("/batch-predict", json={"messages": []})
        if response.status_code == 503:
            pytest.skip("Model not trained yet.")
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 0


class TestMessagesEndpoint:
    """Tests for the /messages endpoint."""

    def test_get_messages(self, client):
        response = client.get("/messages")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "messages" in data

    def test_get_messages_with_filter(self, client):
        response = client.get("/messages", params={"intent": "Shipping_Inquiry"})
        assert response.status_code == 200

    def test_get_messages_with_limit(self, client):
        response = client.get("/messages", params={"limit": 5})
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) <= 5


class TestStatsEndpoint:
    """Tests for the /stats endpoint."""

    def test_stats_returns_200(self, client):
        response = client.get("/stats")
        assert response.status_code == 200

    def test_stats_response_structure(self, client):
        response = client.get("/stats")
        data = response.json()
        assert "total_messages" in data
        assert "auto_replied" in data
        assert "escalated" in data
        assert "discarded" in data
        assert "avg_confidence" in data
        assert "intent_distribution" in data
        assert "escalation_rate" in data
        assert "auto_reply_rate" in data
