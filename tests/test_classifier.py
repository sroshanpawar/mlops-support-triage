"""
Tests for the intent classifier module.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.classifier import (
    IntentClassifier,
    HIGH_CONFIDENCE_THRESHOLD,
    ROUTINE_INTENTS,
    HIGH_RISK_INTENTS,
    AUTO_REPLIES,
)


class TestClassifierConfiguration:
    """Test classifier configuration constants."""

    def test_confidence_threshold(self):
        assert HIGH_CONFIDENCE_THRESHOLD == 0.85

    def test_routine_intents(self):
        assert "Shipping_Inquiry" in ROUTINE_INTENTS
        assert "Price_Inquiry" in ROUTINE_INTENTS
        assert "Spam" in ROUTINE_INTENTS

    def test_high_risk_intents(self):
        assert "Refund_Request" in HIGH_RISK_INTENTS
        assert "Product_Dispute" in HIGH_RISK_INTENTS

    def test_auto_replies_defined(self):
        assert "Shipping_Inquiry" in AUTO_REPLIES
        assert "Price_Inquiry" in AUTO_REPLIES
        assert AUTO_REPLIES["Spam"] is None

    def test_no_overlap_between_risk_levels(self):
        assert len(ROUTINE_INTENTS & HIGH_RISK_INTENTS) == 0


class TestTriageLogic:
    """Test the confidence-based escalation logic."""

    @pytest.fixture
    def clf(self):
        """Create a classifier instance (model may not be loaded)."""
        return IntentClassifier()

    def test_routine_high_confidence_auto_reply(self, clf):
        action, reply = clf._determine_action("Shipping_Inquiry", 0.92, "msg_001")
        assert action == "Auto-Reply"
        assert reply is not None

    def test_routine_low_confidence_escalate(self, clf):
        action, reply = clf._determine_action("Shipping_Inquiry", 0.60, "msg_002")
        assert action == "Escalated"

    def test_high_risk_high_confidence_escalate(self, clf):
        action, reply = clf._determine_action("Refund_Request", 0.95, "msg_003")
        assert action == "Escalated"

    def test_high_risk_low_confidence_escalate(self, clf):
        action, reply = clf._determine_action("Product_Dispute", 0.50, "msg_004")
        assert action == "Escalated"

    def test_spam_high_confidence_discard(self, clf):
        action, reply = clf._determine_action("Spam", 0.98, "msg_005")
        assert action == "Discarded"
        assert reply is None

    def test_spam_low_confidence_escalate(self, clf):
        action, reply = clf._determine_action("Spam", 0.70, "msg_006")
        assert action == "Escalated"

    def test_price_inquiry_high_confidence_auto_reply(self, clf):
        action, reply = clf._determine_action("Price_Inquiry", 0.90, "msg_007")
        assert action == "Auto-Reply"
        assert "pricing" in reply.lower() or "price" in reply.lower()

    def test_escalation_includes_message_id(self, clf):
        action, reply = clf._determine_action("Refund_Request", 0.95, "msg_test_123")
        assert "msg_test_123" in reply

    def test_boundary_confidence_exactly_threshold(self, clf):
        """At exactly the threshold, should escalate (not > threshold)."""
        action, reply = clf._determine_action("Shipping_Inquiry", 0.85, "msg_boundary")
        assert action == "Escalated"

    def test_boundary_confidence_just_above_threshold(self, clf):
        action, reply = clf._determine_action("Shipping_Inquiry", 0.851, "msg_above")
        assert action == "Auto-Reply"


class TestModelPrediction:
    """Test actual model predictions (requires trained model)."""

    @pytest.fixture
    def clf(self):
        clf = IntentClassifier()
        if not clf.is_loaded():
            pytest.skip("Model not trained yet — skipping prediction tests.")
        return clf

    def test_predict_returns_dict(self, clf):
        result = clf.predict("Where is my order?")
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, clf):
        result = clf.predict("I want a refund")
        assert "predicted_intent" in result
        assert "confidence_score" in result
        assert "action_taken" in result
        assert "auto_reply_text" in result

    def test_predict_confidence_range(self, clf):
        result = clf.predict("How much is this product?")
        assert 0 <= result["confidence_score"] <= 1

    def test_predict_valid_intent(self, clf):
        result = clf.predict("The product is broken and defective")
        valid_intents = {"Shipping_Inquiry", "Refund_Request", "Product_Dispute", "Price_Inquiry", "Spam"}
        assert result["predicted_intent"] in valid_intents

    def test_predict_valid_action(self, clf):
        result = clf.predict("Test message for classification")
        assert result["action_taken"] in ("Auto-Reply", "Escalated", "Discarded")
