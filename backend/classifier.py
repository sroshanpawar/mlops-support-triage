"""
Intent classifier module — handles model loading, prediction, and
confidence-based escalation logic.
"""

import json
import os
from datetime import datetime
import joblib
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_classifier.joblib")
OUTBOUND_REPLIES_PATH = os.path.join(BASE_DIR, "outbound_replies.json")

# ── Confidence thresholds ────────────────────────────────────────────────────
HIGH_CONFIDENCE_THRESHOLD = 0.30

# ── Intent risk classification ───────────────────────────────────────────────
ROUTINE_INTENTS = {"Shipping_Inquiry", "Price_Inquiry", "Spam"}
HIGH_RISK_INTENTS = {"Refund_Request", "Product_Dispute"}

# ── Predefined auto-reply templates ─────────────────────────────────────────
AUTO_REPLIES = {
    "Shipping_Inquiry": (
        "Thank you for reaching out! We understand you have a question about "
        "your shipment. Your order is currently being processed through our "
        "logistics network. You can track your package using the tracking link "
        "sent to your registered email. Typical delivery times are 5-7 business "
        "days for standard shipping and 2-3 business days for express. If your "
        "tracking hasn't updated in over 48 hours, please reply to this message "
        "with your order number and we'll investigate immediately."
    ),
    "Price_Inquiry": (
        "Thank you for your interest in our products! We offer competitive "
        "pricing across all categories. For specific pricing details, please "
        "visit our product pages where current prices and any active promotions "
        "are displayed. We also offer bulk discounts for orders over 10 units — "
        "reach out to our sales team at sales@support.com for a custom quote. "
        "Don't forget to check our 'Deals' section for the latest offers!"
    ),
    "Spam": None,  # Spam is discarded, no reply generated
}

ESCALATION_MESSAGE = (
    "Your message has been flagged for priority review by our support team. "
    "A specialist will reach out to you within 2-4 business hours. We take "
    "your concern seriously and will work to resolve it as quickly as possible. "
    "Reference ID: {message_id}"
)


class IntentClassifier:
    """Loads trained model and provides prediction + triage logic."""

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the trained sklearn pipeline from disk."""
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️  No model found at {MODEL_PATH}. Run training first.")

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, text: str) -> dict:
        """
        Classify a single message and return the intent, confidence,
        action decision, and optional auto-reply text.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Run training/train.py first.")

        # Get prediction probabilities
        probas = self.model.predict_proba([text])[0]
        predicted_idx = np.argmax(probas)
        predicted_intent = self.model.classes_[predicted_idx]
        confidence = float(probas[predicted_idx])

        # Determine action based on confidence + intent risk
        action, reply_text = self._determine_action(
            predicted_intent, confidence, message_id="N/A"
        )

        return {
            "predicted_intent": predicted_intent,
            "confidence_score": round(confidence, 4),
            "action_taken": action,
            "auto_reply_text": reply_text,
            "all_probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(self.model.classes_, probas)
            },
        }

    def _determine_action(
        self, intent: str, confidence: float, message_id: str
    ) -> tuple[str, str | None]:
        """
        Core triage logic:
        - High confidence + routine intent → Auto-Reply
        - Spam → Discard (no reply)
        - Low confidence OR high-risk intent → Escalate
        """
        if intent == "Spam" and confidence > HIGH_CONFIDENCE_THRESHOLD:
            return "Discarded", None

        if confidence > HIGH_CONFIDENCE_THRESHOLD and intent in ROUTINE_INTENTS:
            reply_text = AUTO_REPLIES.get(intent)
            return "Auto-Reply", reply_text

        # Low confidence or high-risk intent → escalate
        reply_text = ESCALATION_MESSAGE.format(message_id=message_id)
        return "Escalated", reply_text

    def append_outbound_reply(self, reply_payload: dict):
        """Append a generated reply to the local outbound_replies.json file."""
        replies = []
        if os.path.exists(OUTBOUND_REPLIES_PATH):
            try:
                with open(OUTBOUND_REPLIES_PATH, "r", encoding="utf-8") as f:
                    replies = json.load(f)
            except (json.JSONDecodeError, IOError):
                replies = []

        replies.append(reply_payload)

        with open(OUTBOUND_REPLIES_PATH, "w", encoding="utf-8") as f:
            json.dump(replies, f, indent=2, default=str)


# Singleton instance
classifier = IntentClassifier()
