"""
MLflow-tracked training pipeline for Customer Intent Classification.
Uses TF-IDF + Logistic Regression with comprehensive experiment tracking.
"""

import json
import os
import sys
import warnings
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "training", "data", "training_data.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "intent_classifier.joblib")
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")
EXPERIMENT_NAME = "intent-classification"

# Hyperparameters
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SUBLINEAR_TF = True
LR_C = 1.0
LR_MAX_ITER = 1000
LR_SOLVER = "lbfgs"
LR_CLASS_WEIGHT = "balanced"
CV_FOLDS = 5


def load_training_data(data_path: str) -> tuple[list[str], list[str]]:
    """Load and validate training data from JSON file."""
    print(f"📂 Loading training data from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    labels = [item["intent"] for item in data]

    unique_labels = sorted(set(labels))
    print(f"✅ Loaded {len(texts)} samples across {len(unique_labels)} classes")
    for label in unique_labels:
        count = labels.count(label)
        print(f"   • {label}: {count} samples")

    return texts, labels


def build_pipeline(
    max_features: int,
    ngram_range: tuple,
    sublinear_tf: bool,
    C: float,
    max_iter: int,
    solver: str,
    class_weight: str,
) -> Pipeline:
    """Build the TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=sublinear_tf,
                    strip_accents="unicode",
                    lowercase=True,
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver=solver,
                    class_weight=class_weight,
                    multi_class="multinomial",
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_evaluate(
    texts: list[str], labels: list[str], pipeline: Pipeline, cv_folds: int
) -> dict:
    """Train the model with cross-validation and return metrics."""
    print(f"\n🔄 Running {cv_folds}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_accuracy = cross_val_score(pipeline, texts, labels, cv=skf, scoring="accuracy")
    cv_f1 = cross_val_score(
        pipeline, texts, labels, cv=skf, scoring="f1_weighted"
    )
    cv_precision = cross_val_score(
        pipeline, texts, labels, cv=skf, scoring="precision_weighted"
    )
    cv_recall = cross_val_score(
        pipeline, texts, labels, cv=skf, scoring="recall_weighted"
    )

    metrics = {
        "cv_accuracy_mean": float(np.mean(cv_accuracy)),
        "cv_accuracy_std": float(np.std(cv_accuracy)),
        "cv_f1_weighted_mean": float(np.mean(cv_f1)),
        "cv_f1_weighted_std": float(np.std(cv_f1)),
        "cv_precision_weighted_mean": float(np.mean(cv_precision)),
        "cv_precision_weighted_std": float(np.std(cv_precision)),
        "cv_recall_weighted_mean": float(np.mean(cv_recall)),
        "cv_recall_weighted_std": float(np.std(cv_recall)),
    }

    print(f"   Accuracy:  {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    print(f"   F1 Score:  {metrics['cv_f1_weighted_mean']:.4f} ± {metrics['cv_f1_weighted_std']:.4f}")
    print(f"   Precision: {metrics['cv_precision_weighted_mean']:.4f} ± {metrics['cv_precision_weighted_std']:.4f}")
    print(f"   Recall:    {metrics['cv_recall_weighted_mean']:.4f} ± {metrics['cv_recall_weighted_std']:.4f}")

    # Final fit on all data
    print("\n📊 Training final model on full dataset...")
    pipeline.fit(texts, labels)

    # Full dataset metrics
    y_pred = pipeline.predict(texts)
    metrics["train_accuracy"] = float(accuracy_score(labels, y_pred))
    metrics["train_f1_weighted"] = float(f1_score(labels, y_pred, average="weighted"))
    metrics["train_precision_weighted"] = float(
        precision_score(labels, y_pred, average="weighted")
    )
    metrics["train_recall_weighted"] = float(
        recall_score(labels, y_pred, average="weighted")
    )
    metrics["num_samples"] = len(texts)
    metrics["num_classes"] = len(set(labels))

    print("\n📋 Classification Report (Full Training Set):")
    print(classification_report(labels, y_pred))

    return metrics


def main():
    """Main training entry point with MLflow experiment tracking."""
    print("=" * 65)
    print("  🚀 Intent Classification — Training Pipeline")
    print("=" * 65)

    # ── Load data ────────────────────────────────────────────────────────
    texts, labels = load_training_data(DATA_PATH)

    # ── Set up MLflow ────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"\n🔬 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"🔬 Experiment Name:    {EXPERIMENT_NAME}")

    with mlflow.start_run(run_name="tfidf-logistic-regression") as run:
        print(f"🔬 Run ID: {run.info.run_id}\n")

        # Log hyperparameters
        params = {
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "tfidf_ngram_range": str(TFIDF_NGRAM_RANGE),
            "tfidf_sublinear_tf": TFIDF_SUBLINEAR_TF,
            "lr_C": LR_C,
            "lr_max_iter": LR_MAX_ITER,
            "lr_solver": LR_SOLVER,
            "lr_class_weight": LR_CLASS_WEIGHT,
            "cv_folds": CV_FOLDS,
            "model_type": "TF-IDF + LogisticRegression",
        }
        mlflow.log_params(params)

        # Build pipeline
        pipeline = build_pipeline(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=TFIDF_SUBLINEAR_TF,
            C=LR_C,
            max_iter=LR_MAX_ITER,
            solver=LR_SOLVER,
            class_weight=LR_CLASS_WEIGHT,
        )

        # Train and evaluate
        metrics = train_and_evaluate(texts, labels, pipeline, CV_FOLDS)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log the model via MLflow sklearn
        mlflow.sklearn.log_model(pipeline, "model")

        # ── Save model locally ───────────────────────────────────────────
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        print(f"\n💾 Model saved to: {MODEL_PATH}")

        # Log model artifact
        mlflow.log_artifact(MODEL_PATH, "model_artifacts")

        # Log training data info
        mlflow.log_artifact(DATA_PATH, "training_data")

        print(f"\n✅ Training complete! MLflow Run ID: {run.info.run_id}")
        print(f"   View results: mlflow ui --backend-store-uri file:///{MLFLOW_TRACKING_URI}")

    print("\n" + "=" * 65)
    print("  ✨ Pipeline finished successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
