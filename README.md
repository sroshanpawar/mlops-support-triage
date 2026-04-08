# 🎯 Smart Post-Purchase Support Triage System

An MLOps-driven AI system that automatically classifies customer support messages, determines appropriate actions based on confidence-based escalation logic, and generates auto-replies for routine inquiries.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard (:8501)                   │
│         📊 Analytics │ 📤 Upload │ 💬 Explorer │ 🔍 Live       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP API
┌──────────────────────────▼──────────────────────────────────────┐
│                    FastAPI Backend (:8000)                       │
│      /predict │ /batch-predict │ /messages │ /stats │ /health   │
├─────────────────────────────────────────────────────────────────┤
│                    Intent Classifier                            │
│              TF-IDF + Logistic Regression                       │
│         Confidence-Based Escalation Engine                      │
├──────────────┬──────────────────────────────┬───────────────────┤
│   SQLite DB  │   outbound_replies.json      │  MLflow Tracking  │
└──────────────┴──────────────────────────────┴───────────────────┘
```

## 📋 Intent Categories

| Intent | Risk Level | High-Confidence Action |
|--------|-----------|----------------------|
| `Shipping_Inquiry` | Routine | Auto-reply with tracking info |
| `Price_Inquiry` | Routine | Auto-reply with pricing info |
| `Refund_Request` | High-Risk | Always escalate to human |
| `Product_Dispute` | High-Risk | Always escalate to human |
| `Spam` | Routine | Auto-discard, log only |

**Escalation Logic:**
- Confidence > 85% + Routine intent → **Auto-Reply**
- Confidence > 85% + Spam → **Discard**
- Confidence ≤ 85% (any intent) → **Escalate**
- High-Risk intent (any confidence) → **Escalate**

---

## 🚀 Quick Start

### 1. Set Up Environment

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python training/train.py
```

This will:
- Load the synthetic training dataset (250 samples, 5 categories)
- Train a TF-IDF + Logistic Regression pipeline
- Log parameters and metrics to MLflow
- Save the model to `models/intent_classifier.joblib`

### 3. Start the Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. Launch the Dashboard

```bash
streamlit run frontend/app.py
```

Open your browser to `http://localhost:8501`

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 🔬 MLflow UI

```bash
mlflow ui --backend-store-uri file:///./mlruns
```

Open `http://localhost:5000` to view experiment tracking.

---

## 🐳 Docker

### Build & Run

```bash
# Build the image
docker build -t support-triage-system .

# Run with Docker Compose (recommended)
docker-compose up --build
```

Services:
- **App** → `http://localhost:8000` (API) + `http://localhost:8501` (Dashboard)
- **MLflow** → `http://localhost:5000`

---

## 📦 DVC Setup

```bash
# Initialize DVC and track training data
bash dvc_setup.sh

# Or manually:
dvc init
dvc add training/data/training_data.json
```

---

## 📁 Project Structure

```
├── .github/workflows/
│   └── mlops.yml              # CI/CD: pytest + Docker build
├── backend/
│   ├── main.py                # FastAPI application
│   ├── database.py            # SQLite + SQLAlchemy ORM
│   ├── models.py              # Pydantic request/response schemas
│   └── classifier.py          # Model loading + triage logic
├── frontend/
│   └── app.py                 # Streamlit CRM dashboard
├── training/
│   ├── train.py               # MLflow-tracked training pipeline
│   └── data/
│       └── training_data.json # Synthetic dataset (250 samples)
├── models/                    # Trained model artifacts
├── tests/
│   ├── test_api.py            # FastAPI endpoint tests
│   └── test_classifier.py     # Classifier logic tests
├── data/
│   └── simulated_traffic.json # Sample upload file (20 messages)
├── Dockerfile                 # Multi-stage container build
├── docker-compose.yml         # Full stack orchestration
├── requirements.txt           # Python dependencies
├── dvc_setup.sh               # DVC initialization script
└── README.md
```

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `POST` | `/predict` | Classify single message |
| `POST` | `/batch-predict` | Batch classify messages |
| `POST` | `/upload-traffic` | Upload JSON file for processing |
| `GET` | `/messages` | Retrieve processed messages |
| `GET` | `/stats` | Dashboard statistics |

---

## ⚙️ Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit + Plotly
- **ML:** Scikit-Learn (TF-IDF + Logistic Regression)
- **Database:** SQLite + SQLAlchemy
- **MLOps:** MLflow (tracking) + DVC (data versioning)
- **CI/CD:** GitHub Actions
- **Containers:** Docker + Docker Compose
