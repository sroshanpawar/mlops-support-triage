# 📋 MLOps-Driven Customer Intent Classifier — Project Details

---

## 1. Project Title

**Smart Post-Purchase Support Triage System**
*An MLOps-Driven Customer Intent Classification & Automated Response Platform for E-Commerce*

---

## 2. Project Summary

This project is a production-grade, end-to-end **AI-powered customer support triage system** designed for e-commerce businesses. It automatically classifies incoming customer messages into pre-defined intent categories, determines the appropriate action (auto-reply, escalate, or discard), and generates templated responses — all while tracking experiments, versioning data, and maintaining a CI/CD pipeline. The system combines **Natural Language Processing (NLP)**, **Machine Learning (ML)**, and **MLOps best practices** to reduce human support workload, accelerate response times, and ensure consistent service quality.

---

## 3. Problem Statement

E-commerce businesses receive thousands of repetitive customer support messages daily — order tracking queries, refund requests, pricing questions, and spam. Manually reading, triaging, and responding to each message is:

- **Time-consuming**: Human agents spend significant time on routine inquiries that could be automated.
- **Inconsistent**: Response quality varies depending on the agent and workload.
- **Expensive**: Staffing 24/7 support teams is costly, especially for high-volume businesses.
- **Slow**: Customers experience long wait times for simple queries.

This system addresses all of the above by automating the classification and response pipeline with AI, while keeping humans in the loop for sensitive or ambiguous cases.

---

## 4. Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Build a multi-class text classification model to identify customer intent | ✅ Complete |
| 2 | Implement confidence-based triage logic (auto-reply / escalate / discard) | ✅ Complete |
| 3 | Develop a RESTful API backend using FastAPI for real-time predictions | ✅ Complete |
| 4 | Create a professional CRM dashboard with Streamlit for monitoring | ✅ Complete |
| 5 | Integrate MLflow for experiment tracking and model versioning | ✅ Complete |
| 6 | Set up DVC for training data version control | ✅ Complete |
| 7 | Containerize the application with Docker (multi-stage builds) | ✅ Complete |
| 8 | Automate CI/CD with GitHub Actions for testing, training, and deployment | ✅ Complete |
| 9 | Write automated tests with Pytest for API and classifier logic | ✅ Complete |

---

## 5. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM ARCHITECTURE                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐     ┌──────────────────┐     ┌────────────────────┐      │
│   │   Customer    │     │   FastAPI         │     │   Streamlit CRM    │      │
│   │   Messages    │────▶│   Backend API     │◀────│   Dashboard        │      │
│   │   (JSON)      │     │   (Port 8000)     │     │   (Port 8501)      │      │
│   └──────────────┘     └────────┬─────────┘     └────────────────────┘      │
│                                  │                                           │
│                     ┌────────────┼────────────┐                              │
│                     ▼            ▼            ▼                              │
│              ┌────────────┐ ┌──────────┐ ┌──────────────┐                   │
│              │  Intent     │ │  SQLite   │ │  Outbound    │                   │
│              │  Classifier │ │  Database │ │  Replies.json│                   │
│              │  (Joblib)   │ │           │ │              │                   │
│              └──────┬─────┘ └──────────┘ └──────────────┘                   │
│                     │                                                        │
│              ┌──────┴─────┐                                                  │
│              │  ML Model   │                                                  │
│              │  TF-IDF +   │                                                  │
│              │  Logistic   │                                                  │
│              │  Regression │                                                  │
│              └─────────────┘                                                  │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                       MLOps Layer                                    │    │
│   │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐   │    │
│   │   │  MLflow   │   │   DVC    │   │  Docker  │   │ GitHub       │   │    │
│   │   │  Tracking │   │  Data    │   │  Multi-  │   │ Actions      │   │    │
│   │   │  (P:5000) │   │  Version │   │  Stage   │   │ CI/CD        │   │    │
│   │   └──────────┘   └──────────┘   └──────────┘   └──────────────┘   │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Technology Stack

### 6.1 Core Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11 | Primary programming language |
| **FastAPI** | 0.115.6 | REST API framework for serving predictions |
| **Uvicorn** | 0.34.0 | ASGI server for running FastAPI |
| **SQLAlchemy** | 2.0.36 | ORM for SQLite database interaction |
| **Pydantic** | 2.10.3 | Data validation and schema enforcement |

### 6.2 Machine Learning
| Technology | Version | Purpose |
|------------|---------|---------|
| **Scikit-Learn** | 1.6.0 | ML pipeline (TF-IDF Vectorizer + Logistic Regression) |
| **Joblib** | 1.4.2 | Model serialization and deserialization |
| **NumPy** | 1.26.4 | Numerical computations |
| **Pandas** | 2.2.3 | Data manipulation and analysis |

### 6.3 MLOps & DevOps
| Technology | Version | Purpose |
|------------|---------|---------|
| **MLflow** | 2.19.0 | Experiment tracking, model registry, metric logging |
| **DVC** | 3.56.0 | Data version control for training datasets |
| **Docker** | Multi-stage | Application containerization |
| **Docker Compose** | 3.9 | Multi-container orchestration |
| **GitHub Actions** | v4 | CI/CD pipeline automation |

### 6.4 Frontend & Visualization
| Technology | Version | Purpose |
|------------|---------|---------|
| **Streamlit** | 1.41.0 | CRM dashboard web interface |
| **Plotly** | 5.24.1 | Interactive charts and graphs |
| **Requests** | 2.32.3 | HTTP client for API communication |

### 6.5 Testing
| Technology | Version | Purpose |
|------------|---------|---------|
| **Pytest** | 8.3.4 | Unit and integration testing framework |
| **HTTPX** | 0.28.1 | Async HTTP client for FastAPI test client |

---

## 7. Project Structure

```
MLOps-Driven Customer Intent Classifier/
│
├── backend/                        # FastAPI Backend Application
│   ├── __init__.py                 # Package initializer
│   ├── main.py                     # FastAPI app, routes, and endpoints
│   ├── classifier.py               # Intent classifier + triage logic
│   ├── models.py                   # Pydantic request/response schemas
│   └── database.py                 # SQLAlchemy ORM models & DB setup
│
├── frontend/                       # Streamlit CRM Dashboard
│   └── app.py                      # Multi-page dashboard with charts & controls
│
├── training/                       # ML Training Pipeline
│   ├── train.py                    # MLflow-tracked training script
│   └── data/
│       └── training_data.json      # Labeled training dataset (256 samples)
│
├── models/                         # Trained Model Artifacts
│   └── intent_classifier.joblib    # Serialized sklearn pipeline
│
├── tests/                          # Automated Test Suite
│   ├── __init__.py                 # Package initializer
│   ├── test_api.py                 # FastAPI endpoint tests (14 tests)
│   └── test_classifier.py          # Classifier logic tests (15 tests)
│
├── .github/workflows/
│   └── mlops.yml                   # GitHub Actions CI/CD pipeline
│
├── data/                           # Runtime data directory
├── mlruns/                         # MLflow experiment tracking data
│
├── Dockerfile                      # Multi-stage Docker build
├── docker-compose.yml              # Container orchestration
├── dvc_setup.sh                    # DVC initialization script
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── .dvcignore                      # DVC ignore rules
├── outbound_replies.json           # Generated auto-reply log
├── support_triage.db               # SQLite database (runtime)
└── README.md                       # Project readme
```

---

## 8. Intent Classification Categories

The model classifies customer messages into **5 distinct intent categories**:

| Intent | Samples | Risk Level | Default Action |
|--------|---------|------------|----------------|
| **Shipping_Inquiry** | 51 | 🟢 Routine | Auto-Reply (if confidence > 85%) |
| **Refund_Request** | 51 | 🔴 High-Risk | Always Escalated to human agent |
| **Product_Dispute** | 50 | 🔴 High-Risk | Always Escalated to human agent |
| **Price_Inquiry** | 50 | 🟢 Routine | Auto-Reply (if confidence > 85%) |
| **Spam** | 50 | 🟢 Routine | Discarded (if confidence > 85%) |
| **Total** | **252** | | |

---

## 9. Core Triage Logic

The system uses a **confidence-based escalation strategy** to determine the appropriate action for each classified message:

```
                         ┌──────────────────────┐
                         │ Incoming Customer Msg │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │   ML Model Predicts   │
                         │   Intent + Confidence  │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             ┌────────────┐  ┌────────────┐  ┌────────────┐
             │ Spam        │  │ Routine     │  │ High-Risk  │
             │ (Conf>85%)  │  │ (Conf>85%)  │  │ (Any Conf) │
             └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
                    │               │               │
                    ▼               ▼               ▼
             ┌────────────┐  ┌────────────┐  ┌────────────┐
             │ 🗑️ DISCARD │  │ ✅ AUTO-   │  │ 🚨 ESCALATE│
             │ (No Reply)  │  │   REPLY     │  │ (To Human) │
             └────────────┘  └────────────┘  └────────────┘
```

### Decision Rules

| Condition | Action | Description |
|-----------|--------|-------------|
| `Spam` + confidence > 85% | **Discarded** | Spam filtered out silently, no response generated |
| Routine intent + confidence > 85% | **Auto-Reply** | Templated response sent automatically to customer |
| Any intent + confidence ≤ 85% | **Escalated** | Low confidence → routed to human agent for review |
| High-risk intent + any confidence | **Escalated** | Refunds & disputes always require human handling |

### Confidence Threshold
- **HIGH_CONFIDENCE_THRESHOLD = 0.85** (85%)
- Messages at or below this threshold are **always escalated** regardless of intent category

---

## 10. ML Model Details

### 10.1 Algorithm
- **Pipeline**: TF-IDF Vectorizer → Logistic Regression (Multinomial)
- **Framework**: Scikit-Learn `Pipeline` for reproducible preprocessing + prediction

### 10.2 TF-IDF Vectorizer Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_features` | 5,000 | Maximum vocabulary size |
| `ngram_range` | (1, 2) | Unigrams and bigrams |
| `sublinear_tf` | True | Apply sublinear TF scaling (1 + log(tf)) |
| `strip_accents` | unicode | Normalize unicode characters |
| `lowercase` | True | Convert text to lowercase |
| `stop_words` | english | Remove English stop words |

### 10.3 Logistic Regression Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `C` | 1.0 | Inverse regularization strength |
| `max_iter` | 1,000 | Maximum solver iterations |
| `solver` | lbfgs | Optimization algorithm |
| `class_weight` | balanced | Adjust weights inversely proportional to class frequency |
| `multi_class` | multinomial | Multi-class strategy |
| `random_state` | 42 | Reproducibility seed |

### 10.4 Cross-Validation
- **Strategy**: 5-fold Stratified K-Fold
- **Metrics tracked**: Accuracy, F1 (weighted), Precision (weighted), Recall (weighted)

### 10.5 Model Artifact
- **Format**: Joblib-serialized Scikit-Learn pipeline
- **Path**: `models/intent_classifier.joblib`
- **Contains**: Both the TF-IDF vectorizer and Logistic Regression classifier in a single pipeline object

---

## 11. API Endpoints

The FastAPI backend exposes the following RESTful endpoints:

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `GET` | `/health` | System health check | — | `HealthResponse` |
| `POST` | `/predict` | Classify a single message | `MessageInput` | `PredictionResult` |
| `POST` | `/batch-predict` | Classify a batch of messages | `BatchInput` | `BatchResult` |
| `POST` | `/upload-traffic` | Upload a JSON file for batch processing | `multipart/form-data` | `BatchResult` |
| `GET` | `/messages` | Retrieve processed message history | Query params: `intent`, `action`, `limit`, `offset` | Paginated messages list |
| `GET` | `/stats` | Aggregated dashboard statistics | — | `StatsResponse` |

### Key Data Models

**MessageInput** (Request):
```json
{
  "id": "MSG-001",
  "customer_name": "John Doe",
  "email": "john@example.com",
  "text": "Where is my order?",
  "timestamp": "2026-04-09T10:00:00Z",
  "channel": "email"
}
```

**PredictionResult** (Response):
```json
{
  "message_id": "MSG-001",
  "customer_name": "John Doe",
  "original_text": "Where is my order?",
  "predicted_intent": "Shipping_Inquiry",
  "confidence_score": 0.9412,
  "action_taken": "Auto-Reply",
  "auto_reply_text": "Thank you for reaching out! ...",
  "processed_at": "2026-04-09T10:00:01"
}
```

---

## 12. Database Schema

The system uses **SQLite** with **SQLAlchemy ORM** for persistent storage.

### Table: `processed_messages`
| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-incrementing primary key |
| `message_id` | String(100) | External message identifier |
| `customer_name` | String(200) | Customer name |
| `customer_email` | String(200) | Customer email |
| `channel` | String(50) | Communication channel (email, chat, social_media) |
| `original_text` | Text | Original customer message |
| `predicted_intent` | String(50) | Model-predicted intent category |
| `confidence_score` | Float | Model confidence (0.0 – 1.0) |
| `action_taken` | String(50) | Action: Auto-Reply / Escalated / Discarded |
| `auto_reply_text` | Text | Generated reply text (nullable) |
| `processed_at` | DateTime | Processing timestamp |
| `message_timestamp` | DateTime | Original message timestamp (nullable) |

### Table: `model_metrics`
| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-incrementing primary key |
| `run_id` | String(100) | MLflow run identifier |
| `accuracy` | Float | Model accuracy |
| `f1_score` | Float | Weighted F1 score |
| `precision` | Float | Weighted precision |
| `recall` | Float | Weighted recall |
| `num_samples` | Integer | Training sample count |
| `trained_at` | DateTime | Training timestamp |

---

## 13. CRM Dashboard (Frontend)

The Streamlit-based CRM dashboard provides a professional, dark-themed interface for operations teams:

### Pages

| Page | Functionality |
|------|---------------|
| **📊 Dashboard** | KPI metric cards (Total Messages, Auto-Replied, Escalated, Discarded, Avg Confidence, Escalation Rate), Intent Distribution pie chart, Action Breakdown bar chart, Confidence Score histogram |
| **📤 Upload & Process** | Upload `simulated_traffic.json` files, preview data, batch-process with progress bar, view results in color-coded table |
| **💬 Message Explorer** | Browse processed messages with filters (by intent, action, limit), paginated data table with row highlighting |
| **🔍 Live Classifier** | Real-time single-message classification, instant intent prediction, confidence display, generated response preview |

### Design Features
- Dark gradient theme with glassmorphism-inspired card styling
- `Inter` font from Google Fonts
- Color-coded badges and confidence indicators
- Plotly interactive charts with transparent backgrounds
- Responsive 6-column KPI grid layout
- Row-level highlighting: green for auto-replies, pink/red for escalated

---

## 14. MLOps Pipeline

### 14.1 MLflow — Experiment Tracking
- **Tracking URI**: Local filesystem (`mlruns/` directory)
- **Experiment Name**: `intent-classification`
- **What is logged**:
  - All hyperparameters (TF-IDF config, LR config, CV folds)
  - Cross-validation metrics (accuracy, F1, precision, recall — means and stds)
  - Full training-set metrics
  - Trained model artifact via `mlflow.sklearn.log_model()`
  - Training data file as artifact
- **MLflow UI**: Available on port 5000 via Docker Compose

### 14.2 DVC — Data Version Control
- **Tracked file**: `training/data/training_data.json`
- **Remote**: Configurable (local storage at `/tmp/dvc-storage` for demo)
- **Setup script**: `dvc_setup.sh` automates initialization, tracking, and push

### 14.3 Docker — Containerization
- **Build Strategy**: Multi-stage Dockerfile with 4 stages:
  - `base` — Python 3.11-slim with system dependencies
  - `dependencies` — pip-installed Python packages
  - `application` — Application code copied
  - `production` — Final image with startup script, health check, exposed ports
- **Ports Exposed**: 8000 (FastAPI), 8501 (Streamlit)
- **Health Check**: HTTP check against `/health` endpoint (30s interval, 60s start period)

### 14.4 Docker Compose — Orchestration
Two services defined:
| Service | Container Name | Ports | Description |
|---------|---------------|-------|-------------|
| `app` | `support-triage-system` | 8000, 8501 | Full application stack (API + Dashboard) |
| `mlflow` | `mlflow-ui` | 5000 | MLflow tracking UI |

### 14.5 GitHub Actions — CI/CD Pipeline
Three-job pipeline triggered on pushes to `main`/`develop` and PRs to `main`:

| Job | Name | Dependencies | Steps |
|-----|------|--------------|-------|
| **1** | 🧪 Test Suite | — | Checkout → Setup Python 3.11 → Install deps → Train model → Run Pytest → Upload test results |
| **2** | 🤖 Train & Validate Model | Job 1 | Checkout → Setup Python → Install deps → Train with MLflow → Verify artifacts → Upload model + MLflow runs |
| **3** | 🐳 Build Docker Image | Jobs 1 & 2 (main branch only) | Checkout → Download trained model → Setup Buildx → Build image → Verify image |

---

## 15. Testing Strategy

### Test Suite Overview
| File | Tests | Coverage Area |
|------|-------|---------------|
| `tests/test_classifier.py` | 15 tests | Classifier configuration, triage logic, model predictions |
| `tests/test_api.py` | 14 tests | Health, predict, batch-predict, messages, and stats endpoints |

### Test Categories

**Configuration Tests** — Verify that classifier constants (thresholds, intent sets, reply templates) are correctly defined.

**Triage Logic Tests** — Validate the decision engine:
- Routine intent + high confidence → Auto-Reply
- Routine intent + low confidence → Escalate
- High-risk intent + any confidence → Escalate
- Spam + high confidence → Discard
- Boundary condition at exactly the 85% threshold

**Model Prediction Tests** — End-to-end prediction validation (requires trained model):
- Correct output structure (dict with required keys)
- Confidence score in valid range [0, 1]
- Predicted intent is one of the 5 valid classes
- Action is one of the 3 valid actions

**API Endpoint Tests** — HTTP-level integration tests:
- Health check returns 200 with correct schema
- Single prediction returns valid classification result
- Batch prediction processes multiple messages correctly
- Messages endpoint supports filtering and pagination
- Stats endpoint returns complete statistics structure

### Run Tests
```bash
pytest tests/ -v --tb=short
```

---

## 16. How to Run

### Prerequisites
- Python 3.11+
- pip (Python package manager)
- Docker & Docker Compose (optional, for containerized deployment)

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/sroshanpawar/mlops-support-triage.git
cd mlops-support-triage

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the ML model
python training/train.py

# 5. Start the FastAPI backend
uvicorn backend.main:app --reload --port 8000

# 6. Start the Streamlit dashboard (in a new terminal)
streamlit run frontend/app.py --server.port 8501

# 7. Run tests
pytest tests/ -v
```

### Docker Deployment

```bash
# Build and run all services
docker-compose up --build

# Access:
#   API        → http://localhost:8000
#   Dashboard  → http://localhost:8501
#   MLflow UI  → http://localhost:5000
```

---

## 17. Key Features Summary

| Feature | Description |
|---------|-------------|
| 🤖 **AI-Powered Classification** | TF-IDF + Logistic Regression pipeline classifies messages into 5 intent categories |
| ⚡ **Confidence-Based Triage** | Smart routing — auto-reply for routine queries, escalate for risky/uncertain ones |
| 📝 **Automated Responses** | Predefined reply templates for routine intents, escalation messages for human follow-up |
| 🗑️ **Spam Filtering** | High-confidence spam is silently discarded without generating responses |
| 📊 **Real-Time Dashboard** | KPI cards, pie charts, bar charts, histograms — all updating from live API data |
| 📤 **Batch Processing** | Upload JSON files with hundreds of messages for bulk classification |
| 🔍 **Live Classifier** | Interactive text input for real-time classification testing |
| 💾 **Persistent Storage** | SQLite database stores every processed message with full audit trail |
| 🔬 **Experiment Tracking** | MLflow logs every training run with hyperparameters, metrics, and model artifacts |
| 📦 **Data Versioning** | DVC tracks changes to training data with remote storage support |
| 🐳 **Containerized Deployment** | Multi-stage Docker build with health checks and volume persistence |
| 🔄 **CI/CD Automation** | GitHub Actions pipeline: test → train → build Docker → deploy |
| 🧪 **Comprehensive Testing** | 29 automated tests covering logic, API, and model predictions |
| 🌐 **CORS-Enabled API** | Cross-origin support for frontend-backend communication |

---

## 18. Future Scope

| Enhancement | Description |
|-------------|-------------|
| 🧠 **Transformer Models** | Replace TF-IDF + LR with DistilBERT or a fine-tuned Transformer for higher accuracy |
| 📊 **Model Monitoring** | Add data drift detection and model performance monitoring in production |
| 🔄 **Online Learning** | Implement feedback loops where human agent corrections retrain the model |
| 🌐 **Multi-Language Support** | Extend classification to support non-English customer messages |
| 📧 **Email/Slack Integration** | Connect auto-replies directly to email/Slack for real delivery |
| 🗃️ **PostgreSQL Migration** | Migrate from SQLite to PostgreSQL for production-grade data storage |
| 📈 **A/B Testing** | Compare multiple model versions side-by-side in production |
| 🔐 **Authentication** | Add JWT-based authentication for API and dashboard access |
| ☁️ **Cloud Deployment** | Deploy to AWS/GCP/Azure with Kubernetes orchestration |
| 📱 **Custom Web Frontend** | Replace Streamlit with a professional React/Next.js frontend |

---

## 19. Author

**Roshan Pawar**
- GitHub: [sroshanpawar](https://github.com/sroshanpawar)

---

> *This document was auto-generated based on the current state of the codebase and reflects all implemented features, architecture, and configurations as of April 2026.*
