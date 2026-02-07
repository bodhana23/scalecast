<h1 align="center">🚀 ScaleCast</h1>

<p align="center">
  <strong>End-to-End MLOps Pipeline for Demand Forecasting</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Airflow-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white" alt="Airflow" />
  <img src="https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="AWS" />
  <img src="https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" alt="GitHub Actions" />
</p>

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ScaleCast MLOps Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────┐      ┌─────────────────┐      ┌────────────┐      ┌─────────┐   │
│   │   S3    │─────▶│      Great      │─────▶│ PostgreSQL │─────▶│   ML    │   │
│   │  (raw)  │      │  Expectations   │      │ (warehouse)│      │Training │   │
│   └─────────┘      │  (validation)   │      └────────────┘      └────┬────┘   │
│                    └─────────────────┘                               │        │
│                            │                                         ▼        │
│                            │ ❌ Circuit                        ┌─────────┐    │
│                            │    Breaker                        │   S3    │    │
│                            ▼                                   │(models) │    │
│                    ┌───────────────┐                           └────┬────┘    │
│                    │  Alert/Stop   │                                │         │
│                    │   Pipeline    │                                ▼         │
│                    └───────────────┘                          ┌─────────┐     │
│                                                               │ FastAPI │     │
│                                                               │(serving)│     │
│                                                               └─────────┘     │
│                                                                                 │
│                    ┌─────────────────────────────────────┐                     │
│                    │         Apache Airflow              │                     │
│                    │      (Orchestration Layer)          │                     │
│                    └─────────────────────────────────────┘                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- 🛡️ **Automated Data Validation** — Circuit breaker pattern stops pipeline on bad data
- 📦 **Data Versioning** — Track datasets with DVC for reproducibility
- ⚙️ **Workflow Orchestration** — Apache Airflow schedules and monitors pipelines
- 🌐 **Model Serving** — FastAPI provides low-latency prediction endpoints
- 🔄 **CI/CD Pipeline** — GitHub Actions for linting, testing, and Docker builds
- 🐳 **Infrastructure as Code** — Fully containerized with Docker Compose

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | Apache Airflow 2.7 | Workflow scheduling and monitoring |
| **Data Validation** | Great Expectations | Schema and quality checks |
| **Database** | PostgreSQL 15 | Data warehouse for training data |
| **ML Framework** | scikit-learn | Random Forest demand forecasting |
| **API** | FastAPI 0.104 | Model serving with auto-generated docs |
| **Cloud Storage** | AWS S3 | Artifact and model storage |
| **Version Control** | DVC 3.30 | Data and model versioning |
| **CI/CD** | GitHub Actions | Automated testing and builds |
| **Containerization** | Docker Compose | Local development environment |

---

## 📁 Project Structure

```
scalecast/
├── 📂 airflow/
│   ├── dags/                   # Airflow DAG definitions
│   │   └── scalecast_pipeline.py
│   ├── logs/                   # Airflow logs (gitignored)
│   └── plugins/                # Custom Airflow plugins
├── 📂 configs/
│   └── config.yaml             # Central configuration
├── 📂 data/
│   ├── raw/                    # Raw input data (DVC tracked)
│   └── processed/              # Processed datasets
├── 📂 models/                  # Trained model artifacts
├── 📂 scripts/
│   ├── setup_aws.py            # AWS S3 bucket setup
│   ├── generate_keys.py        # Generate Fernet keys
│   └── init_db.sql             # Database schema initialization
├── 📂 src/
│   ├── api/                    # FastAPI application
│   │   └── main.py
│   ├── data_ingestion/         # Data loading utilities
│   ├── data_validation/        # Great Expectations checks
│   │   └── validate_demand_data.py
│   └── training/               # Model training pipeline
├── 📂 tests/                   # Test suite
│   └── test_validation.py
├── 📂 .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── docker-compose.yml          # Docker services definition
├── Dockerfile.airflow          # Airflow container
├── Dockerfile.api              # FastAPI container
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/scalecast.git
cd scalecast
```

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
```

### 3. Add AWS Credentials

Add these to your `.env` file:

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name
```

### 4. Start Services

```bash
# Start all services (PostgreSQL, Airflow Webserver, Scheduler)
docker-compose up -d

# Check service status
docker-compose ps
```

### 5. Access Airflow UI

Open [http://localhost:8080](http://localhost:8080) in your browser.

| Field | Value |
|-------|-------|
| Username | `admin` |
| Password | `admin` |

### 6. Trigger the Pipeline

1. Navigate to **DAGs** in the Airflow UI
2. Find `scalecast_demand_pipeline`
3. Toggle the DAG to **On**
4. Click the **Play** button to trigger manually

---

## 📊 Pipeline Overview

The `scalecast_demand_pipeline` DAG executes four sequential tasks:

| Task | Description |
|------|-------------|
| **validate_data** | Validates raw CSV against schema and business rules. Implements circuit breaker — pipeline stops if validation fails. |
| **load_to_postgres** | Loads validated data into `warehouse.demand_data` table. Truncates existing data before insert. |
| **train_model** | Trains Random Forest regressor with feature engineering (day of week, month, weekend flag). Outputs MAE, RMSE, R² metrics. |
| **upload_model** | Uploads `demand_model.pkl` and `encoders.pkl` to S3 for model serving. |

```
validate_data  ──▶  load_to_postgres  ──▶  train_model  ──▶  upload_model
```

---

## 🔌 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-03-15",
    "store_id": "STORE_001",
    "product_id": "PROD_A",
    "price": 29.99,
    "promotion": true
  }'
```

**Response:**
```json
{
  "prediction": 142.5,
  "model_version": "1.0.0"
}
```

### API Documentation

Interactive Swagger docs available at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Running Tests

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests with verbose output
pytest tests/ -v
```

**Expected output:**
```
tests/test_validation.py::test_validation_passes_with_valid_data PASSED
tests/test_validation.py::test_validation_fails_with_missing_column PASSED
tests/test_validation.py::test_validation_fails_with_null_values PASSED
```

---

## 🧩 Core Engineering Challenges Addressed

| Challenge | Solution |
|-----------|----------|
| **Reproducibility** | DVC links trained models to exact training data versions, enabling rollback and audit |
| **Data Quality** | Great Expectations acts as a circuit breaker — bad data stops the pipeline before corrupting models |
| **Decoupling** | S3 serves as middleware between training and serving, allowing independent scaling |
| **Automation** | Airflow orchestrates the entire pipeline with scheduling, retries, and dependency management |

---

## 🔮 Future Improvements

- 📈 **MLflow Integration** — Add experiment tracking and model registry
- 🔬 **Model A/B Testing** — Implement traffic splitting for model comparison
- 📊 **Grafana Dashboards** — Add monitoring for pipeline metrics and model performance
- ☸️ **AWS ECS/EKS Deployment** — Migrate to managed container orchestration for production

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>ScaleCast</strong> — Built for scalable demand forecasting 📊
</p>
