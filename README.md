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

## ML Model Details

### Feature Engineering

We engineered 16 features for demand forecasting:

| Category | Features | Description |
|----------|----------|-------------|
| Temporal | day_of_week, month, week_of_year | Capture seasonal patterns |
| Boolean | is_weekend, is_month_start, is_month_end | Special day indicators |
| Lag | lag_1, lag_7, lag_14 | Historical sales (1 day, 1 week, 2 weeks ago) |
| Rolling | rolling_mean_7, rolling_std_7, rolling_mean_14 | Trend and volatility |
| Numeric | price, promotion | Business factors |
| Categorical | store_id_encoded, product_id_encoded | Location and product identity |

### Model Comparison

We compared 6 models including traditional ML and deep learning:

| Model | Type | MAE | RMSE | MAPE | R² | Notes |
|-------|------|-----|------|------|-----|-------|
| Naive Baseline | Baseline | 115.06 | 322.69 | 382.9% | -0.975 | Uses yesterday's sales |
| Mean Baseline | Baseline | 97.96 | 237.91 | 405.4% | -0.074 | Uses 7-day rolling mean |
| Linear Regression | Traditional ML | 92.47 | 227.20 | 436.7% | 0.021 | Simple linear model |
| Random Forest | Traditional ML | 81.15 | 226.00 | 332.6% | 0.031 | Tuned with GridSearchCV |
| XGBoost | Gradient Boosting | **79.38** | 244.07 | **270.6%** | -0.130 | **Best model** |
| PyTorch NN | Deep Learning | 90.88 | 226.62 | 424.3% | 0.026 | Custom training loop |

### Why XGBoost Won

1. **Tabular data favors tree-based models** — Neural networks excel at unstructured data (images, text), but XGBoost handles tabular features better
2. **Feature interactions** — XGBoost automatically captures non-linear relationships between features
3. **31% improvement over baseline** — Significant reduction in prediction error

### Top Predictive Features

| Rank | Feature | Importance | Why It Matters |
|------|---------|------------|----------------|
| 1 | price | 0.195 | Price elasticity affects demand |
| 2 | rolling_std_7 | 0.132 | Sales volatility indicates predictability |
| 3 | month | 0.096 | Strong seasonal patterns |
| 4 | lag_7 | 0.090 | Weekly purchase cycles |
| 5 | lag_14 | 0.082 | Bi-weekly patterns |

### PyTorch Implementation

We included a PyTorch neural network to demonstrate deep learning skills:

- **Architecture**: Input(16) → Dense(64) → ReLU → Dropout(0.2) → Dense(32) → ReLU → Dropout(0.2) → Output(1)
- **Training**: 100 epochs, Adam optimizer, MSE loss, batch size 32
- **Result**: 21% improvement over baseline (competitive but XGBoost was better for this dataset)

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
| **ML Framework** | scikit-learn, XGBoost, PyTorch | Model comparison pipeline (6 models) |
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
| **train_model** | Compares 6 models (baselines, Linear Regression, Random Forest, XGBoost, PyTorch NN) with 16 engineered features. Selects best model by MAE. |
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

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| XGBoost over Neural Network | Tabular data with engineered features favors gradient boosting |
| Lag features (1, 7, 14 days) | Captures daily, weekly, and bi-weekly purchase patterns |
| GridSearchCV with cv=3 | Balance between thorough tuning and computation time |
| StandardScaler for PyTorch only | Tree-based models don't require feature scaling |

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
