# ScaleCast

**End-to-end MLOps pipeline for demand forecasting**

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Apache Airflow](https://img.shields.io/badge/Airflow-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)

---

## Architecture

<!-- TODO: Add architecture diagram -->
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ScaleCast Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │   S3     │───▶│ Airflow  │───▶│ Training │───▶│  MLflow  │         │
│   │  (Data)  │    │  (DAGs)  │    │ Pipeline │    │ (Track)  │         │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘         │
│        │                                               │                │
│        │              ┌──────────────┐                 │                │
│        └─────────────▶│  PostgreSQL  │◀────────────────┘                │
│                       │  (Warehouse) │                                  │
│                       └──────────────┘                                  │
│                              │                                          │
│                       ┌──────────────┐                                  │
│                       │   FastAPI    │                                  │
│                       │  (Serving)   │                                  │
│                       └──────────────┘                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestration | Apache Airflow 2.7 | Workflow scheduling and monitoring |
| ML Tracking | MLflow 2.9 | Experiment tracking and model registry |
| API | FastAPI 0.104 | Model serving and predictions |
| Database | PostgreSQL 15 | Data warehouse and metadata store |
| Data Validation | Great Expectations | Data quality checks |
| Version Control | DVC | Data and model versioning |
| Cloud Storage | AWS S3 | Artifact and data storage |
| Containerization | Docker Compose | Local development environment |

## Prerequisites

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **Python** >= 3.10 (for local development)
- **AWS Account** with S3 access

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/scalecast.git
cd scalecast
```

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your credentials
# Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.
```

### 3. Set Up AWS Resources

```bash
# Install dependencies (if running locally)
pip install boto3 python-dotenv

# Run AWS setup script
python scripts/setup_aws.py
```

### 4. Start Services

```bash
# Start all services (PostgreSQL, Airflow, MLflow)
docker-compose up -d

# Check service status
docker-compose ps
```

### 5. Access UIs

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5000 | - |
| API | http://localhost:8000 | - |
| API Docs | http://localhost:8000/docs | - |

## Project Structure

```
scalecast/
├── airflow/
│   ├── dags/              # Airflow DAG definitions
│   ├── logs/              # Airflow logs (gitignored)
│   └── plugins/           # Custom Airflow plugins
├── configs/
│   └── config.yaml        # Central configuration
├── data/
│   ├── raw/               # Raw input data (gitignored)
│   └── processed/         # Processed datasets (gitignored)
├── models/                # Trained model artifacts (gitignored)
├── scripts/
│   ├── setup_aws.py       # AWS S3 setup script
│   └── init_db.sql        # Database initialization
├── src/
│   ├── api/               # FastAPI application
│   ├── data_ingestion/    # Data loading utilities
│   ├── data_validation/   # Great Expectations checks
│   └── training/          # Model training pipeline
├── tests/                 # Test suite
├── docker-compose.yml     # Docker services definition
├── Dockerfile.api         # FastAPI container
├── requirements.txt       # Python dependencies
└── README.md
```

## Configuration

All configuration is centralized in `configs/config.yaml`:

```yaml
# Key configuration sections
data:
  raw_path: "data/raw"
  s3_bucket: "${S3_BUCKET_NAME}"

model:
  name: "demand_forecaster"
  features: [...]
  target: "demand"

api:
  host: "0.0.0.0"
  port: 8000
```

Environment variables can override config values using `${VAR_NAME}` syntax.

## Development

### Local Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Unix
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### Running Individual Services

```bash
# Start only PostgreSQL and MLflow
docker-compose up -d postgres mlflow

# Start API locally
uvicorn src.api.main:app --reload --port 8000
```

## Useful Commands

```bash
# View logs
docker-compose logs -f airflow-webserver
docker-compose logs -f mlflow

# Stop all services
docker-compose down

# Remove all data (volumes)
docker-compose down -v

# Rebuild containers
docker-compose build --no-cache

# Access PostgreSQL
docker exec -it scalecast-postgres psql -U scalecast -d scalecast
```

## Troubleshooting

### Airflow webserver not starting
```bash
# Check logs
docker-compose logs airflow-init
docker-compose logs airflow-webserver

# Reset Airflow
docker-compose down
docker volume rm scalecast-postgres-data
docker-compose up -d
```

### MLflow connection issues
```bash
# Verify AWS credentials
python scripts/setup_aws.py

# Check MLflow logs
docker-compose logs mlflow
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**ScaleCast** - Built for scalable demand forecasting
