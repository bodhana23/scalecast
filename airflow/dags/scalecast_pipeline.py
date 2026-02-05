"""
ScaleCast Demand Forecasting Pipeline DAG.

This DAG orchestrates the end-to-end demand forecasting pipeline:
1. Validate raw demand data
2. Load validated data to PostgreSQL
3. Train Random Forest model
4. Upload model artifacts to S3
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add src to path for imports
sys.path.insert(0, "/opt/airflow")

# =============================================================================
# Configuration
# =============================================================================
DATA_PATH = "/opt/airflow/data/raw/demand_data.csv"
MODELS_DIR = Path("/opt/airflow/models")
DB_HOST = "postgres"  # Docker network hostname

# =============================================================================
# Task Functions
# =============================================================================


def validate_data(**kwargs):
    """Validate raw demand data against expected schema and rules."""
    print(f"Validating data from: {DATA_PATH}")

    from src.data_validation.validate_demand_data import validate_demand_data

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    result = validate_demand_data(DATA_PATH)

    print(f"Rows validated: {result.row_count:,}")
    print(f"Passed checks: {len(result.passed_expectations)}")
    print(f"Failed checks: {len(result.failed_expectations)}")

    if not result.success:
        for failure in result.failed_expectations:
            print(f"  FAILED: {failure}")
        raise ValueError("Data validation failed")

    print("Data validation PASSED")
    return result.row_count


def load_to_postgres(**kwargs):
    """Load validated demand data to PostgreSQL warehouse."""
    print(f"Loading data from: {DATA_PATH}")

    import pandas as pd
    from dotenv import load_dotenv
    from sqlalchemy import create_engine, text

    load_dotenv("/opt/airflow/.env")

    # Build connection string with Docker network hostname
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_name = os.getenv("POSTGRES_DB")
    db_port = os.getenv("POSTGRES_PORT", "5432")

    connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{DB_HOST}:{db_port}/{db_name}"

    # Load CSV
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    print(f"Loaded {len(df):,} rows from CSV")

    # Connect and load
    engine = create_engine(connection_string)

    # Truncate existing data using begin() for automatic commit
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE warehouse.demand_data"))

    # Load new data directly with engine
    df.to_sql(
        name="demand_data",
        con=engine,
        schema="warehouse",
        if_exists="append",
        index=False,
    )

    engine.dispose()
    print(f"Loaded {len(df):,} rows to warehouse.demand_data")
    return len(df)


def train_model(**kwargs):
    """Train demand forecasting model using data from PostgreSQL."""
    print("Starting model training...")

    import joblib
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sqlalchemy import create_engine

    load_dotenv("/opt/airflow/.env")

    # Database connection
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_name = os.getenv("POSTGRES_DB")
    db_port = os.getenv("POSTGRES_PORT", "5432")

    connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{DB_HOST}:{db_port}/{db_name}"

    # Load data
    engine = create_engine(connection_string)
    df = pd.read_sql("SELECT * FROM warehouse.demand_data", engine)
    engine.dispose()

    print(f"Loaded {len(df):,} rows from PostgreSQL")

    # Feature engineering
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    store_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    df["store_id_encoded"] = store_encoder.fit_transform(df["store_id"])
    df["product_id_encoded"] = product_encoder.fit_transform(df["product_id"])
    df["promotion"] = df["promotion"].astype(int)

    # Prepare features
    feature_names = [
        "day_of_week", "month", "is_weekend",
        "store_id_encoded", "product_id_encoded", "price", "promotion"
    ]
    X = df[feature_names].values
    y = df["quantity_sold"].values

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    # Save model and encoders
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "demand_model.pkl")
    joblib.dump({"store_encoder": store_encoder, "product_encoder": product_encoder}, MODELS_DIR / "encoders.pkl")

    print(f"Model saved to {MODELS_DIR}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def upload_model(**kwargs):
    """Upload trained model files to S3."""
    print("Uploading model to S3...")

    import boto3
    from botocore.exceptions import ClientError

    bucket = os.getenv("S3_BUCKET_NAME", "scalecast-mlops-bodhana-2026")

    files = [
        (MODELS_DIR / "demand_model.pkl", "models/demand_model.pkl"),
        (MODELS_DIR / "encoders.pkl", "models/encoders.pkl"),
    ]

    s3 = boto3.client("s3")
    uploaded = 0

    for local_path, s3_key in files:
        if not local_path.exists():
            print(f"Warning: {local_path} not found, skipping")
            continue

        local_size = local_path.stat().st_size
        print(f"Uploading {local_path.name} ({local_size / 1024:.2f} KB)...")

        s3.upload_file(str(local_path), bucket, s3_key)

        # Verify
        response = s3.head_object(Bucket=bucket, Key=s3_key)
        if response["ContentLength"] == local_size:
            print(f"  Uploaded to s3://{bucket}/{s3_key}")
            uploaded += 1
        else:
            print(f"  Warning: Size mismatch for {s3_key}")

    print(f"Upload complete: {uploaded}/{len(files)} files")
    return uploaded


# =============================================================================
# DAG Definition
# =============================================================================
with DAG(
    dag_id="scalecast_demand_pipeline",
    description="End-to-end demand forecasting pipeline",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "demand-forecasting"],
) as dag:

    # Task 1: Validate raw data
    task_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    # Task 2: Load to PostgreSQL
    task_load = PythonOperator(
        task_id="load_to_postgres",
        python_callable=load_to_postgres,
    )

    # Task 3: Train model
    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    # Task 4: Upload to S3
    task_upload = PythonOperator(
        task_id="upload_model",
        python_callable=upload_model,
    )

    # =============================================================================
    # Task Dependencies
    # =============================================================================
    # validate_data >> load_to_postgres >> train_model >> upload_model
    task_validate >> task_load >> task_train >> task_upload
