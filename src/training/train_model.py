#!/usr/bin/env python3
"""Train demand forecasting model using data from PostgreSQL."""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine

# Load environment variables
load_dotenv()

# Database configuration
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")
DB_HOST = "localhost"
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# Model paths
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "demand_model.pkl"
ENCODERS_PATH = MODEL_DIR / "encoders.pkl"

# Feature configuration
FEATURE_NAMES = [
    "day_of_week",
    "month",
    "is_weekend",
    "store_id_encoded",
    "product_id_encoded",
    "price",
    "promotion",
]


def get_connection_string() -> str:
    """Build PostgreSQL connection string."""
    return f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def load_data_from_postgres() -> pd.DataFrame:
    """Load demand data from PostgreSQL."""
    engine = create_engine(get_connection_string())
    query = "SELECT * FROM warehouse.demand_data"
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df


def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create features for model training."""
    df = df.copy()

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Date-based features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Encode categorical features
    store_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    df["store_id_encoded"] = store_encoder.fit_transform(df["store_id"])
    df["product_id_encoded"] = product_encoder.fit_transform(df["product_id"])

    # Ensure promotion is int
    df["promotion"] = df["promotion"].astype(int)

    encoders = {
        "store_encoder": store_encoder,
        "product_encoder": product_encoder,
    }

    return df, encoders


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """Train RandomForest model."""
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestRegressor, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }
    return metrics


def print_report(train_size: int, test_size: int, metrics: dict) -> None:
    """Print training report."""
    print("=" * 60)
    print("Demand Forecasting Model Training Report")
    print("=" * 60)
    print(f"Dataset size: {train_size + test_size:,} total")
    print(f"  Train: {train_size:,} ({train_size / (train_size + test_size) * 100:.0f}%)")
    print(f"  Test:  {test_size:,} ({test_size / (train_size + test_size) * 100:.0f}%)")
    print("-" * 60)
    print("Features:")
    for feature in FEATURE_NAMES:
        print(f"  - {feature}")
    print("-" * 60)
    print("Model Metrics:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print("-" * 60)
    print(f"Model saved: {MODEL_PATH}")
    print(f"Encoders saved: {ENCODERS_PATH}")
    print("=" * 60)


def main():
    try:
        print("Loading data from PostgreSQL...")
        df = load_data_from_postgres()

        if df.empty:
            print("Error: No data found in warehouse.demand_data")
            sys.exit(1)

        print(f"Loaded {len(df):,} rows")

        print("Creating features...")
        df, encoders = create_features(df)

        # Prepare features and target
        X = df[FEATURE_NAMES].values
        y = df["quantity_sold"].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Training model...")
        model = train_model(X_train, y_train)

        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)

        # Save model and encoders
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(encoders, ENCODERS_PATH)

        # Print report
        print_report(len(X_train), len(X_test), metrics)

        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
