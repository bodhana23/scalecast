#!/usr/bin/env python3
"""Train and compare demand forecasting models using data from PostgreSQL.

This script compares multiple model families to find the best demand predictor:
  - Baselines (naive lag, rolling mean) to establish a performance floor
  - Linear Regression as a simple parametric baseline
  - Random Forest and XGBoost as strong tree-based ensembles
  - A PyTorch feedforward neural network for non-linear pattern capture

Trade-offs:
  - Tree-based models handle tabular data well and require minimal preprocessing,
    but can overfit on small datasets and lack extrapolation ability.
  - Neural networks can capture complex interactions but need more data, careful
    tuning, and feature scaling. They also lack built-in feature importance.
  - Baselines are critical: if a learned model cannot beat lag_1 or rolling_mean_7,
    the added complexity is not justified.
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import create_engine
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

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
PYTORCH_MODEL_PATH = MODEL_DIR / "demand_model_pytorch.pt"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODERS_PATH = MODEL_DIR / "encoders.pkl"

# Feature configuration
FEATURE_NAMES = [
    # Seasonality
    "day_of_week",
    "month",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "week_of_year",
    # Categorical (encoded)
    "store_id_encoded",
    "product_id_encoded",
    # Numeric
    "price",
    "promotion",
    # Lag features
    "lag_1",
    "lag_7",
    "lag_14",
    # Rolling statistics
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
]

# Indices of lag_1 and rolling_mean_7 within FEATURE_NAMES for baseline access
LAG_1_IDX = FEATURE_NAMES.index("lag_1")
ROLLING_MEAN_7_IDX = FEATURE_NAMES.index("rolling_mean_7")


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
    initial_cols = len(df.columns)

    # Convert date to datetime and sort for correct lag/rolling calculations
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)

    # --- Seasonality features ---
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # --- Lag features (grouped by store + product) ---
    grouped = df.groupby(["store_id", "product_id"])["quantity_sold"]
    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["lag_14"] = grouped.shift(14)

    # --- Rolling statistics (grouped by store + product) ---
    df["rolling_mean_7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df["rolling_std_7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).std()
    )
    df["rolling_mean_14"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).mean()
    )

    # Drop rows with NaN from lag features (first rows per group)
    rows_before = len(df)
    df = df.dropna(subset=["lag_1", "lag_7", "lag_14"]).reset_index(drop=True)
    rows_after = len(df)

    print(f"  Features before engineering: {initial_cols}")
    print(f"  Features after engineering:  {len(df.columns)}")
    print(f"  Rows dropped (NaN from lags): {rows_before - rows_after:,}")
    print(f"  Rows remaining: {rows_after:,}")
    print(f"\n  Sample of new features (first 5 rows):")
    new_feature_cols = [
        "lag_1", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_std_7", "rolling_mean_14",
        "is_month_start", "is_month_end", "week_of_year",
    ]
    print(df[new_feature_cols].head().to_string(index=False))

    # --- Encode categorical features ---
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


# ---------------------------------------------------------------------------
# PyTorch model
# ---------------------------------------------------------------------------

class DemandPredictor(nn.Module):
    """Feedforward neural network for demand forecasting."""

    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_pytorch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
) -> tuple:
    """Train a PyTorch feedforward network for demand forecasting.

    Returns:
        (model, scaler, y_pred) tuple where y_pred are predictions on X_test.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test_scaled)

    # DataLoader
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = DemandPredictor(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(dataset)
        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

    # Predict on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).squeeze().numpy()

    return model, scaler, y_pred


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate regression metrics including MAPE.

    Returns dict with keys: mae, rmse, mape, r2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE with epsilon to avoid division by zero
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def train_and_compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train multiple models and compare their performance.

    Trains baselines, Linear Regression, Random Forest (GridSearchCV),
    XGBoost (GridSearchCV), and a PyTorch neural network, then prints a
    comparison table and feature importance from the best tree model.

    Returns:
        Dictionary mapping model names to {model, metrics, y_pred} dicts.
    """
    results = {}

    # ------------------------------------------------------------------
    # 1. Naive Baseline (predict lag_1)
    # ------------------------------------------------------------------
    print("\n  [1/6] Evaluating Naive Baseline (lag_1)...")
    naive_pred = X_test[:, LAG_1_IDX]
    results["Naive Baseline"] = {
        "model": None,
        "metrics": calculate_metrics(y_test, naive_pred),
        "y_pred": naive_pred,
    }

    # ------------------------------------------------------------------
    # 2. Mean Baseline (predict rolling_mean_7)
    # ------------------------------------------------------------------
    print("  [2/6] Evaluating Mean Baseline (rolling_mean_7)...")
    mean_pred = X_test[:, ROLLING_MEAN_7_IDX]
    results["Mean Baseline"] = {
        "model": None,
        "metrics": calculate_metrics(y_test, mean_pred),
        "y_pred": mean_pred,
    }

    # ------------------------------------------------------------------
    # 3. Linear Regression
    # ------------------------------------------------------------------
    print("  [3/6] Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    results["Linear Regression"] = {
        "model": lr_model,
        "metrics": calculate_metrics(y_test, lr_pred),
        "y_pred": lr_pred,
    }

    # ------------------------------------------------------------------
    # 4. Random Forest with GridSearchCV
    # ------------------------------------------------------------------
    print("  [4/6] Training Random Forest (GridSearchCV)...")
    rf_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, None],
    }
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    rf_model = rf_grid.best_estimator_
    rf_pred = rf_model.predict(X_test)
    print(f"    Best params: {rf_grid.best_params_}")
    results["Random Forest"] = {
        "model": rf_model,
        "metrics": calculate_metrics(y_test, rf_pred),
        "y_pred": rf_pred,
    }

    # ------------------------------------------------------------------
    # 5. XGBoost with GridSearchCV
    # ------------------------------------------------------------------
    print("  [5/6] Training XGBoost (GridSearchCV)...")
    xgb_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
    }
    xgb_grid = GridSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1),
        xgb_param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    xgb_grid.fit(X_train, y_train)
    xgb_model = xgb_grid.best_estimator_
    xgb_pred = xgb_model.predict(X_test)
    print(f"    Best params: {xgb_grid.best_params_}")
    results["XGBoost"] = {
        "model": xgb_model,
        "metrics": calculate_metrics(y_test, xgb_pred),
        "y_pred": xgb_pred,
    }

    # ------------------------------------------------------------------
    # 6. PyTorch Neural Network
    # ------------------------------------------------------------------
    print("  [6/6] Training PyTorch Neural Network...")
    pt_model, pt_scaler, pt_pred = train_pytorch_model(
        X_train, y_train, X_test, y_test
    )
    results["PyTorch Neural Network"] = {
        "model": pt_model,
        "metrics": calculate_metrics(y_test, pt_pred),
        "y_pred": pt_pred,
        "scaler": pt_scaler,
    }

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print_comparison_table(results)

    # ------------------------------------------------------------------
    # Feature importance from best tree-based model
    # ------------------------------------------------------------------
    print_feature_importance(results)

    return results


def print_comparison_table(results: dict) -> None:
    """Print a formatted comparison table of all model metrics."""
    print("\n" + "=" * 60)
    print("              MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>10} {'MAPE':>9} {'R²':>8}")
    print("-" * 60)

    for name, data in results.items():
        m = data["metrics"]
        print(
            f"{name:<25} {m['mae']:>8.2f} {m['rmse']:>10.2f} "
            f"{m['mape']:>8.2f}% {m['r2']:>8.4f}"
        )

    # Identify best model by MAE
    best_name = min(results, key=lambda k: results[k]["metrics"]["mae"])
    best_mae = results[best_name]["metrics"]["mae"]
    naive_mae = results["Naive Baseline"]["metrics"]["mae"]
    improvement = (1 - best_mae / naive_mae) * 100 if naive_mae > 0 else 0.0

    print("-" * 60)
    print(f"Best Model: {best_name}")
    print(f"Improvement over Naive Baseline: {improvement:.1f}%")
    print("=" * 60)


def print_feature_importance(results: dict) -> None:
    """Print top 10 feature importances from the best tree-based model."""
    # Pick best tree model by MAE
    tree_models = {
        k: v for k, v in results.items() if k in ("Random Forest", "XGBoost")
    }
    if not tree_models:
        return

    best_tree_name = min(tree_models, key=lambda k: tree_models[k]["metrics"]["mae"])
    best_tree = tree_models[best_tree_name]["model"]
    importances = best_tree.feature_importances_

    indices = np.argsort(importances)[::-1][:10]
    print(f"\nTop 10 Most Important Features ({best_tree_name}):")
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:>2}. {FEATURE_NAMES[idx]:<25} {importances[idx]:.4f}")


def save_best_model(results: dict, encoders: dict) -> None:
    """Save the best model (by MAE) and encoders to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_name = min(results, key=lambda k: results[k]["metrics"]["mae"])
    best_data = results[best_name]
    best_mae = best_data["metrics"]["mae"]

    print(f"\nSaving best model: {best_name} (MAE={best_mae:.4f})")

    if best_name == "PyTorch Neural Network":
        torch.save(best_data["model"].state_dict(), PYTORCH_MODEL_PATH)
        joblib.dump(best_data["scaler"], SCALER_PATH)
        print(f"  Model saved: {PYTORCH_MODEL_PATH}")
        print(f"  Scaler saved: {SCALER_PATH}")
    else:
        joblib.dump(best_data["model"], MODEL_PATH)
        print(f"  Model saved: {MODEL_PATH}")

    joblib.dump(encoders, ENCODERS_PATH)
    print(f"  Encoders saved: {ENCODERS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full training and comparison pipeline."""
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

        print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")
        print(f"Features: {len(FEATURE_NAMES)}")

        print("\nTraining and comparing models...")
        results = train_and_compare_models(X_train, y_train, X_test, y_test)

        # Save best model and encoders
        save_best_model(results, encoders)

        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
