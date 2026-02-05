"""
ScaleCast FastAPI Application.

This module provides the main FastAPI application for serving demand
forecasting predictions via REST endpoints.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Model paths
MODEL_PATH = Path("models/demand_model.pkl")
ENCODERS_PATH = Path("models/encoders.pkl")

# Global model state
model = None
encoders = None
model_loaded = False


class PredictionRequest(BaseModel):
    """Request schema for demand prediction."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    store_id: str = Field(..., description="Store identifier")
    product_id: str = Field(..., description="Product identifier")
    price: float = Field(..., gt=0, description="Product price")
    promotion: bool = Field(..., description="Whether promotion is active")


class PredictionResponse(BaseModel):
    """Response schema for demand prediction."""

    prediction: float = Field(..., description="Predicted quantity sold")
    model_version: str = Field(..., description="Model version identifier")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, encoders, model_loaded

    try:
        if MODEL_PATH.exists() and ENCODERS_PATH.exists():
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODERS_PATH)
            model_loaded = True
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print("Warning: Model files not found, prediction endpoint will be unavailable")
    except Exception as e:
        print(f"Error loading model: {e}")

    yield


app = FastAPI(
    title="ScaleCast API",
    description="Demand Forecasting API for ScaleCast MLOps Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "model_loaded": model_loaded}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ScaleCast API",
        "version": "1.0.0",
        "description": "Demand Forecasting API",
        "docs_url": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate demand prediction for given inputs."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Parse date
        date = datetime.strptime(request.date, "%Y-%m-%d")
        day_of_week = date.weekday()
        month = date.month
        is_weekend = 1 if day_of_week in [5, 6] else 0

        # Encode store_id
        store_encoder = encoders["store_encoder"]
        if request.store_id in store_encoder.classes_:
            store_encoded = store_encoder.transform([request.store_id])[0]
        else:
            store_encoded = -1

        # Encode product_id
        product_encoder = encoders["product_encoder"]
        if request.product_id in product_encoder.classes_:
            product_encoded = product_encoder.transform([request.product_id])[0]
        else:
            product_encoded = -1

        # Build feature array
        features = np.array([[
            day_of_week,
            month,
            is_weekend,
            store_encoded,
            product_encoded,
            request.price,
            int(request.promotion),
        ]])

        # Predict
        prediction = model.predict(features)[0]

        return PredictionResponse(
            prediction=round(float(prediction), 2),
            model_version="1.0.0",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
