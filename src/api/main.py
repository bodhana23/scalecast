"""
ScaleCast FastAPI Application.

This module provides the main FastAPI application for serving demand
forecasting predictions via REST endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ScaleCast API",
    description="Demand Forecasting API for ScaleCast MLOps Pipeline",
    version="1.0.0",
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
    return {"status": "healthy", "service": "scalecast-api"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ScaleCast API",
        "version": "1.0.0",
        "description": "Demand Forecasting API",
        "docs_url": "/docs",
    }
