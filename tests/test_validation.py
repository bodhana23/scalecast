"""Tests for demand data validation logic."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_validation.validate_demand_data import validate_demand_data


@pytest.fixture
def valid_data():
    """Create a valid DataFrame with all required columns."""
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "store_id": [1, 2, 3],
        "product_id": [101, 102, 103],
        "quantity_sold": [10, 20, 30],
        "price": [9.99, 19.99, 29.99],
        "promotion": [0, 1, 0],
    })


@pytest.fixture
def data_missing_column():
    """Create a DataFrame missing the 'store_id' column."""
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "product_id": [101, 102],
        "quantity_sold": [10, 20],
        "price": [9.99, 19.99],
        "promotion": [0, 1],
    })


@pytest.fixture
def data_with_nulls():
    """Create a DataFrame with null values in a required column."""
    return pd.DataFrame({
        "date": ["2024-01-01", None, "2024-01-03"],
        "store_id": [1, 2, 3],
        "product_id": [101, 102, 103],
        "quantity_sold": [10, 20, 30],
        "price": [9.99, 19.99, 29.99],
        "promotion": [0, 1, 0],
    })


def test_validation_passes_with_valid_data(valid_data):
    """Verify validation passes when all required columns are present and valid."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        valid_data.to_csv(f.name, index=False)
        result = validate_demand_data(f.name)
        Path(f.name).unlink()

    assert result.success is True
    assert len(result.failed_expectations) == 0


def test_validation_fails_with_missing_column(data_missing_column):
    """Verify validation fails when a required column is missing."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        data_missing_column.to_csv(f.name, index=False)
        result = validate_demand_data(f.name)
        Path(f.name).unlink()

    assert result.success is False
    assert any("store_id" in msg for msg in result.failed_expectations)


def test_validation_fails_with_null_values(data_with_nulls):
    """Verify validation fails when required columns contain null values."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        data_with_nulls.to_csv(f.name, index=False)
        result = validate_demand_data(f.name)
        Path(f.name).unlink()

    assert result.success is False
    assert any("null" in msg.lower() for msg in result.failed_expectations)
