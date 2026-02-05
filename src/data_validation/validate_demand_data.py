#!/usr/bin/env python3
"""Validate demand forecasting data against expected schema and rules."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class ValidationResult:
    """Container for validation results."""

    success: bool = True
    failed_expectations: list = field(default_factory=list)
    passed_expectations: list = field(default_factory=list)
    row_count: int = 0
    columns_validated: int = 0


def validate_column_exists(df: pd.DataFrame, column: str, result: ValidationResult) -> bool:
    """Check if column exists in DataFrame."""
    if column in df.columns:
        return True
    result.success = False
    result.failed_expectations.append(f"Column '{column}' does not exist")
    return False


def validate_no_nulls(df: pd.DataFrame, column: str, result: ValidationResult) -> bool:
    """Check if column has no null values."""
    null_count = df[column].isna().sum()
    if null_count == 0:
        return True
    result.success = False
    result.failed_expectations.append(f"Column '{column}' has {null_count} null values")
    return False


def validate_positive(df: pd.DataFrame, column: str, result: ValidationResult) -> bool:
    """Check if all values in column are positive (> 0)."""
    non_positive = (df[column] <= 0).sum()
    if non_positive == 0:
        return True
    result.success = False
    result.failed_expectations.append(f"Column '{column}' has {non_positive} non-positive values")
    return False


def validate_demand_data(file_path: str) -> ValidationResult:
    """Validate demand data CSV file against expected schema and rules."""
    result = ValidationResult()

    # Load data
    df = pd.read_csv(file_path)
    result.row_count = len(df)

    # Define validation rules: (column, checks)
    rules = [
        ("date", ["exists", "no_nulls"]),
        ("store_id", ["exists", "no_nulls"]),
        ("product_id", ["exists", "no_nulls"]),
        ("quantity_sold", ["exists", "no_nulls", "positive"]),
        ("price", ["exists", "no_nulls", "positive"]),
        ("promotion", ["exists"]),
    ]

    # Run validations
    for column, checks in rules:
        result.columns_validated += 1

        # Check column exists first
        if not validate_column_exists(df, column, result):
            continue

        for check in checks:
            if check == "exists":
                result.passed_expectations.append(f"Column '{column}' exists")
            elif check == "no_nulls":
                if validate_no_nulls(df, column, result):
                    result.passed_expectations.append(f"Column '{column}' has no nulls")
            elif check == "positive":
                if validate_positive(df, column, result):
                    result.passed_expectations.append(f"Column '{column}' all values > 0")

    return result


def print_report(result: ValidationResult, file_path: str) -> None:
    """Print a formatted validation report."""
    print("=" * 60)
    print("Demand Data Validation Report")
    print("=" * 60)
    print(f"File: {file_path}")
    print(f"Rows: {result.row_count:,}")
    print(f"Columns validated: {result.columns_validated}")
    print("-" * 60)

    # Print passed checks
    for expectation in result.passed_expectations:
        print(f"  [PASSED] {expectation}")

    # Print failed checks
    for expectation in result.failed_expectations:
        print(f"  [FAILED] {expectation}")

    # Summary
    print("-" * 60)
    total = len(result.passed_expectations) + len(result.failed_expectations)
    passed = len(result.passed_expectations)

    if result.success:
        print(f"Result: PASSED ({passed}/{total} checks)")
    else:
        print(f"Result: FAILED ({passed}/{total} checks passed)")
    print("=" * 60)


def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_demand_data.py <csv_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    try:
        result = validate_demand_data(file_path)
        print_report(result, file_path)
        sys.exit(0 if result.success else 1)
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
