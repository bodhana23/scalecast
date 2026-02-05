#!/usr/bin/env python3
"""Load validated demand data into PostgreSQL warehouse."""

import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables from .env
load_dotenv()

import os

# Database configuration
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")
DB_HOST = "localhost"  # Override for local access
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# Table configuration
SCHEMA = "warehouse"
TABLE = "demand_data"


def get_connection_string() -> str:
    """Build PostgreSQL connection string."""
    return f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def load_csv(file_path: str) -> pd.DataFrame:
    """Load and prepare CSV data for database insertion."""
    df = pd.read_csv(file_path)

    # Ensure date column is properly formatted
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def truncate_table(engine) -> None:
    """Truncate existing data from the target table."""
    with engine.connect() as conn:
        conn.execute(text(f"TRUNCATE TABLE {SCHEMA}.{TABLE}"))
        conn.commit()


def load_to_database(df: pd.DataFrame, engine) -> int:
    """Load DataFrame into PostgreSQL table."""
    rows_loaded = df.to_sql(
        name=TABLE,
        con=engine,
        schema=SCHEMA,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )
    return len(df)


def print_summary(df: pd.DataFrame, rows_loaded: int, elapsed_time: float) -> None:
    """Print load summary."""
    print("=" * 60)
    print("PostgreSQL Load Summary")
    print("=" * 60)
    print(f"Target: {SCHEMA}.{TABLE}")
    print(f"Rows loaded: {rows_loaded:,}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print("-" * 60)
    print("Sample data (first 3 rows):")
    print(df.head(3).to_string(index=False))
    print("=" * 60)


def main():
    if len(sys.argv) != 2:
        print("Usage: python load_to_postgres.py <csv_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    try:
        start_time = time.time()

        # Load CSV data
        print(f"Loading CSV: {file_path}")
        df = load_csv(file_path)

        # Create database connection
        connection_string = get_connection_string()
        engine = create_engine(connection_string)

        # Truncate existing data
        print(f"Truncating {SCHEMA}.{TABLE}...")
        truncate_table(engine)

        # Load data
        print("Loading data to PostgreSQL...")
        rows_loaded = load_to_database(df, engine)

        elapsed_time = time.time() - start_time

        # Print summary
        print_summary(df, rows_loaded, elapsed_time)

        engine.dispose()
        sys.exit(0)

    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
