#!/usr/bin/env python3
"""
Initialize PostgreSQL database schema for ScaleCast.

This script:
1. Connects to PostgreSQL using credentials from .env
2. Creates the 'warehouse' schema for clean data
3. Creates the 'demand_data' table for storing sales data
"""

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


def main():
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    # Get database connection settings
    db_config = {
        "host": "localhost",
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }

    # Validate required environment variables
    missing = [k for k, v in db_config.items() if v is None and k != "port"]
    if missing:
        print(f"Error: Missing environment variables: {missing}")
        print("Make sure .env file exists with POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD")
        sys.exit(1)

    conn = None
    cursor = None

    try:
        # Connect to PostgreSQL
        print(f"Connecting to PostgreSQL at {db_config['host']}:{db_config['port']}...")
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cursor = conn.cursor()

        print("Connected successfully!\n")

        # Create warehouse schema
        print("Creating schema 'warehouse'...")
        cursor.execute("CREATE SCHEMA IF NOT EXISTS warehouse;")
        print("Schema 'warehouse' created.\n")

        # Create demand_data table
        print("Creating table 'warehouse.demand_data'...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS warehouse.demand_data (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                store_id VARCHAR(50) NOT NULL,
                product_id VARCHAR(50) NOT NULL,
                quantity_sold INTEGER NOT NULL,
                price DECIMAL(10,2),
                promotion BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("Table 'warehouse.demand_data' created.\n")

        # Get row count
        cursor.execute("SELECT COUNT(*) FROM warehouse.demand_data;")
        row_count = cursor.fetchone()[0]

        # Print summary
        print("=" * 50)
        print("Database Initialization Complete!")
        print("=" * 50)
        print(f"Database:     {db_config['database']}")
        print(f"Schema:       warehouse")
        print(f"Table:        warehouse.demand_data")
        print(f"Row count:    {row_count}")
        print("=" * 50)

    except psycopg2.OperationalError as e:
        print(f"Error: Could not connect to PostgreSQL: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL container is running (docker-compose up -d)")
        print("  2. Port 5432 is accessible")
        print("  3. Credentials in .env are correct")
        sys.exit(1)

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("\nConnection closed.")


if __name__ == "__main__":
    main()
