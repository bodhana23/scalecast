#!/usr/bin/env python3
"""
Download and prepare the UCI Online Retail dataset for demand forecasting.

This script downloads the Online Retail dataset, cleans it, transforms it
to our schema, and saves it as a CSV file for further processing.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests

# Dataset URL
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

# Output path
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "demand_data.csv"

# Top countries to keep (others grouped as 'OTHER')
TOP_COUNTRIES = 5

# Number of top products to keep (by total sales volume)
TOP_PRODUCTS = 20


def download_dataset(url: str) -> bytes:
    """Download the dataset from UCI repository."""
    print(f"Downloading dataset from:\n{url}\n")
    print("This may take a minute...")

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        print(f"Downloaded {len(response.content) / 1024 / 1024:.2f} MB\n")
        return response.content
    except requests.exceptions.Timeout:
        print("Error: Download timed out. Please try again.")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)


def load_excel_data(content: bytes) -> pd.DataFrame:
    """Load Excel data from bytes content."""
    print("Loading Excel file...")
    from io import BytesIO

    df = pd.read_excel(BytesIO(content), engine="openpyxl")
    print(f"Loaded {len(df):,} rows\n")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by removing invalid rows."""
    print("Cleaning data...")
    original_count = len(df)

    # Remove rows where Quantity <= 0 (returns/cancellations)
    df = df[df["Quantity"] > 0]
    print(f"  After removing Quantity <= 0: {len(df):,} rows")

    # Remove rows where UnitPrice <= 0
    df = df[df["UnitPrice"] > 0]
    print(f"  After removing UnitPrice <= 0: {len(df):,} rows")

    # Remove rows with missing CustomerID
    df = df[df["CustomerID"].notna()]
    print(f"  After removing missing CustomerID: {len(df):,} rows")

    # Remove rows with missing Description
    df = df[df["Description"].notna()]
    print(f"  After removing missing Description: {len(df):,} rows")

    cleaned_count = len(df)
    print(f"\nRemoved {original_count - cleaned_count:,} invalid rows")
    print(f"Cleaned dataset: {cleaned_count:,} rows\n")

    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform data to our schema."""
    print("Transforming data to target schema...")

    # Convert InvoiceDate to date only
    df["date"] = pd.to_datetime(df["InvoiceDate"]).dt.date

    # Identify top countries by transaction count
    country_counts = df["Country"].value_counts()
    top_countries = country_counts.head(TOP_COUNTRIES).index.tolist()
    print(f"  Top {TOP_COUNTRIES} countries: {top_countries}")

    # Map countries: keep top countries, group others as 'OTHER'
    df["store_id"] = df["Country"].apply(
        lambda x: x if x in top_countries else "OTHER"
    )

    # Map other columns
    df["product_id"] = df["StockCode"].astype(str)
    df["quantity_sold"] = df["Quantity"]
    df["price"] = df["UnitPrice"]
    df["promotion"] = False

    # Select only the columns we need
    result = df[["date", "store_id", "product_id", "quantity_sold", "price", "promotion"]]

    print(f"  Transformed {len(result):,} rows\n")
    return result


def aggregate_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales by date, store, and product."""
    print("Aggregating daily sales...")

    aggregated = df.groupby(["date", "store_id", "product_id"]).agg(
        quantity_sold=("quantity_sold", "sum"),
        price=("price", "mean"),
        promotion=("promotion", "first"),
    ).reset_index()

    # Round price to 2 decimal places
    aggregated["price"] = aggregated["price"].round(2)

    print(f"  Aggregated to {len(aggregated):,} rows\n")
    return aggregated


def sample_top_products(df: pd.DataFrame, n_products: int) -> pd.DataFrame:
    """Keep only top N products by total sales volume."""
    print(f"Sampling top {n_products} products by sales volume...")

    # Calculate total quantity sold per product
    product_sales = df.groupby("product_id")["quantity_sold"].sum()
    top_products = product_sales.nlargest(n_products).index.tolist()

    # Filter to keep only top products
    filtered = df[df["product_id"].isin(top_products)]

    print(f"  Filtered to {len(filtered):,} rows\n")
    return filtered


def save_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to CSV file."""
    print(f"Saving to {output_path}...")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    file_size = output_path.stat().st_size / 1024
    print(f"  Saved {file_size:.2f} KB\n")


def print_summary(df: pd.DataFrame, original_count: int, cleaned_count: int) -> None:
    """Print summary statistics."""
    print("=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"Original row count:      {original_count:,}")
    print(f"Cleaned row count:       {cleaned_count:,}")
    print(f"Final row count:         {len(df):,}")
    print(f"Date range:              {df['date'].min()} to {df['date'].max()}")
    print(f"Number of stores:        {df['store_id'].nunique()}")
    print(f"Number of products:      {df['product_id'].nunique()}")
    print(f"Output file:             {OUTPUT_FILE}")
    print("=" * 60)
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print()


def main():
    """Main function to orchestrate the data preparation pipeline."""
    print("=" * 60)
    print("UCI Online Retail Dataset Preparation")
    print("=" * 60 + "\n")

    # Download dataset
    content = download_dataset(DATASET_URL)

    # Load Excel data
    df = load_excel_data(content)
    original_count = len(df)

    # Clean data
    df = clean_data(df)
    cleaned_count = len(df)

    # Transform to our schema
    df = transform_data(df)

    # Aggregate daily sales
    df = aggregate_daily_sales(df)

    # Sample top products to reduce file size
    df = sample_top_products(df, TOP_PRODUCTS)

    # Sort by date
    df = df.sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)

    # Save to CSV
    save_to_csv(df, OUTPUT_FILE)

    # Print summary
    print_summary(df, original_count, cleaned_count)


if __name__ == "__main__":
    main()
