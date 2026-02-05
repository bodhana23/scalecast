#!/usr/bin/env python3
"""Upload raw demand data to S3."""

import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Configuration
LOCAL_FILE = Path(__file__).parent.parent / "data" / "raw" / "demand_data.csv"
S3_BUCKET = "scalecast-mlops-bodhana-2026"
S3_KEY = "raw/demand_data.csv"


def main():
    # Check local file exists
    if not LOCAL_FILE.exists():
        print(f"Error: Source file not found: {LOCAL_FILE}")
        print("Run 'python scripts/download_dataset.py' first.")
        sys.exit(1)

    local_size = LOCAL_FILE.stat().st_size

    print(f"Source file:  {LOCAL_FILE}")
    print(f"Source size:  {local_size / 1024:.2f} KB")
    print(f"Destination:  s3://{S3_BUCKET}/{S3_KEY}\n")

    try:
        # Create S3 client
        s3 = boto3.client("s3")

        # Upload file
        print("Uploading...")
        s3.upload_file(str(LOCAL_FILE), S3_BUCKET, S3_KEY)

        # Verify upload
        response = s3.head_object(Bucket=S3_BUCKET, Key=S3_KEY)
        s3_size = response["ContentLength"]

        print(f"\nUpload successful!")
        print(f"S3 file size: {s3_size / 1024:.2f} KB")

        if s3_size == local_size:
            print("Size verification: PASSED")
        else:
            print(f"Size verification: MISMATCH (local: {local_size}, s3: {s3_size})")

    except ClientError as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
