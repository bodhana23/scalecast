#!/usr/bin/env python3
"""Upload trained model files to S3."""

import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Configuration
S3_BUCKET = "scalecast-mlops-bodhana-2026"
MODEL_DIR = Path(__file__).parent.parent / "models"

FILES_TO_UPLOAD = [
    ("demand_model.pkl", "models/demand_model.pkl"),
    ("encoders.pkl", "models/encoders.pkl"),
]


def upload_file(s3, local_path: Path, s3_key: str) -> bool:
    """Upload a file to S3 and verify."""
    if not local_path.exists():
        print(f"  Error: File not found: {local_path}")
        return False

    local_size = local_path.stat().st_size
    print(f"  Local:  {local_path} ({local_size / 1024:.2f} KB)")
    print(f"  S3:     s3://{S3_BUCKET}/{s3_key}")

    s3.upload_file(str(local_path), S3_BUCKET, s3_key)

    # Verify upload
    response = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
    s3_size = response["ContentLength"]

    if s3_size == local_size:
        print("  Status: SUCCESS")
        return True
    else:
        print(f"  Status: SIZE MISMATCH (s3: {s3_size})")
        return False


def main():
    print("Uploading model files to S3...\n")

    try:
        s3 = boto3.client("s3")
        success_count = 0

        for local_name, s3_key in FILES_TO_UPLOAD:
            local_path = MODEL_DIR / local_name
            if upload_file(s3, local_path, s3_key):
                success_count += 1
            print()

        print(f"Completed: {success_count}/{len(FILES_TO_UPLOAD)} files uploaded")
        sys.exit(0 if success_count == len(FILES_TO_UPLOAD) else 1)

    except ClientError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
