#!/usr/bin/env python3
"""
AWS Setup Script for ScaleCast.

This script verifies AWS credentials and sets up the required S3 bucket
and folder structure for the ScaleCast MLOps pipeline.

Features:
    - Loads environment variables from .env using python-dotenv
    - Verifies AWS credentials using STS get_caller_identity()
    - Checks if the S3 bucket exists and is accessible
    - Creates folder structure with .gitkeep placeholder files
    - Uses ap-southeast-1 (Singapore) region explicitly
    - Comprehensive error handling with helpful messages

Usage:
    python scripts/setup_aws.py

Environment Variables Required:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_REGION (should be ap-southeast-1)
    - S3_BUCKET_NAME
"""

import os
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# Default region for ScaleCast (Singapore)
DEFAULT_REGION = "ap-southeast-1"

try:
    import boto3
    from botocore.exceptions import (
        ClientError,
        NoCredentialsError,
        EndpointConnectionError,
        ConnectTimeoutError,
    )
    from dotenv import load_dotenv
except ImportError as e:
    print(f"✗ Error: Missing required package - {e}")
    print("\nPlease install dependencies:")
    print("  pip install boto3 python-dotenv")
    sys.exit(1)


def load_environment() -> dict:
    """
    Load environment variables from .env file.

    Returns:
        dict: Configuration dictionary with AWS credentials and settings.

    Raises:
        SystemExit: If required environment variables are missing.
    """
    env_file = PROJECT_ROOT / ".env"

    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment from {env_file}")
    else:
        print(f"⚠ No .env file found at {env_file}")
        print("  Attempting to use system environment variables...")

    required_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "S3_BUCKET_NAME",
    ]

    config = {}
    missing = []
    placeholders = []

    placeholder_values = [
        "PASTE_YOUR_ACCESS_KEY_HERE",
        "PASTE_YOUR_SECRET_KEY_HERE",
        "your_aws_access_key",
        "your_aws_secret_key",
    ]

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)
        elif value in placeholder_values:
            placeholders.append(var)
        else:
            config[var] = value

    if missing:
        print(f"\n✗ Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease set these in your .env file.")
        sys.exit(1)

    if placeholders:
        print(f"\n✗ Placeholder values detected (not replaced with real credentials):")
        for var in placeholders:
            print(f"  - {var}")
        print("\nPlease update your .env file with actual AWS credentials.")
        sys.exit(1)

    # Validate region
    if config.get("AWS_REGION") != DEFAULT_REGION:
        print(f"\n⚠ Warning: AWS_REGION is set to '{config.get('AWS_REGION')}'")
        print(f"  Expected region: {DEFAULT_REGION} (Singapore)")
        print("  Continuing with configured region...")

    return config


def mask_account_id(account_id: str) -> str:
    """
    Mask AWS account ID for security (show first 4 and last 2 digits).

    Args:
        account_id: The AWS account ID to mask.

    Returns:
        str: Masked account ID (e.g., "1234****89").
    """
    if len(account_id) >= 6:
        return f"{account_id[:4]}****{account_id[-2:]}"
    return "****"


def verify_credentials(config: dict) -> tuple[boto3.Session, dict]:
    """
    Verify AWS credentials are valid using STS get_caller_identity.

    Args:
        config: Configuration dictionary with AWS credentials.

    Returns:
        tuple: (boto3.Session, identity_info dict)

    Raises:
        SystemExit: If credentials are invalid or verification fails.
    """
    print("\n" + "─" * 50)
    print("  Verifying AWS Credentials")
    print("─" * 50)

    try:
        session = boto3.Session(
            aws_access_key_id=config["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"],
            region_name=config["AWS_REGION"],
        )

        # Test credentials by getting caller identity
        sts = session.client("sts")
        identity = sts.get_caller_identity()

        masked_account = mask_account_id(identity["Account"])

        print(f"✓ AWS credentials valid")
        print(f"  Account ID: {masked_account}")
        print(f"  User ARN: {identity['Arn']}")
        print(f"  Region: {config['AWS_REGION']}")

        return session, identity

    except NoCredentialsError:
        print("✗ AWS credentials not found")
        print("\nTroubleshooting:")
        print("  1. Check that AWS_ACCESS_KEY_ID is set correctly in .env")
        print("  2. Check that AWS_SECRET_ACCESS_KEY is set correctly in .env")
        print("  3. Ensure there are no extra spaces or quotes around the values")
        sys.exit(1)

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))

        print(f"✗ AWS credential verification failed")
        print(f"  Error Code: {error_code}")
        print(f"  Message: {error_message}")

        if error_code == "InvalidClientTokenId":
            print("\nTroubleshooting:")
            print("  - Your AWS_ACCESS_KEY_ID appears to be invalid")
            print("  - Verify the access key in AWS IAM Console")
            print("  - Check if the key has been deactivated or deleted")
        elif error_code == "SignatureDoesNotMatch":
            print("\nTroubleshooting:")
            print("  - Your AWS_SECRET_ACCESS_KEY appears to be incorrect")
            print("  - Re-copy the secret key from AWS IAM Console")
            print("  - Ensure no extra whitespace was included")

        sys.exit(1)

    except EndpointConnectionError:
        print("✗ Network connection error")
        print("\nTroubleshooting:")
        print("  - Check your internet connection")
        print("  - Verify you can reach AWS endpoints")
        print("  - Check if a firewall is blocking the connection")
        sys.exit(1)

    except ConnectTimeoutError:
        print("✗ Connection timeout")
        print("\nTroubleshooting:")
        print("  - Check your internet connection")
        print("  - Try again in a few moments")
        print("  - Verify the AWS_REGION is correct")
        sys.exit(1)


def check_bucket_access(session: boto3.Session, bucket_name: str, region: str) -> bool:
    """
    Check if the S3 bucket exists and is accessible.

    Args:
        session: boto3 Session with valid credentials.
        bucket_name: Name of the S3 bucket to check.
        region: AWS region for the bucket.

    Returns:
        bool: True if bucket is accessible, False otherwise.

    Raises:
        SystemExit: If bucket doesn't exist or access is denied.
    """
    print("\n" + "─" * 50)
    print("  Checking S3 Bucket Access")
    print("─" * 50)

    s3 = session.client("s3", region_name=region)

    try:
        # Check if bucket exists and we have access
        s3.head_bucket(Bucket=bucket_name)

        # Get bucket location to confirm
        location = s3.get_bucket_location(Bucket=bucket_name)
        bucket_region = location.get("LocationConstraint") or "us-east-1"

        print(f"✓ Bucket '{bucket_name}' exists and is accessible")
        print(f"  Bucket Region: {bucket_region}")

        if bucket_region != region:
            print(f"\n⚠ Warning: Bucket is in {bucket_region}, but AWS_REGION is {region}")
            print("  This may cause issues. Consider updating AWS_REGION in .env")

        return True

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")

        if error_code == "404" or error_code == "NoSuchBucket":
            print(f"✗ Bucket '{bucket_name}' does not exist")
            print("\nTroubleshooting:")
            print(f"  1. Create the bucket in AWS S3 Console in region {region}")
            print(f"  2. Or update S3_BUCKET_NAME in .env to an existing bucket")
            print(f"  3. Bucket names must be globally unique")
            sys.exit(1)

        elif error_code == "403" or error_code == "AccessDenied":
            print(f"✗ Access denied to bucket '{bucket_name}'")
            print("\nTroubleshooting:")
            print("  1. Check that your IAM user has s3:ListBucket permission")
            print("  2. Check that your IAM user has s3:GetBucketLocation permission")
            print("  3. Verify the bucket policy allows your IAM user access")
            print("  4. Ensure the bucket isn't owned by a different AWS account")
            sys.exit(1)

        else:
            print(f"✗ Error checking bucket: {error_code}")
            print(f"  Message: {e.response.get('Error', {}).get('Message', str(e))}")
            sys.exit(1)


def create_folder_structure(session: boto3.Session, bucket_name: str, region: str) -> list:
    """
    Create folder structure in S3 bucket by uploading .gitkeep placeholder files.

    Args:
        session: boto3 Session with valid credentials.
        bucket_name: Name of the S3 bucket.
        region: AWS region for the bucket.

    Returns:
        list: List of successfully created folders.
    """
    print("\n" + "─" * 50)
    print("  Creating Folder Structure")
    print("─" * 50)

    s3 = session.client("s3", region_name=region)

    folders = [
        "raw/.gitkeep",
        "processed/.gitkeep",
        "models/.gitkeep",
        "mlflow/.gitkeep",
    ]

    created_folders = []

    for folder_path in folders:
        folder_name = folder_path.replace(".gitkeep", "")

        try:
            # Check if folder already has content
            response = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=folder_name,
                MaxKeys=1
            )

            if response.get("KeyCount", 0) > 0:
                print(f"✓ Folder exists: s3://{bucket_name}/{folder_name}")
                created_folders.append(folder_name)
            else:
                # Create .gitkeep placeholder
                s3.put_object(
                    Bucket=bucket_name,
                    Key=folder_path,
                    Body=b"",  # Empty file
                    ContentType="application/octet-stream"
                )
                print(f"✓ Created folder: s3://{bucket_name}/{folder_name}")
                created_folders.append(folder_name)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            print(f"✗ Failed to create {folder_name}: {error_code}")
            print(f"  {e.response.get('Error', {}).get('Message', str(e))}")

    return created_folders


def print_summary(bucket_name: str, region: str, account_id: str, folders: list):
    """
    Print setup summary and next steps.

    Args:
        bucket_name: Name of the S3 bucket.
        region: AWS region.
        account_id: AWS account ID (will be masked).
        folders: List of created folders.
    """
    masked_account = mask_account_id(account_id)

    print("\n" + "=" * 50)
    print("  ScaleCast AWS Setup Complete!")
    print("=" * 50)

    print(f"""
┌─────────────────────────────────────────────────┐
│  Configuration Summary                          │
├─────────────────────────────────────────────────┤
│  AWS Account:  {masked_account:<33}│
│  Bucket:       {bucket_name:<33}│
│  Region:       {region:<33}│
└─────────────────────────────────────────────────┘

Folder Structure Created:""")

    for folder in folders:
        print(f"  ✓ s3://{bucket_name}/{folder}")

    print(f"""
Folder Purposes:
  • raw/        → Raw input data files
  • processed/  → Processed/transformed datasets
  • models/     → Trained model artifacts
  • mlflow/     → MLflow experiment tracking artifacts

Next Steps:
  1. Generate security keys:
     python scripts/generate_keys.py

  2. Update .env with generated keys

  3. Verify complete setup:
     python scripts/verify_setup.py

  4. Start Docker services:
     docker-compose up -d

  5. Access the UIs:
     • Airflow:  http://localhost:8080
     • MLflow:   http://localhost:5000
""")


def main():
    """Main entry point for AWS setup script."""
    print("=" * 50)
    print("  ScaleCast AWS Setup Script")
    print("  Region: ap-southeast-1 (Singapore)")
    print("=" * 50)

    # Step 1: Load configuration from .env
    config = load_environment()

    # Step 2: Verify AWS credentials
    session, identity = verify_credentials(config)

    # Step 3: Check bucket access
    bucket_name = config["S3_BUCKET_NAME"]
    region = config["AWS_REGION"]

    check_bucket_access(session, bucket_name, region)

    # Step 4: Create folder structure
    created_folders = create_folder_structure(session, bucket_name, region)

    if not created_folders:
        print("\n✗ Setup failed: Could not create any folders")
        sys.exit(1)

    # Step 5: Print summary
    print_summary(bucket_name, region, identity["Account"], created_folders)

    print("✓ AWS setup completed successfully!\n")


if __name__ == "__main__":
    main()
