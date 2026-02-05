#!/usr/bin/env python3
"""
Setup Verification Script for ScaleCast.

This script performs comprehensive verification of the ScaleCast environment setup,
checking all required components before starting the services.

Checks performed:
    1. All required environment variables are set (not placeholder values)
    2. AWS credentials are valid (STS call)
    3. S3 bucket is accessible (head_bucket call)
    4. S3 folder structure exists
    5. PostgreSQL connection (only when Docker is running, skip gracefully if not)

Usage:
    python scripts/verify_setup.py

Output:
    Color-coded results with overall status and helpful next steps.
"""

import os
import socket
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def colored(text: str, color: str) -> str:
    """
    Wrap text with ANSI color codes.

    Args:
        text: The text to colorize.
        color: The color code from Colors class.

    Returns:
        str: Colorized text string.
    """
    return f"{color}{text}{Colors.RESET}"


def print_pass(message: str):
    """Print a passing check result in green."""
    print(f"  {colored('✓', Colors.GREEN)} {message}")


def print_fail(message: str):
    """Print a failing check result in red."""
    print(f"  {colored('✗', Colors.RED)} {message}")


def print_warn(message: str):
    """Print a warning in yellow."""
    print(f"  {colored('⚠', Colors.YELLOW)} {message}")


def print_info(message: str):
    """Print an info message in blue."""
    print(f"  {colored('ℹ', Colors.BLUE)} {message}")


def print_section(title: str):
    """Print a section header."""
    print()
    print(colored(f"─── {title} ───", Colors.BOLD))


# Track overall status
class VerificationStatus:
    """Track verification results across all checks."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.skipped = 0
        self.issues = []

    def add_pass(self):
        """Record a passing check."""
        self.passed += 1

    def add_fail(self, issue: str):
        """Record a failing check with issue description."""
        self.failed += 1
        self.issues.append(issue)

    def add_warn(self):
        """Record a warning."""
        self.warnings += 1

    def add_skip(self):
        """Record a skipped check."""
        self.skipped += 1


def load_dotenv_manual() -> dict:
    """
    Manually load environment variables from .env file.

    Returns:
        dict: Dictionary of environment variables from .env file.
    """
    env_file = PROJECT_ROOT / ".env"
    env_vars = {}

    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Parse key=value
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove surrounding quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    env_vars[key] = value
                    # Also set in os.environ for later use
                    os.environ[key] = value

    return env_vars


def check_env_file_exists(status: VerificationStatus) -> bool:
    """
    Check if .env file exists.

    Args:
        status: VerificationStatus instance to track results.

    Returns:
        bool: True if .env file exists.
    """
    env_file = PROJECT_ROOT / ".env"

    if env_file.exists():
        print_pass(f".env file exists at {env_file}")
        status.add_pass()
        return True
    else:
        print_fail(f".env file not found at {env_file}")
        status.add_fail("Missing .env file - copy from .env.example")
        return False


def check_required_env_vars(env_vars: dict, status: VerificationStatus) -> bool:
    """
    Check that all required environment variables are set with real values.

    Args:
        env_vars: Dictionary of environment variables.
        status: VerificationStatus instance to track results.

    Returns:
        bool: True if all required variables are set correctly.
    """
    required_vars = {
        "POSTGRES_USER": None,
        "POSTGRES_PASSWORD": None,
        "POSTGRES_DB": None,
        "AWS_ACCESS_KEY_ID": None,
        "AWS_SECRET_ACCESS_KEY": None,
        "AWS_REGION": "ap-southeast-1",  # Expected value
        "S3_BUCKET_NAME": None,
        "AIRFLOW__CORE__FERNET_KEY": None,
        "AIRFLOW__WEBSERVER__SECRET_KEY": None,
    }

    placeholder_values = [
        "PASTE_YOUR_ACCESS_KEY_HERE",
        "PASTE_YOUR_SECRET_KEY_HERE",
        "PASTE_GENERATED_FERNET_KEY_HERE",
        "PASTE_GENERATED_SECRET_KEY_HERE",
        "your_aws_access_key",
        "your_aws_secret_key",
        "your_fernet_key_here",
        "your_webserver_secret_key_here",
        "your_secure_password_here",
    ]

    all_valid = True

    for var, expected in required_vars.items():
        value = env_vars.get(var)

        if not value:
            print_fail(f"{var} is not set")
            status.add_fail(f"Missing environment variable: {var}")
            all_valid = False

        elif value in placeholder_values:
            print_fail(f"{var} has placeholder value (not replaced)")
            status.add_fail(f"Placeholder value for: {var}")
            all_valid = False

        elif expected and value != expected:
            print_warn(f"{var} = {value} (expected: {expected})")
            status.add_warn()

        else:
            # Mask sensitive values in output
            if "KEY" in var or "PASSWORD" in var or "SECRET" in var:
                masked = value[:4] + "****" + value[-4:] if len(value) > 8 else "****"
                print_pass(f"{var} = {masked}")
            else:
                print_pass(f"{var} = {value}")
            status.add_pass()

    return all_valid


def check_aws_credentials(status: VerificationStatus) -> bool:
    """
    Verify AWS credentials using STS get_caller_identity.

    Args:
        status: VerificationStatus instance to track results.

    Returns:
        bool: True if AWS credentials are valid.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        print_warn("boto3 not installed - skipping AWS credential check")
        print_info("Install with: pip install boto3")
        status.add_skip()
        return False

    try:
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "ap-southeast-1"),
        )

        sts = session.client("sts")
        identity = sts.get_caller_identity()

        account_id = identity["Account"]
        masked_account = f"{account_id[:4]}****{account_id[-2:]}"

        print_pass(f"AWS credentials valid (Account: {masked_account})")
        status.add_pass()
        return True

    except NoCredentialsError:
        print_fail("AWS credentials not found or invalid")
        status.add_fail("Invalid AWS credentials")
        return False

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        print_fail(f"AWS credential check failed: {error_code}")
        status.add_fail(f"AWS credential error: {error_code}")
        return False

    except Exception as e:
        print_fail(f"AWS check error: {str(e)}")
        status.add_fail(f"AWS check exception: {str(e)}")
        return False


def check_s3_bucket(status: VerificationStatus) -> bool:
    """
    Check if S3 bucket exists and is accessible.

    Args:
        status: VerificationStatus instance to track results.

    Returns:
        bool: True if bucket is accessible.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print_warn("boto3 not installed - skipping S3 bucket check")
        status.add_skip()
        return False

    bucket_name = os.getenv("S3_BUCKET_NAME")
    region = os.getenv("AWS_REGION", "ap-southeast-1")

    if not bucket_name:
        print_fail("S3_BUCKET_NAME not set")
        status.add_fail("S3_BUCKET_NAME not configured")
        return False

    try:
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region,
        )

        s3 = session.client("s3")
        s3.head_bucket(Bucket=bucket_name)

        print_pass(f"S3 bucket '{bucket_name}' is accessible")
        status.add_pass()
        return True

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "404":
            print_fail(f"S3 bucket '{bucket_name}' does not exist")
            status.add_fail(f"S3 bucket not found: {bucket_name}")
        elif error_code == "403":
            print_fail(f"Access denied to S3 bucket '{bucket_name}'")
            status.add_fail(f"S3 bucket access denied: {bucket_name}")
        else:
            print_fail(f"S3 bucket check failed: {error_code}")
            status.add_fail(f"S3 bucket error: {error_code}")
        return False

    except Exception as e:
        print_fail(f"S3 check error: {str(e)}")
        status.add_fail(f"S3 check exception: {str(e)}")
        return False


def check_s3_folder_structure(status: VerificationStatus) -> bool:
    """
    Check if S3 folder structure exists.

    Args:
        status: VerificationStatus instance to track results.

    Returns:
        bool: True if all folders exist.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print_warn("boto3 not installed - skipping S3 folder check")
        status.add_skip()
        return False

    bucket_name = os.getenv("S3_BUCKET_NAME")
    region = os.getenv("AWS_REGION", "ap-southeast-1")

    if not bucket_name:
        print_warn("S3_BUCKET_NAME not set - skipping folder check")
        status.add_skip()
        return False

    required_folders = ["raw/", "processed/", "models/", "mlflow/"]

    try:
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region,
        )

        s3 = session.client("s3")
        all_exist = True

        for folder in required_folders:
            response = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=folder,
                MaxKeys=1
            )

            if response.get("KeyCount", 0) > 0:
                print_pass(f"Folder exists: s3://{bucket_name}/{folder}")
                status.add_pass()
            else:
                print_fail(f"Folder missing: s3://{bucket_name}/{folder}")
                status.add_fail(f"Missing S3 folder: {folder}")
                all_exist = False

        return all_exist

    except ClientError as e:
        print_fail(f"S3 folder check failed: {str(e)}")
        status.add_fail("S3 folder check failed")
        return False

    except Exception as e:
        print_fail(f"S3 folder check error: {str(e)}")
        status.add_fail(f"S3 folder check exception: {str(e)}")
        return False


def check_postgres_connection(status: VerificationStatus) -> bool:
    """
    Check PostgreSQL connection (only if Docker is running).

    Args:
        status: VerificationStatus instance to track results.

    Returns:
        bool: True if connection successful or Docker not running.
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))

    # First check if the port is open
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)

    try:
        # If host is 'postgres' (Docker service name), check localhost
        check_host = "localhost" if host == "postgres" else host
        result = sock.connect_ex((check_host, port))
        sock.close()

        if result != 0:
            print_warn(f"PostgreSQL not reachable at {check_host}:{port}")
            print_info("This is expected if Docker is not running yet")
            status.add_skip()
            return True  # Not a failure, just not running

    except socket.error:
        print_warn(f"Could not check PostgreSQL at {host}:{port}")
        print_info("This is expected if Docker is not running yet")
        status.add_skip()
        return True

    # Port is open, try to connect with psycopg2
    try:
        import psycopg2
    except ImportError:
        print_warn("psycopg2 not installed - skipping PostgreSQL connection test")
        print_info("Install with: pip install psycopg2-binary")
        status.add_skip()
        return True

    try:
        conn = psycopg2.connect(
            host="localhost",  # Use localhost for local Docker
            port=port,
            database=os.getenv("POSTGRES_DB", "scalecast"),
            user=os.getenv("POSTGRES_USER", "scalecast"),
            password=os.getenv("POSTGRES_PASSWORD"),
            connect_timeout=5,
        )

        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        print_pass(f"PostgreSQL connected successfully")
        print_info(f"Version: {version.split(',')[0]}")
        status.add_pass()
        return True

    except Exception as e:
        error_msg = str(e).split("\n")[0]
        print_warn(f"PostgreSQL connection failed: {error_msg}")
        print_info("This may be expected if Docker is not running")
        status.add_skip()
        return True


def print_summary(status: VerificationStatus):
    """
    Print verification summary and next steps.

    Args:
        status: VerificationStatus instance with results.
    """
    print()
    print("=" * 60)
    print(colored("  Verification Summary", Colors.BOLD))
    print("=" * 60)

    print(f"""
  {colored('✓', Colors.GREEN)} Passed:   {status.passed}
  {colored('✗', Colors.RED)} Failed:   {status.failed}
  {colored('⚠', Colors.YELLOW)} Warnings: {status.warnings}
  {colored('○', Colors.BLUE)} Skipped:  {status.skipped}
""")

    if status.failed == 0:
        print(colored("═" * 60, Colors.GREEN))
        print(colored("  All critical checks passed!", Colors.GREEN))
        print(colored("═" * 60, Colors.GREEN))
        print("""
Next Steps:
  1. Start the services:
     docker-compose up -d

  2. Wait for services to initialize (about 60 seconds)

  3. Access the UIs:
     • Airflow:  http://localhost:8080 (admin/admin)
     • MLflow:   http://localhost:5000
     • API:      http://localhost:8000 (when started with --profile api)

  4. Monitor logs:
     docker-compose logs -f
""")
    else:
        print(colored("═" * 60, Colors.RED))
        print(colored("  Some checks failed!", Colors.RED))
        print(colored("═" * 60, Colors.RED))
        print("\nIssues to resolve:")
        for i, issue in enumerate(status.issues, 1):
            print(f"  {i}. {issue}")

        print("""
Troubleshooting Steps:
  1. Ensure .env file exists (copy from .env.example)
  2. Run: python scripts/generate_keys.py
  3. Update .env with your AWS credentials and generated keys
  4. Run: python scripts/setup_aws.py
  5. Re-run this verification: python scripts/verify_setup.py
""")


def main():
    """Main entry point for verification script."""
    print("=" * 60)
    print(colored("  ScaleCast Setup Verification", Colors.BOLD))
    print("=" * 60)

    status = VerificationStatus()

    # Check 1: .env file exists
    print_section("Environment File")
    if not check_env_file_exists(status):
        print_summary(status)
        sys.exit(1)

    # Load environment variables
    env_vars = load_dotenv_manual()

    # Check 2: Required environment variables
    print_section("Environment Variables")
    check_required_env_vars(env_vars, status)

    # Check 3: AWS credentials
    print_section("AWS Credentials")
    check_aws_credentials(status)

    # Check 4: S3 bucket access
    print_section("S3 Bucket Access")
    check_s3_bucket(status)

    # Check 5: S3 folder structure
    print_section("S3 Folder Structure")
    check_s3_folder_structure(status)

    # Check 6: PostgreSQL connection
    print_section("PostgreSQL Connection")
    check_postgres_connection(status)

    # Print summary
    print_summary(status)

    # Exit with appropriate code
    sys.exit(0 if status.failed == 0 else 1)


if __name__ == "__main__":
    main()
