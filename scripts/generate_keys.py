#!/usr/bin/env python3
"""
Security Key Generator for ScaleCast.

This script generates the required security keys for the ScaleCast MLOps pipeline:
    - Fernet key for Airflow encryption (AIRFLOW__CORE__FERNET_KEY)
    - Secret key for Airflow webserver (AIRFLOW__WEBSERVER__SECRET_KEY)

The generated keys are cryptographically secure and suitable for production use.

Usage:
    python scripts/generate_keys.py

Output:
    Prints both keys with clear labels and instructions for updating .env file.
"""

import secrets
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

try:
    from cryptography.fernet import Fernet
except ImportError:
    print("✗ Error: Missing required package 'cryptography'")
    print("\nPlease install it:")
    print("  pip install cryptography>=41.0.0")
    print("\nOr install all project dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def generate_fernet_key() -> str:
    """
    Generate a valid Fernet key for Airflow encryption.

    Fernet keys are used by Airflow to encrypt sensitive data like
    connection passwords and variables.

    Returns:
        str: A base64-encoded 32-byte Fernet key.
    """
    key = Fernet.generate_key()
    return key.decode("utf-8")


def generate_secret_key(length: int = 32) -> str:
    """
    Generate a random secret key for Airflow webserver.

    This key is used to sign session cookies and other security tokens.

    Args:
        length: Length of the secret key in characters (default: 32).

    Returns:
        str: A random hexadecimal string of the specified length.
    """
    # Generate random bytes and convert to hex
    # Each byte becomes 2 hex characters, so we need length/2 bytes
    return secrets.token_hex(length // 2)


def print_keys(fernet_key: str, secret_key: str):
    """
    Print the generated keys with formatting and instructions.

    Args:
        fernet_key: The generated Fernet key.
        secret_key: The generated secret key.
    """
    print("=" * 60)
    print("  ScaleCast Security Key Generator")
    print("=" * 60)
    print()

    print("─" * 60)
    print("  Generated Keys")
    print("─" * 60)
    print()

    # Fernet Key Section
    print("┌" + "─" * 58 + "┐")
    print("│  AIRFLOW__CORE__FERNET_KEY                               │")
    print("├" + "─" * 58 + "┤")
    print("│                                                          │")
    print(f"│  {fernet_key}  │")
    print("│                                                          │")
    print("└" + "─" * 58 + "┘")
    print()

    # Secret Key Section
    print("┌" + "─" * 58 + "┐")
    print("│  AIRFLOW__WEBSERVER__SECRET_KEY                          │")
    print("├" + "─" * 58 + "┤")
    print("│                                                          │")
    print(f"│  {secret_key}                          │")
    print("│                                                          │")
    print("└" + "─" * 58 + "┘")
    print()

    print("─" * 60)
    print("  Copy-Paste Block for .env")
    print("─" * 60)
    print()
    print("Copy the following lines and paste them into your .env file,")
    print("replacing the placeholder values:")
    print()
    print("```")
    print(f"AIRFLOW__CORE__FERNET_KEY={fernet_key}")
    print(f"AIRFLOW__WEBSERVER__SECRET_KEY={secret_key}")
    print("```")
    print()

    print("─" * 60)
    print("  Instructions")
    print("─" * 60)
    print("""
1. Open your .env file:
   nano .env
   # or
   code .env

2. Find these lines:
   AIRFLOW__CORE__FERNET_KEY=PASTE_GENERATED_FERNET_KEY_HERE
   AIRFLOW__WEBSERVER__SECRET_KEY=PASTE_GENERATED_SECRET_KEY_HERE

3. Replace the placeholder values with the generated keys above.

4. Save the file.

5. Verify your setup:
   python scripts/verify_setup.py

IMPORTANT SECURITY NOTES:
  • Never commit these keys to version control
  • Never share these keys publicly
  • Generate new keys for each environment (dev, staging, prod)
  • The .env file is already in .gitignore
""")


def main():
    """Main entry point for key generation script."""
    print("\nGenerating security keys...\n")

    # Generate keys
    fernet_key = generate_fernet_key()
    secret_key = generate_secret_key(32)

    # Print formatted output
    print_keys(fernet_key, secret_key)

    print("✓ Keys generated successfully!\n")


if __name__ == "__main__":
    main()
