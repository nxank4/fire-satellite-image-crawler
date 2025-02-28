"""
Utility to help set up environment files for credential storage.
"""

import os
import json
import getpass
import configparser
from pathlib import Path
import subprocess
import datetime


def is_gee_authenticated():
    """Check if Google Earth Engine is already authenticated."""
    try:
        # Try running a simple earthengine command
        result = subprocess.run(
            ["earthengine", "ls"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def prompt_for_credentials():
    """Interactively prompt for Google Earth Engine credentials."""
    print("\n========== Google Earth Engine Credentials Setup ==========\n")

    # Check if GEE is already authenticated
    if is_gee_authenticated():
        print("Google Earth Engine credentials detected.")
        proceed = input("Do you want to update them? (y/n): ").lower().strip()
        if proceed != "y":
            return None
    else:
        print("No Google Earth Engine credentials detected.")
        print("Let's set them up now.")

    credentials = {}

    # Project ID is required - with more guidance
    print("\nYour Google Earth Engine Project ID is required.")
    print(
        "You can find this in the Google Cloud Console: https://console.cloud.google.com"
    )
    print("It's displayed at the top of the page or in the project selector dropdown.")
    print("Example: 'my-project-123456' or a custom name you chose.")

    while True:
        project_id = input("\nGoogle Earth Engine Project ID: ").strip()
        if project_id:
            credentials["GEE_PROJECT_ID"] = project_id
            break
        else:
            print("Project ID is required for Google Earth Engine.")

    # Service account is optional - with more guidance
    print("\nYou can optionally use a service account for authentication.")
    print("Service accounts are useful for automated or server applications.")
    print("If you don't have a service account, you can create one at:")
    print("https://console.cloud.google.com/iam-admin/serviceaccounts")
    print(
        "After creating it, register it at: https://signup.earthengine.google.com/#!/service_accounts"
    )

    use_service_account = (
        input("\nDo you want to use a service account? (y/n): ").lower().strip() == "y"
    )

    if use_service_account:
        print("\nEnter your service account email.")
        print("It typically looks like: 'name@your-project-id.iam.gserviceaccount.com'")
        credentials["GEE_SERVICE_ACCOUNT"] = input("Service Account Email: ").strip()

        # Check if user wants to provide a key file
        print("\nIf you have a service account key file (JSON format),")
        print("enter its path below. If not, leave blank.")
        key_file_path = input("Path to service account key file: ").strip()

        if key_file_path and os.path.exists(key_file_path):
            try:
                with open(key_file_path, "r") as f:
                    key_data = json.load(f)

                # Store the key file path
                credentials["GEE_CREDENTIALS_FILE"] = key_file_path
                print("Key file successfully loaded.")
            except Exception as e:
                print(f"Error reading key file: {e}")
                print("Proceeding without key file.")
        elif key_file_path:
            print(f"Warning: Key file not found at '{key_file_path}'")

    return credentials


def create_env_file(credentials=None, env_path=".env", force=False):
    """
    Create or update .env file with credentials.

    Args:
        credentials: Dict of credentials to add to .env
        env_path: Path to .env file
        force: Whether to overwrite existing values

    Returns:
        bool: True if file was created/updated successfully
    """
    if credentials is None:
        credentials = prompt_for_credentials()
        if credentials is None:
            return False

    env_path = Path(env_path)

    # Read existing .env if it exists
    existing_env = {}
    if env_path.exists() and not force:
        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        existing_env[key.strip()] = value.strip()
        except Exception as e:
            print(f"Warning: Could not read existing .env file: {e}")

    # Merge credentials (don't overwrite existing unless force=True)
    for key, value in credentials.items():
        if key not in existing_env or force:
            existing_env[key] = value

    # Write the .env file
    try:
        with open(env_path, "w") as f:
            f.write(
                "# Auto-generated environment file for fire-satellite-image-crawler\n"
            )
            f.write(
                "# Created: {}\n\n".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            )

            for key, value in sorted(existing_env.items()):
                # Quote the value if it contains spaces
                if " " in str(value):
                    value = f'"{value}"'
                f.write(f"{key}={value}\n")

        print(f"\nEnvironment file created at: {env_path.absolute()}")
        return True
    except Exception as e:
        print(f"Error creating environment file: {e}")
        return False


if __name__ == "__main__":
    # If run directly, create .env file interactively
    import datetime

    create_env_file()
