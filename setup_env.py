"""
Setup script to create environment files for Google Earth Engine credentials.
"""

import argparse
from utils.env_setup import create_env_file


def main():
    parser = argparse.ArgumentParser(
        description="Set up environment file for Google Earth Engine credentials"
    )
    parser.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Path to environment file (default: .env)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing credentials"
    )

    args = parser.parse_args()

    print("Fire Satellite Image Crawler - Environment Setup\n")
    print("This script will help you set up your Google Earth Engine credentials.")
    print("These credentials are needed to access satellite imagery.")
    print("\nUseful links:")
    print("- Google Cloud Console: https://console.cloud.google.com")
    print(
        "- Earth Engine API Library: https://console.cloud.google.com/apis/library/earthengine.googleapis.com"
    )
    print(
        "- Service Account Creation: https://console.cloud.google.com/iam-admin/serviceaccounts"
    )
    print(
        "- Earth Engine Service Account Registration: https://signup.earthengine.google.com/#!/service_accounts"
    )

    if create_env_file(env_path=args.env_file, force=args.force):
        print("\nSetup completed successfully!")
        print("\nYou can now run the crawler with:")
        print("python main.py")
    else:
        print("\nSetup was cancelled or failed.")
        print("You can try again or manually create a .env file.")


if __name__ == "__main__":
    main()
