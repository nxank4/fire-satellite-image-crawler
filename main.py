"""
Main script for the fire satellite image crawler.
"""

import os
import sys
import datetime
import argparse
import ee
from pathlib import Path
from dotenv import load_dotenv

# First try to load environment variables from .env
env_file = Path(".env")
if env_file.exists():
    print(f"Loading environment from {env_file.absolute()}")
    load_dotenv()
else:
    print("No .env file found. You may need to run setup_env.py first.")

# Import after loading .env to ensure credentials are available
from utils.gee_fire_crawler import FireSatelliteImageCrawler
from config import DEFAULT_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Fire Satellite Image Crawler")
    # Credentials arguments
    parser.add_argument(
        "--project_id",
        type=str,
        default=os.environ.get("GEE_PROJECT_ID"),
        help="Google Earth Engine project ID (required)",
    )
    parser.add_argument(
        "--service_account",
        type=str,
        default=os.environ.get("GEE_SERVICE_ACCOUNT"),
        help="Google Earth Engine service account email",
    )
    parser.add_argument(
        "--credentials_file",
        type=str,
        default=os.environ.get("GEE_CREDENTIALS_FILE"),
        help="Path to Google Earth Engine credentials JSON file",
    )
    parser.add_argument(
        "--prompt_credentials",
        action="store_true",
        help="Prompt for credentials if not provided",
    )

    # Other arguments
    parser.add_argument(
        "--start_date",
        type=str,
        default=(datetime.datetime.now() - datetime.timedelta(days=30)).strftime(
            "%Y-%m-%d"
        ),
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=datetime.datetime.now().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--satellite",
        type=str,
        choices=["sentinel", "landsat8"],
        default="sentinel",
        help="Satellite to use",
    )
    parser.add_argument(
        "--confidence",
        type=str,
        choices=["low", "nominal", "high"],
        default="high",
        help="Fire detection confidence",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for images",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Region of interest as 'lon,lat' (center of a 100km square)",
    )

    args = parser.parse_args()

    # Check if we need to set up credentials
    if not args.project_id and not args.prompt_credentials:
        print("ERROR: Google Earth Engine project ID is required.")
        print("You can:")
        print("1. Run setup_env.py to create a .env file with your credentials")
        print("2. Provide --project_id on the command line")
        print("3. Use --prompt_credentials to enter credentials interactively")
        print("\nTo find your project ID:")
        print("- Visit the Google Cloud Console: https://console.cloud.google.com")
        print(
            "- Your project ID is displayed at the top of the page or in the project dropdown"
        )
        print(
            "- Make sure the Earth Engine API is enabled: https://console.cloud.google.com/apis/library"
        )
        return 1

    # Initialize the crawler
    crawler = FireSatelliteImageCrawler(
        service_account=args.service_account,
        project_id=args.project_id,
        credentials_file=args.credentials_file,
        prompt_credentials=args.prompt_credentials,
    )

    if not crawler.initialized:
        print("Failed to initialize Earth Engine. Check your credentials.")
        print("\nTroubleshooting tips:")
        print("1. Make sure your Google Cloud project exists and the ID is correct")
        print(
            "2. Ensure the Earth Engine API is enabled: https://console.cloud.google.com/apis/library"
        )
        print("3. Verify you've authenticated with: earthengine authenticate")
        print(
            "4. If using a service account, ensure it's registered at: https://signup.earthengine.google.com/#!/service_accounts"
        )
        return 1

    # Create region geometry if provided
    region = None
    if args.region:
        try:
            lon, lat = map(float, args.region.split(","))
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(50000)  # 50km buffer around point
            print(f"Set region to 100km square around {lon},{lat}")
        except Exception as e:
            print(f"Invalid region format: {e}")
            print("Using global search.")

    # Search and download fire images with better error handling
    try:
        downloaded_images = crawler.search_and_download_fire_images(
            start_date=args.start_date,
            end_date=args.end_date,
            region=region,
            satellite=args.satellite,
            confidence=args.confidence,
            output_dir=args.output_dir,
        )

        if downloaded_images:
            print(f"Successfully downloaded {len(downloaded_images)} images.")
            print(f"First few images: {downloaded_images[:3]}")
            return 0
        else:
            print("No images were downloaded. Try changing parameters or date range.")
            print("\nCommon troubleshooting tips:")
            print("1. Ensure your date range is valid and in the past (not future)")
            print("2. Try a wider date range")
            print("3. Try a different region")
            print(
                "4. Check if there were active fires in your selected region/timeframe"
            )
            return 0

    except Exception as e:
        print(f"\nError while processing: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your date range format (YYYY-MM-DD)")
        print("2. Make sure your region coordinates are valid")
        print("3. Try a smaller date range")
        print("4. Ensure your Earth Engine authentication is valid")
        return 1


if __name__ == "__main__":
    sys.exit(main())
