"""
Utility functions for the fire satellite image crawler.
"""

import os
import datetime
import time
import ee
import requests
from PIL import Image
from io import BytesIO
import json


def get_earth_engine_credentials(
    service_account=None, project_id=None, credentials_file=None, prompt=False
):
    """
    Get Google Earth Engine credentials from various sources.

    Args:
        service_account: Service account email (optional)
        project_id: GEE project ID (optional)
        credentials_file: Path to credentials JSON file (optional)
        prompt: Whether to prompt for credentials if not found

    Returns:
        tuple: (service_account, project_id, credentials_dict)
    """
    credentials_dict = None

    # Try to get credentials from environment variables
    if not service_account:
        service_account = os.environ.get("GEE_SERVICE_ACCOUNT")
    if not project_id:
        project_id = os.environ.get("GEE_PROJECT_ID")

    # Try to load credentials from file
    if credentials_file and os.path.exists(credentials_file):
        try:
            with open(credentials_file, "r") as f:
                credentials_dict = json.load(f)
                if not service_account and "client_email" in credentials_dict:
                    service_account = credentials_dict["client_email"]
        except Exception as e:
            print(f"Error loading credentials file: {str(e)}")

    # If credentials not found and prompt is enabled, ask the user
    if prompt:
        if not project_id:
            project_id = input("Enter your Google Earth Engine project ID: ").strip()

        if not service_account:
            use_service_account = (
                input("Do you want to use a service account? (y/n): ")
                .lower()
                .startswith("y")
            )
            if use_service_account:
                service_account = input("Enter your service account email: ").strip()
                key_file = input(
                    "Enter path to service account key file (leave blank if none): "
                ).strip()
                if key_file and os.path.exists(key_file):
                    try:
                        with open(key_file, "r") as f:
                            credentials_dict = json.load(f)
                    except Exception as e:
                        print(f"Error loading key file: {str(e)}")

    return service_account, project_id, credentials_dict


def initialize_earth_engine(
    service_account=None, project_id=None, credentials_dict=None
):
    """Initialize Google Earth Engine API with authentication."""
    try:
        if service_account and credentials_dict:
            # Initialize with service account and loaded credentials
            credentials = ee.ServiceAccountCredentials(
                service_account, key_data=json.dumps(credentials_dict)
            )
            ee.Initialize(credentials, project=project_id)
        elif service_account:
            # Initialize with service account using default credentials location
            credentials = ee.ServiceAccountCredentials(service_account)
            ee.Initialize(credentials, project=project_id)
        else:
            # Use default authentication with explicit project ID
            try:
                ee.Authenticate()
            except Exception as auth_error:
                print(f"Authentication note: {auth_error}")
                print("Proceeding with existing credentials...")

            # Initialize with project
            ee.Initialize(project=project_id)

        print(
            f"Google Earth Engine initialized successfully with project: {project_id}"
        )
        return True
    except Exception as e:
        print(f"Error initializing Google Earth Engine: {str(e)}")
        return False


def date_to_ee_format(date_str):
    """Convert date string to Earth Engine format."""
    if isinstance(date_str, str):
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return ee.Date(date_obj.strftime("%Y-%m-%d"))
    return ee.Date(date_str)


def get_date_range(center_date, days_before=0, days_after=7):
    """Get date range for image search."""
    if isinstance(center_date, str):
        center_date = datetime.datetime.strptime(center_date, "%Y-%m-%d")

    start_date = center_date - datetime.timedelta(days=days_before)
    end_date = center_date + datetime.timedelta(days=days_after)

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def create_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def download_image_from_url(url, output_path):
    """Download image from URL and save to file."""
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)  # Added timeout
            response.raise_for_status()

            # Check if the response is actually an image
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type and attempt < max_retries - 1:
                print(
                    f"Response is not an image (content type: {content_type}). Retrying..."
                )
                time.sleep(retry_delay)
                continue

            with Image.open(BytesIO(response.content)) as img:
                img.save(output_path)

            return True

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"Request timed out. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Could not download image due to timeout.")
                return False

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error: {http_err}")
            if response.status_code == 429:  # Too Many Requests
                wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                print(f"Rate limited. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            return False

        except Exception as e:
            print(f"Error downloading image: {str(e)}")
            return False

    return False


def apply_fire_indices(image, satellite="sentinel"):
    """Apply fire-related indices to enhance fire visibility."""
    if image is None:
        print("Warning: Attempted to apply fire indices to None image")
        return None

    try:
        if satellite == "sentinel":
            # Get band names as ee.List
            band_names = image.bandNames()

            # Create safer normalized difference function with correct band checking
            def safe_normalized_difference(img, band1, band2):
                # Check if both bands exist in the image
                has_band1 = band_names.contains(band1)
                has_band2 = band_names.contains(band2)
                has_both = ee.Algorithms.And(has_band1, has_band2)

                # Only compute normalized difference if both bands exist
                return ee.Algorithms.If(
                    has_both,
                    img.normalizedDifference([band1, band2]),
                    ee.Image(0),  # Default value if bands are missing
                )

            # Normalized Burn Ratio (NBR) for Sentinel-2
            nbr = safe_normalized_difference(image, "B8", "B12").rename("NBR")

            # Normalized Burn Ratio 2 (NBR2)
            nbr2 = safe_normalized_difference(image, "B11", "B12").rename("NBR2")

            # Mid-Infrared Burn Index
            def compute_mirbi():
                return image.expression(
                    "10 * SWIR1 - 9.8 * SWIR2 + 2",
                    {"SWIR1": image.select("B11"), "SWIR2": image.select("B12")},
                ).rename("MIRBI")

            # Check if required bands exist for MIRBI
            has_b11 = band_names.contains("B11")
            has_b12 = band_names.contains("B12")
            has_required = ee.Algorithms.And(has_b11, has_b12)

            mirbi = ee.Algorithms.If(
                has_required, compute_mirbi(), ee.Image(0).rename("MIRBI")
            )

            # Add the indices as bands
            return image.addBands([nbr, nbr2, mirbi])

        return image

    except Exception as e:
        print(f"Error applying fire indices: {str(e)}")
        return image
