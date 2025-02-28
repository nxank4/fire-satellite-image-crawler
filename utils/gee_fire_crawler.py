"""
Main module for the fire satellite image crawler using Google Earth Engine.
"""

import os
import ee
import time
from tqdm import tqdm

from config import (
    SENTINEL_2_COLLECTION,
    LANDSAT_8_COLLECTION,
    MODIS_COLLECTION,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_IMAGE_SCALE,
    MAX_CLOUD_COVER,
    DEFAULT_TIME_WINDOW_DAYS,
    VIS_PARAMS,
)
from utils.utils import (
    initialize_earth_engine,
    get_earth_engine_credentials,
    date_to_ee_format,
    create_output_dir,
    download_image_from_url,
    apply_fire_indices,
)


class FireSatelliteImageCrawler:
    """
    A class to search and download satellite images of fire events using Google Earth Engine.
    """

    def __init__(
        self,
        service_account=None,
        project_id=None,
        credentials_file=None,
        prompt_credentials=False,
    ):
        """
        Initialize the crawler.

        Args:
            service_account: GEE service account credentials
            project_id: GEE project ID
            credentials_file: Path to JSON credentials file
            prompt_credentials: Whether to prompt for credentials if not provided
        """
        # Get credentials from parameters, environment, or prompt
        service_account, project_id, credentials_dict = get_earth_engine_credentials(
            service_account,
            project_id,
            credentials_file,
            prompt_credentials,
        )

        # Initialize Earth Engine
        self.initialized = initialize_earth_engine(
            service_account, project_id, credentials_dict
        )

    def find_active_fires(self, start_date, end_date, region=None, confidence="high"):
        # Convert dates to ee.Date objects
        start = date_to_ee_format(start_date)
        end = date_to_ee_format(end_date)

        # Load MODIS fire product collection
        modis_fires = (
            ee.ImageCollection(MODIS_COLLECTION).filterDate(start, end).select("MaxFRP")
        )  # Maximum Fire Radiative Power

        # Define a default region if none provided (use smaller region instead of global bounds)
        if region is None:
            # Use a smaller region as default (Continental US as example)
            region = ee.Geometry.Rectangle([-125, 24, -66, 50])
            print("No region specified. Using Continental US as default region.")
        else:
            region = ee.Geometry(region)

        # Convert the ImageCollection to a FeatureCollection of fire points
        def extract_fires(image):
            # Get image date
            date = image.date().format("YYYY-MM-DD")

            # Extract pixels with fire (FRP > 0)
            fires = image.gt(0)

            # Create a mask for confidence levels
            if confidence == "high":
                confidence_mask = image.gt(50)
            elif confidence == "nominal":
                confidence_mask = image.gt(20)
            else:
                confidence_mask = image.gt(0)

            # Apply confidence mask
            fires = fires.updateMask(confidence_mask)

            # Reproject to ensure consistent projection before clipping
            fires = fires.reproject(crs="EPSG:4326", scale=1000)

            # Clip fires to region - use clipToCollection for more stability
            try:
                # Convert region to feature collection for more robust clipping
                region_fc = ee.FeatureCollection([ee.Feature(region)])
                fires = fires.clipToCollection(region_fc)
            except Exception:
                # Fallback to regular clip if clipToCollection fails
                fires = fires.clip(region)

            # Use the region as the geometry parameter for reduceToVectors
            vectors = fires.reduceToVectors(
                reducer=ee.Reducer.countEvery(),
                geometry=region,
                scale=1000,
                geometryType="centroid",
                maxPixels=1e9,
            )

            # Add FRP and date as properties
            vectors = vectors.map(
                lambda f: f.set(
                    {
                        "frp": image.reduceRegion(
                            reducer=ee.Reducer.max(), geometry=f.geometry(), scale=1000
                        ).get("MaxFRP"),
                        "date": date,
                    }
                )
            )

            return vectors

        # Apply extract_fires to each image and flatten the results
        try:
            fire_points = ee.FeatureCollection(modis_fires.map(extract_fires).flatten())
            return fire_points
        except Exception as e:
            print(f"Error finding active fires: {str(e)}")
            # Return empty feature collection as fallback
            return ee.FeatureCollection([])

    def get_satellite_images(
        self,
        fire_points,
        satellite="sentinel",
        days_after=DEFAULT_TIME_WINDOW_DAYS,
        max_cloud_cover=MAX_CLOUD_COVER,
    ):
        """
        Get satellite images for detected fire points.
        """
        # Choose collection based on satellite parameter
        collection_id = (
            SENTINEL_2_COLLECTION if satellite == "sentinel" else LANDSAT_8_COLLECTION
        )
        cloud_band = (
            "CLOUDY_PIXEL_PERCENTAGE" if satellite == "sentinel" else "CLOUD_COVER"
        )

        # Function to get images for each fire point
        def get_images_for_point(feature):
            """Get satellite images for a single fire point."""
            # Initialize default values outside try block
            null_feature = ee.Feature(None, {"has_image": 0})

            try:
                point = feature.geometry()
                fire_date_str = feature.get("date")

                # Create richer null_feature with fire info
                null_feature = ee.Feature(
                    None,
                    {
                        "fire_id": feature.id(),
                        "fire_date": fire_date_str,
                        "fire_frp": feature.get("frp"),
                        "has_image": 0,  # Flag to indicate no image found
                    },
                )

                # Create proper EE dates with safe date handling
                fire_date = ee.Date(fire_date_str)

                # Use milliseconds for safer date arithmetic
                start_date = fire_date
                end_date = fire_date.advance(days_after, "day")

                # Add safety check for valid dates
                valid_date = ee.Algorithms.If(
                    ee.Number(end_date.millis()).gt(0),
                    1,  # Valid date
                    0,  # Invalid date
                )

                # Get images within date range and region
                images = (
                    ee.ImageCollection(collection_id)
                    .filterDate(start_date, end_date)
                    .filterBounds(point)
                    .filter(ee.Filter.lt(cloud_band, max_cloud_cover))
                )

                # Check if we have any images
                is_empty = images.size().eq(0)

                # Define process_image function within the try block
                def process_image():
                    # Sort by cloud cover and get the least cloudy image
                    best_image = ee.Image(images.sort(cloud_band).first())

                    try:
                        # Store enhanced image as a property to avoid reprocessing later
                        return ee.Feature(
                            None,
                            {
                                "fire_id": feature.id(),
                                "fire_date": fire_date.format("YYYY-MM-DD"),
                                "fire_frp": feature.get("frp"),
                                "image_id": best_image.id(),
                                "image_date": best_image.date().format("YYYY-MM-DD"),
                                "cloud_cover": best_image.get(cloud_band),
                                "point_geometry": point,
                                "has_image": 1,  # Flag to indicate image found
                                "enhanced": True,  # Flag to indicate enhancement was done
                            },
                        )
                    except Exception as inner_e:
                        print(f"Error processing image: {str(inner_e)}")
                        return null_feature

                # Use ee.Algorithms.If to handle the case of empty collection
                return ee.Algorithms.If(is_empty, null_feature, process_image())

            except Exception as e:
                print(f"Error processing fire point: {str(e)}")
                return null_feature

        # Map over fire points and get images
        image_features = fire_points.map(get_images_for_point)

        # Filter to only keep features with valid images
        image_features = image_features.filter(ee.Filter.eq("has_image", 1))

        return image_features

    def export_fire_images(
        self,
        image_features,
        output_dir=DEFAULT_OUTPUT_DIR,
        scale=DEFAULT_IMAGE_SCALE,
        visualization="fire",
    ):
        """
        Export fire images to local files.

        Args:
            image_features (ee.FeatureCollection): Features with fire images
            output_dir (str): Directory to save images
            scale (int): Image resolution in meters per pixel
            visualization (str): Visualization type ('rgb' or 'fire')

        Returns:
            list: Paths to downloaded images
        """
        create_output_dir(output_dir)
        downloaded_images = []

        # First check if we have any features
        try:
            feature_count = image_features.size().getInfo()
            print(f"Processing {feature_count} valid satellite images")

            if feature_count == 0:
                print("No valid satellite images found for the detected fire points.")
                return []
        except Exception as e:
            print(f"Error checking feature count: {str(e)}")
            return []

        # Convert to Python list to track progress - with error handling
        try:
            features_list = image_features.getInfo()["features"]
        except Exception as e:
            print(f"Error fetching feature information: {str(e)}")
            print(
                "This could be due to errors in Earth Engine processing or network issues."
            )
            print("Please try with a smaller date range or different region.")
            return []

        # Process each feature
        for feature in tqdm(features_list, desc="Exporting fire images"):
            try:
                properties = feature["properties"]
                fire_id = properties["fire_id"]
                fire_date = properties["fire_date"]
                image_date = properties["image_date"]
                image_id = properties["image_id"]

                # Skip features without image information
                if not image_id:
                    continue

                # Determine satellite type from image_id
                satellite = "sentinel" if "S2" in image_id else "landsat8"

                # Get visualization parameters
                vis_params = VIS_PARAMS[satellite][visualization]

                # Get geometry
                point_geometry = ee.Geometry.Point(
                    feature["properties"]["point_geometry"]["coordinates"]
                )

                # Buffer the point to create a region around the fire (1km buffer)
                region = point_geometry.buffer(1000)

                # Get the image with error handling for non-existent assets
                try:
                    if satellite == "sentinel":
                        image_path = SENTINEL_2_COLLECTION + "/" + image_id
                    else:
                        image_path = LANDSAT_8_COLLECTION + "/" + image_id

                    # Check if image exists before loading it
                    image_exists = ee.data.getInfo(image_path) is not None

                    if not image_exists:
                        print(
                            f"Warning: Image asset '{image_path}' not found. Skipping..."
                        )
                        continue

                    # Load the image only if it exists
                    image = ee.Image(image_path)

                    # Apply fire indices only if not already applied
                    image = apply_fire_indices(image, satellite)

                except ee.ee_exception.EEException as asset_error:
                    if "not found" in str(asset_error):
                        print(
                            f"Image '{image_id}' is not available in Earth Engine. Skipping..."
                        )
                    else:
                        print(f"Error accessing image {image_id}: {str(asset_error)}")
                    continue

                # Create export parameters
                filename = f"fire_{fire_id}_{fire_date}_{image_date}.png"
                filepath = os.path.join(output_dir, filename)

                # Get download URL with error handling
                try:
                    url = image.visualize(**vis_params).getThumbURL(
                        {
                            "region": region.bounds().getInfo(),
                            "dimensions": 1024,
                            "format": "png",
                        }
                    )

                    # Download the image
                    if download_image_from_url(url, filepath):
                        downloaded_images.append(filepath)
                        print(f"Downloaded: {filepath}")

                except Exception as url_error:
                    print(f"Error generating URL for fire {fire_id}: {str(url_error)}")
                    continue

                # Sleep to avoid rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"Error exporting image for fire {fire_id}: {str(e)}")
                continue

        return downloaded_images

    def search_and_download_fire_images(
        self,
        start_date,
        end_date,
        region=None,
        satellite="sentinel",
        confidence="high",
        output_dir=DEFAULT_OUTPUT_DIR,
    ):
        """
        Unified method to search for fires and download images in one call.
        """
        if not self.initialized:
            print("Earth Engine not initialized. Cannot proceed.")
            return []

        # Find fire points with error handling
        print(f"Searching for fire points between {start_date} and {end_date}...")
        try:
            fire_points = self.find_active_fires(
                start_date=start_date,
                end_date=end_date,
                region=region,
                confidence=confidence,
            )

            # Get count of fire points
            fire_count = fire_points.size().getInfo()
            print(f"Found {fire_count} fire points")

            if fire_count == 0:
                print("No fire points found. Try changing your search parameters.")
                return []

        except Exception as e:
            print(f"Error finding fire points: {str(e)}")
            return []

        # Get satellite images for fire points with error handling
        print(f"Retrieving {satellite} images for fire points...")
        try:
            image_features = self.get_satellite_images(
                fire_points=fire_points, satellite=satellite
            )

            # Check if we got any valid images
            valid_count = image_features.size().getInfo()
            print(f"Found {valid_count} valid satellite images")

            if valid_count == 0:
                print("No valid satellite images found for the detected fire points.")
                print("This could be due to cloud cover or lack of satellite coverage.")
                return []

        except Exception as e:
            print(f"Error finding satellite images: {str(e)}")
            print("This might be due to invalid date ranges or region specifications.")
            return []

        # Export images
        print("Exporting fire images...")
        downloaded_images = self.export_fire_images(
            image_features=image_features, output_dir=output_dir
        )

        if downloaded_images:
            print(f"Successfully downloaded {len(downloaded_images)} fire images")
        else:
            print("No images could be downloaded. This might be due to:")
            print("1. High cloud cover over fire areas")
            print("2. Limited satellite coverage for the specified time range")
            print("3. Network or API limitations")

        return downloaded_images
