from datetime import timedelta
import os
import pickle
import pandas as pd
from pystac_client import Client
import planetary_computer as pc
import numpy as np
import rasterio
from rasterio import windows, warp

from config import CACHE_DIR
from utils.cache import get_cache_key


def get_sentinel_image(
    fire_row, buffer_size=0.05, cloud_cover_limit=10, search_days=10, use_cache=True
):
    """
    Get Sentinel-2 image for a single fire location with caching.

    Args:
        fire_row: Row from DataFrame containing fire data with geometry and acq_date
        buffer_size: Size of buffer around fire point in degrees (default: 0.05)
        cloud_cover_limit: Maximum acceptable cloud cover percentage (default: 10)
        search_days: Days before and after the fire date to search (default: 10)
        use_cache: Whether to use cached images (default: True)

    Returns:
        pystac.Item or None: The selected Sentinel-2 image or None if no suitable image found
    """
    try:
        # Extract location data
        fire_point = fire_row.geometry
        fire_date = pd.to_datetime(fire_row.acq_date)
        location_str = f"{fire_date.date()} at {fire_point.y:.4f}, {fire_point.x:.4f}"

        # Try loading from cache first
        if use_cache:
            cached_item = _load_from_cache(
                fire_date, fire_point, cloud_cover_limit, search_days
            )
            if cached_item:
                print(f"Loaded image from cache for {location_str}")
                return cached_item

        print(f"Searching for images around {location_str}")

        # Prepare search parameters
        area_of_interest = fire_point.buffer(buffer_size).__geo_interface__
        date_start = (fire_date - timedelta(days=search_days)).strftime("%Y-%m-%d")
        date_end = (fire_date + timedelta(days=search_days)).strftime("%Y-%m-%d")
        time_of_interest = f"{date_start}/{date_end}"

        # Search the Planetary Computer catalog
        items = _search_planetary_computer(
            area_of_interest, time_of_interest, cloud_cover_limit
        )

        if not items:
            print(f"No suitable images found for {location_str}")
            return None

        # Select the best image (least cloudy)
        selected_item = _select_best_image(items)

        return selected_item

    except Exception as e:
        print(f"Error in get_sentinel_image: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def _load_from_cache(fire_date, fire_point, cloud_cover_limit, search_days):
    """Load image from cache if available."""
    try:
        cache_key = get_cache_key(
            fire_date, fire_point.y, fire_point.x, cloud_cover_limit, search_days
        )
        cache_file = os.path.join(CACHE_DIR, f"sentinel_{cache_key}.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading from cache: {str(e)}")

    return None


def create_rgb_array(item, fire_row, buffer_size=0.05, auto_adjust=True):
    """
    Create an RGB visualization array from Sentinel-2 data with auto-adjusted brightness.

    Args:
        item: STAC item from Sentinel-2
        fire_row: Row from DataFrame containing fire data
        buffer_size: Size of buffer around fire point in degrees
        auto_adjust: Whether to automatically adjust brightness based on image content (default: True)

    Returns:
        numpy.ndarray: RGB array (height, width, 3) normalized to 0-1 range or None if failed
    """
    try:
        # Check if item is available
        if item is None:
            print("Cannot create RGB array: STAC item is None")
            return None

        # Handle different possible asset keys for visual imagery
        visual_assets = ["visual", "true-color", "true_color"]
        available_assets = set(item.assets.keys())

        # Find the right asset key
        chosen_asset = None
        for asset in visual_assets:
            if asset in available_assets:
                chosen_asset = asset
                break

        if chosen_asset is None:
            # Try to use raw bands (B04, B03, B02) if available
            if all(band in available_assets for band in ["B04", "B03", "B02"]):
                print("Using raw bands (B04, B03, B02) for RGB visualization")
                return create_rgb_array_from_bands(
                    item,
                    fire_row,
                    bands=["B04", "B03", "B02"],
                    buffer_size=buffer_size,
                    auto_adjust=auto_adjust,
                )
            else:
                print(
                    f"Cannot create RGB array: No visual assets found. Available assets: {', '.join(available_assets)}"
                )
                return None

        # Get the href to the visual asset
        asset_href = pc.sign(item.assets[chosen_asset].href)

        # Process the image
        with rasterio.open(asset_href) as ds:
            # Create buffer and calculate bounds
            point_buffer = fire_row.geometry.buffer(buffer_size)
            from rasterio import features

            aoi_bounds = features.bounds(point_buffer.__geo_interface__)
            warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)

            # Get window for the area of interest
            aoi_window = windows.from_bounds(*warped_aoi_bounds, transform=ds.transform)

            # For RGB, we need to read 3 bands
            if ds.count >= 3:
                rgb_data = ds.read((1, 2, 3), window=aoi_window)

                # Transpose to get bands as last dimension (height, width, channels)
                rgb_data = np.transpose(rgb_data, (1, 2, 0))

                # Check if we got valid data
                if rgb_data.size == 0:
                    print(
                        f"Warning: Empty RGB data for fire at {fire_row.geometry.y:.4f}, {fire_row.geometry.x:.4f}"
                    )
                    return None

                # Auto-adjust parameters based on image brightness
                if auto_adjust:
                    # Get image statistics
                    mean_brightness = np.mean(rgb_data)
                    brightness_std = np.std(rgb_data)
                    p95_brightness = np.percentile(rgb_data, 95)

                    # Debug info
                    print(
                        f"Image stats: mean={mean_brightness:.1f}, std={brightness_std:.1f}, p95={p95_brightness:.1f}"
                    )

                    # Adjust parameters based on image content
                    if (
                        p95_brightness > 3000 or mean_brightness > 1500
                    ):  # Very bright image (snow, clouds, desert)
                        contrast_clip = (5, 95)
                        brightness_factor = 0.75
                        gamma = 1.3
                        print(
                            "Auto-adjust: Very bright image detected, applying stronger adjustments"
                        )
                    elif (
                        p95_brightness > 2000 or mean_brightness > 1000
                    ):  # Moderately bright
                        contrast_clip = (3, 97)
                        brightness_factor = 0.85
                        gamma = 1.2
                        print(
                            "Auto-adjust: Moderately bright image detected, applying medium adjustments"
                        )
                    else:  # Normal or darker image
                        contrast_clip = (2, 98)
                        brightness_factor = 0.95
                        gamma = 1.1
                        print(
                            "Auto-adjust: Normal brightness image detected, applying mild adjustments"
                        )

                    # Additional adjustment for high contrast scenes
                    if brightness_std > 1000:
                        contrast_clip = (
                            5,
                            95,
                        )  # Use more aggressive contrast clipping for high-variance scenes
                        print(
                            "Auto-adjust: High contrast image detected, using tighter contrast limits"
                        )
                else:
                    # Default settings if auto_adjust is False
                    contrast_clip = (2, 98)
                    brightness_factor = 0.85
                    gamma = 1.2

                # Normalize data to 0-1 range for each band independently
                rgb_normalized = np.zeros_like(rgb_data, dtype=float)
                for i in range(rgb_data.shape[2]):
                    band = rgb_data[:, :, i].astype(float)

                    min_val = np.percentile(band, contrast_clip[0])
                    max_val = np.percentile(band, contrast_clip[1])

                    if max_val > min_val:
                        rgb_normalized[:, :, i] = np.clip(
                            (band - min_val) / (max_val - min_val), 0, 1
                        )
                    else:
                        # Default to mid-gray if no contrast range
                        rgb_normalized[:, :, i] = 0.5

                # Apply gamma correction to increase contrast
                rgb_normalized = np.power(rgb_normalized, gamma)

                # Apply brightness reduction
                rgb_normalized = rgb_normalized * brightness_factor

            else:
                # If not enough bands, use grayscale
                gray_data = ds.read(1, window=aoi_window)

                # Check if we got valid data
                if gray_data.size == 0:
                    print(
                        f"Warning: Empty grayscale data for fire at {fire_row.geometry.y:.4f}, {fire_row.geometry.x:.4f}"
                    )
                    return None

                # Auto-adjust parameters for grayscale image
                if auto_adjust:
                    mean_brightness = np.mean(gray_data)
                    p95_brightness = np.percentile(gray_data, 95)

                    if p95_brightness > 3000 or mean_brightness > 1500:
                        contrast_clip = (5, 95)
                        brightness_factor = 0.75
                        gamma = 1.3
                    elif p95_brightness > 2000 or mean_brightness > 1000:
                        contrast_clip = (3, 97)
                        brightness_factor = 0.85
                        gamma = 1.2
                    else:
                        contrast_clip = (2, 98)
                        brightness_factor = 0.95
                        gamma = 1.1
                else:
                    contrast_clip = (2, 98)
                    brightness_factor = 0.85
                    gamma = 1.2

                min_val = np.percentile(gray_data, contrast_clip[0])
                max_val = np.percentile(gray_data, contrast_clip[1])

                if max_val > min_val:
                    normalized = np.clip(
                        (gray_data - min_val) / (max_val - min_val), 0, 1
                    )
                    # Apply gamma correction
                    normalized = np.power(normalized, gamma)
                    # Apply brightness reduction
                    normalized = normalized * brightness_factor
                    # Convert to 3D array for matplotlib
                    rgb_normalized = np.stack(
                        [normalized, normalized, normalized], axis=2
                    )
                else:
                    print(
                        f"Warning: Could not normalize grayscale data (min={min_val}, max={max_val})"
                    )
                    rgb_normalized = (
                        np.ones((gray_data.shape[0], gray_data.shape[1], 3)) * 0.5
                    )

        # Return normalized RGB array
        return rgb_normalized

    except Exception as e:
        print(f"Error creating RGB array: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def create_rgb_array_from_bands(
    item, fire_row, bands=["B04", "B03", "B02"], buffer_size=0.05, auto_adjust=True
):
    """
    Create an RGB visualization array from individual Sentinel-2 bands.

    Args:
        item: STAC item from Sentinel-2
        fire_row: Row from DataFrame containing fire data
        bands: List of band identifiers to use for RGB (default: ["B04", "B03", "B02"])
        buffer_size: Size of buffer around fire point in degrees
        auto_adjust: Whether to automatically adjust brightness based on image content

    Returns:
        numpy.ndarray: RGB array (height, width, 3) normalized to 0-1 range or None if failed
    """
    try:
        # Check if item and required bands are available
        if item is None or not all(band in item.assets for band in bands):
            missing = [band for band in bands if band not in item.assets]
            print(f"Cannot create RGB array: Missing bands {missing}")
            return None

        # Create buffer around point
        point_buffer = fire_row.geometry.buffer(buffer_size)
        from rasterio import features

        aoi_bounds = features.bounds(point_buffer.__geo_interface__)

        # Load individual bands
        band_data = []
        band_shapes = []

        for band_id in bands:
            band_href = pc.sign(item.assets[band_id].href)

            with rasterio.open(band_href) as ds:
                warped_aoi_bounds = warp.transform_bounds(
                    "epsg:4326", ds.crs, *aoi_bounds
                )
                aoi_window = windows.from_bounds(
                    *warped_aoi_bounds, transform=ds.transform
                )
                data = ds.read(1, window=aoi_window)

                if data.size == 0:
                    print(f"Warning: Empty data for band {band_id}")
                    return None

                band_data.append(data)
                band_shapes.append(data.shape)

        # Determine the target shape (use the highest resolution)
        target_shape = max(band_shapes, key=lambda shape: shape[0] * shape[1])

        # Create output array
        rgb_array = np.zeros((*target_shape, 3), dtype=float)

        # Auto-adjust parameters based on image brightness
        if auto_adjust:
            # Calculate mean brightness across all bands
            mean_values = [np.mean(data) for data in band_data]
            mean_brightness = np.mean(mean_values)
            p95_values = [np.percentile(data, 95) for data in band_data]
            p95_brightness = np.mean(p95_values)

            print(f"Band stats: mean={mean_brightness:.1f}, p95={p95_brightness:.1f}")

            # Adjust parameters based on image content
            if p95_brightness > 3000 or mean_brightness > 1500:
                contrast_clip = (5, 95)
                brightness_factor = 0.75
                gamma = 1.3
                print(
                    "Auto-adjust: Very bright bands detected, applying stronger adjustments"
                )
            elif p95_brightness > 2000 or mean_brightness > 1000:
                contrast_clip = (3, 97)
                brightness_factor = 0.85
                gamma = 1.2
                print(
                    "Auto-adjust: Moderately bright bands detected, applying medium adjustments"
                )
            else:
                contrast_clip = (2, 98)
                brightness_factor = 0.95
                gamma = 1.1
                print(
                    "Auto-adjust: Normal brightness bands detected, applying mild adjustments"
                )
        else:
            # Default settings if auto_adjust is False
            contrast_clip = (2, 98)
            brightness_factor = 0.85
            gamma = 1.2

        # Process each band
        for i, data in enumerate(band_data):
            # Resize if necessary (some bands may have different resolutions)
            if data.shape != target_shape:
                from skimage.transform import resize

                data = resize(data, target_shape, preserve_range=True)

            # Normalize to 0-1 range using percentile clipping
            p_low = np.percentile(data, contrast_clip[0])
            p_high = np.percentile(data, contrast_clip[1])

            if p_high > p_low:
                normalized = np.clip((data - p_low) / (p_high - p_low), 0, 1)
            else:
                normalized = np.ones_like(data, dtype=float) * 0.5

            rgb_array[:, :, i] = normalized

        # Apply gamma correction to increase contrast
        rgb_array = np.power(rgb_array, gamma)

        # Apply brightness reduction
        rgb_array = rgb_array * brightness_factor

        return rgb_array

    except Exception as e:
        print(f"Error creating RGB array from bands: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def _search_planetary_computer(area_of_interest, time_of_interest, cloud_cover_limit):
    """Search Planetary Computer for Sentinel-2 images."""
    try:
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=area_of_interest,
            datetime=time_of_interest,
            query={"eo:cloud_cover": {"lt": cloud_cover_limit}},
        )

        items = search.item_collection()
        print(f"Found {len(items)} Sentinel-2 images matching criteria")
        return items

    except Exception as e:
        print(f"Error searching Planetary Computer: {str(e)}")
        return []


def _select_best_image(items):
    """Select the best image from collection based on cloud cover."""
    from pystac.extensions.eo import EOExtension as eo

    # Select the least cloudy image
    least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)

    print(
        f"Selected image {least_cloudy_item.id} from {least_cloudy_item.datetime.date()}"
        f" with {eo.ext(least_cloudy_item).cloud_cover:.2f}% cloud cover"
    )

    return least_cloudy_item
