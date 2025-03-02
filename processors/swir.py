import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio import features, windows, warp
import planetary_computer as pc


def create_swir_array(
    item, fire_row, buffer_size=0.05, contrast_clip=(2, 98), colormap="inferno"
):
    """
    Create a SWIR (B12) image array from Sentinel-2 data without saving to disk.

    Args:
        item: STAC item from Sentinel-2
        fire_row: Row from DataFrame containing fire data
        buffer_size: Size of buffer around fire point in degrees
        contrast_clip: Percentiles for contrast enhancement
        colormap: Matplotlib colormap to use

    Returns:
        numpy.ndarray: The color-processed SWIR image array or None if processing failed
    """
    try:
        # Check if item and required band are available
        if item is None or "B12" not in item.assets:
            print("Cannot create SWIR array: item is None or B12 band not available")
            return None

        # Get SWIR band (B12 - best for fire detection)
        asset_href = pc.sign(item.assets["B12"].href)

        # Process the image data
        with rasterio.open(asset_href) as ds:
            # Create buffer and calculate bounds
            point_buffer = fire_row.geometry.buffer(buffer_size)
            aoi_bounds = features.bounds(point_buffer.__geo_interface__)
            warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)

            # Get window for the area of interest
            aoi_window = windows.from_bounds(*warped_aoi_bounds, transform=ds.transform)
            swir_data = ds.read(1, window=aoi_window)

            # Check if we got valid data
            if swir_data.size == 0:
                print(
                    f"Warning: Empty SWIR data for fire at {fire_row.geometry.y:.4f}, {fire_row.geometry.x:.4f}"
                )
                return None

        # Normalize the data with enhanced contrast for fire visibility
        min_val = np.percentile(swir_data, contrast_clip[0])
        max_val = np.percentile(swir_data, contrast_clip[1])

        # Avoid division by zero
        if max_val > min_val:
            normalized = np.clip((swir_data - min_val) / (max_val - min_val), 0, 1)
        else:
            print(
                f"Warning: Could not normalize SWIR data (min={min_val}, max={max_val})"
            )
            normalized = np.zeros_like(swir_data)

        # Create a colorized image array using a colormap
        cmap = plt.get_cmap(colormap)
        colored_image = cmap(normalized)

        # Remove alpha channel if present
        if colored_image.shape[2] == 4:
            colored_image = colored_image[:, :, :3]

        return colored_image

    except Exception as e:
        print(f"Error creating SWIR array: {str(e)}")
        import traceback

        traceback.print_exc()
        return None
