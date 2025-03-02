import numpy as np
import rasterio
from rasterio import windows, features, warp
from skimage.transform import resize
import traceback
import planetary_computer as pc


def create_fire_composite_array(
    item, fire_row, buffer_size=0.05, contrast_clip=(2, 98)
):
    """
    Create a false color composite array optimized for fire detection (SWIR-NIR-Red).
    Returns a numpy array suitable for direct visualization without intermediate file saving.

    Args:
        item: STAC item from Sentinel-2
        fire_row: GeoDataFrame row with fire information
        buffer_size: Buffer size in degrees around the point
        contrast_clip: Percentiles for contrast enhancement (default: (2, 98))

    Returns:
        numpy.ndarray: RGB array (height, width, 3) normalized to 0-1 range or None if failed
    """
    if (
        item is None
        or "B12" not in item.assets
        or "B08" not in item.assets
        or "B04" not in item.assets
    ):
        print("Cannot create fire composite array: Missing required bands")
        return None

    try:
        # Get the three bands for fire detection composite - apply signing to URLs
        swir_href = pc.sign(item.assets["B12"].href)  # SWIR (active fires)
        nir_href = pc.sign(item.assets["B08"].href)  # NIR (vegetation)
        red_href = pc.sign(item.assets["B04"].href)  # Red (burn scars)

        # Create buffer around point
        point_buffer = fire_row.geometry.buffer(buffer_size)
        aoi_bounds = features.bounds(point_buffer.__geo_interface__)

        # Read each band
        bands = []
        shapes = []

        # First read all bands and track their shapes
        for href in [swir_href, nir_href, red_href]:
            with rasterio.open(href) as ds:
                warped_aoi_bounds = warp.transform_bounds(
                    "epsg:4326", ds.crs, *aoi_bounds
                )
                aoi_window = windows.from_bounds(
                    *warped_aoi_bounds, transform=ds.transform
                )
                band_data = ds.read(1, window=aoi_window)

                # Check if we got valid data
                if band_data.size == 0:
                    print(
                        f"Warning: Empty band data for fire at {fire_row.geometry.y:.4f}, {fire_row.geometry.x:.4f}"
                    )
                    return None

                bands.append(band_data)
                shapes.append(band_data.shape)

        # Find the largest shape (highest resolution band)
        max_shape = max(shapes, key=lambda x: x[0] * x[1])

        # Create composite array with the highest resolution (using float for 0-1 range)
        composite = np.zeros((max_shape[0], max_shape[1], 3), dtype=np.float32)

        # Normalize and resize each band to match the target shape
        for i, band in enumerate(bands):
            # Only resize if needed
            if band.shape != max_shape:
                band = resize(band, max_shape, preserve_range=True, anti_aliasing=True)

            # Normalize to 0-1 range using percentile clipping for better contrast
            p_low, p_high = np.percentile(band, contrast_clip)

            # Avoid division by zero
            if p_high > p_low:
                normalized = np.clip((band - p_low) / (p_high - p_low), 0, 1)
            else:
                normalized = np.zeros_like(band, dtype=np.float32)

            composite[:, :, i] = normalized

        # Return the normalized array (already in 0-1 range for matplotlib)
        return composite

    except Exception as e:
        print(f"Error creating fire composite array: {str(e)}")
        traceback.print_exc()
        return None
