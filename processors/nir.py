import numpy as np
import planetary_computer as pc
import rasterio
from rasterio import windows, features, warp
import matplotlib.pyplot as plt


def create_nir_array(
    item, fire_row, buffer_size=0.05, contrast_clip=(2, 98), colormap="RdYlGn_r"
):
    """
    Create a colorized NIR (Near Infrared) visualization array from Sentinel-2 Band 8.
    Returns a visualization-ready array without saving to disk.

    Args:
        item: STAC item from Sentinel-2
        fire_row: GeoDataFrame row with fire information
        buffer_size: Buffer size in degrees around the point
        contrast_clip: Percentiles for contrast enhancement (default: (2, 98))
        colormap: Matplotlib colormap to use for NIR visualization (default: 'RdYlGn_r')

    Returns:
        numpy.ndarray: RGB array (height, width, 3) normalized to 0-1 range or None if failed
    """
    if item is None or "B08" not in item.assets:
        print("Cannot create NIR array: STAC item is None or B08 band not available")
        return None

    try:
        # Get the NIR band with signed URL
        asset_href = pc.sign(item.assets["B08"].href)

        # Create buffer around point
        point_buffer = fire_row.geometry.buffer(buffer_size)
        aoi_bounds = features.bounds(point_buffer.__geo_interface__)

        # Read the NIR data
        with rasterio.open(asset_href) as ds:
            warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
            aoi_window = windows.from_bounds(*warped_aoi_bounds, transform=ds.transform)
            nir_data = ds.read(1, window=aoi_window)

            # Check if we got valid data
            if nir_data.size == 0:
                print(
                    f"Warning: Empty NIR data for fire at {fire_row.geometry.y:.4f}, {fire_row.geometry.x:.4f}"
                )
                return None

        # Calculate statistics for normalization
        p_low = np.percentile(nir_data, contrast_clip[0])
        p_high = np.percentile(nir_data, contrast_clip[1])

        # Normalize the data to 0-1 range
        if p_high > p_low:
            normalized = np.clip((nir_data - p_low) / (p_high - p_low), 0, 1)
        else:
            print(f"Warning: Could not normalize NIR data (min={p_low}, max={p_high})")
            normalized = np.zeros_like(nir_data, dtype=float)

        # Apply colormap to create RGB visualization
        # This converts the single-band NIR to a 3-channel RGB visualization
        cmap = plt.get_cmap(colormap)
        colored_image = cmap(normalized)

        # Remove the alpha channel if present
        if colored_image.shape[2] == 4:
            colored_image = colored_image[:, :, :3]

        return colored_image

    except Exception as e:
        print(f"Error creating NIR array: {str(e)}")
        import traceback

        traceback.print_exc()
        return None
