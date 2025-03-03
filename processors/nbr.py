import numpy as np
import rasterio
from rasterio import windows, features, warp
from skimage.transform import resize
import traceback
import planetary_computer as pc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from datetime import timedelta


def create_nbr_array(
    item,
    fire_row,
    buffer_size=0.05,
    mode="nbr",  # Options: "nbr", "dnbr", or "dnbr_classified"
    pre_post_days=30,  # Search for pre-fire image this many days before fire date
    colormap=None,  # Leave None for automatic selection based on mode
):
    """
    Create a Normalized Burn Ratio (NBR) or Differenced NBR visualization.
    NBR = (NIR - SWIR) / (NIR + SWIR) using bands B08 and B12.
    dNBR = pre-fire NBR - post-fire NBR

    Args:
        item: STAC item from Sentinel-2 (post-fire image)
        fire_row: GeoDataFrame row with fire information
        buffer_size: Buffer size in degrees around the point
        mode: Visualization mode - 'nbr', 'dnbr', or 'dnbr_classified'
        pre_post_days: Days before fire date to search for pre-fire image
        colormap: Custom colormap (None for default)

    Returns:
        numpy.ndarray: RGB array (height, width, 3) normalized to 0-1 range or None if failed
    """
    try:
        # Check required bands for NBR calculation
        if item is None or "B08" not in item.assets or "B12" not in item.assets:
            print("Cannot create NBR array: Missing required bands (B08 or B12)")
            return None

        # Default visualization params based on mode
        vis_params = {
            "nbr": {
                "colormap": "RdYlGn" if colormap is None else colormap,
                "range": (-1, 1),
                "title": "Normalized Burn Ratio (NBR)",
                "description": "Vegetation: high values; Burn scars: low values",
            },
            "dnbr": {
                "colormap": "RdYlBu_r" if colormap is None else colormap,
                "range": (-1, 1),
                "title": "Differenced NBR (dNBR)",
                "description": "High values = severe burns; Low/negative values = regrowth",
            },
            "dnbr_classified": {
                "colormap": None,  # Custom colormap below
                "range": (-0.5, 1.5),
                "title": "Burn Severity Classification",
                "description": "Based on dNBR thresholds from USGS",
            },
        }

        # Create buffer around point
        point_buffer = fire_row.geometry.buffer(buffer_size)
        aoi_bounds = features.bounds(point_buffer.__geo_interface__)

        # Calculate NBR for post-fire image (current item)
        post_nbr, post_shape = _calculate_nbr(item, aoi_bounds)
        if post_nbr is None:
            return None

        # If just NBR is requested, return the single-image NBR
        if mode == "nbr":
            return _visualize_nbr(
                post_nbr,
                title=vis_params["nbr"]["title"],
                description=vis_params["nbr"]["description"],
                colormap=vis_params["nbr"]["colormap"],
                value_range=vis_params["nbr"]["range"],
            )

        # For dNBR, we need to find a pre-fire image
        pre_item = _get_pre_fire_image(fire_row, pre_post_days, buffer_size)
        if pre_item is None:
            print("Cannot create dNBR: Unable to find suitable pre-fire image")

            # Fall back to showing just post-fire NBR
            return _visualize_nbr(
                post_nbr,
                title="Post-fire NBR (no pre-fire image found)",
                description=vis_params["nbr"]["description"],
                colormap=vis_params["nbr"]["colormap"],
                value_range=vis_params["nbr"]["range"],
            )

        # Calculate pre-fire NBR, ensuring it has the same shape as post-fire NBR
        pre_nbr, pre_shape = _calculate_nbr(
            pre_item, aoi_bounds, target_shape=post_shape
        )
        if pre_nbr is None:
            print("Failed to calculate pre-fire NBR. Falling back to post-fire NBR.")

            # Fall back to showing just post-fire NBR
            return _visualize_nbr(
                post_nbr,
                title="Post-fire NBR (pre-fire calculation failed)",
                description=vis_params["nbr"]["description"],
                colormap=vis_params["nbr"]["colormap"],
                value_range=vis_params["nbr"]["range"],
            )

        # Verify shapes are the same before calculating dNBR
        if pre_nbr.shape != post_nbr.shape:
            print(
                f"Shape mismatch: pre-NBR {pre_nbr.shape} vs post-NBR {post_nbr.shape}"
            )
            print("Resizing pre-NBR to match post-NBR")
            pre_nbr = resize(pre_nbr, post_nbr.shape, preserve_range=True)

        # Calculate dNBR
        dnbr = pre_nbr - post_nbr

        # Visualize based on mode
        if mode == "dnbr":
            # Show continuous dNBR
            return _visualize_nbr(
                dnbr,
                title=vis_params["dnbr"]["title"],
                description=vis_params["dnbr"]["description"],
                colormap=vis_params["dnbr"]["colormap"],
                value_range=vis_params["dnbr"]["range"],
            )
        elif mode == "dnbr_classified":
            # Show classified burn severity
            return _visualize_classified_dnbr(
                dnbr,
                title=vis_params["dnbr_classified"]["title"],
                description=vis_params["dnbr_classified"]["description"],
            )
        else:
            print(f"Unknown mode: {mode}. Defaulting to NBR.")
            return _visualize_nbr(
                post_nbr,
                title=vis_params["nbr"]["title"],
                description=vis_params["nbr"]["description"],
                colormap=vis_params["nbr"]["colormap"],
                value_range=vis_params["nbr"]["range"],
            )

    except Exception as e:
        print(f"Error creating NBR array: {str(e)}")
        traceback.print_exc()
        return None


def _calculate_nbr(item, aoi_bounds, target_shape=None):
    """
    Calculate Normalized Burn Ratio for a given image and area of interest.

    Args:
        item: STAC item from Sentinel-2
        aoi_bounds: Area of interest bounds
        target_shape: Optional target shape to resize result to

    Returns:
        tuple: (NBR array, shape) or (None, None) if failed
    """
    try:
        # Get the NIR and SWIR bands
        nir_href = pc.sign(item.assets["B08"].href)  # NIR
        swir_href = pc.sign(item.assets["B12"].href)  # SWIR

        # Read both bands
        nir_array = None
        swir_array = None

        with rasterio.open(nir_href) as nir_ds:
            warped_aoi_bounds = warp.transform_bounds(
                "epsg:4326", nir_ds.crs, *aoi_bounds
            )
            aoi_window = windows.from_bounds(
                *warped_aoi_bounds, transform=nir_ds.transform
            )
            nir_array = nir_ds.read(1, window=aoi_window).astype(np.float32)

        with rasterio.open(swir_href) as swir_ds:
            warped_aoi_bounds = warp.transform_bounds(
                "epsg:4326", swir_ds.crs, *aoi_bounds
            )
            aoi_window = windows.from_bounds(
                *warped_aoi_bounds, transform=swir_ds.transform
            )
            swir_array = swir_ds.read(1, window=aoi_window).astype(np.float32)

        # Validate data
        if (
            nir_array is None
            or swir_array is None
            or nir_array.size == 0
            or swir_array.size == 0
        ):
            print("Error: Empty data in NIR or SWIR band")
            return None, None

        # Resize to match if different resolutions
        if nir_array.shape != swir_array.shape:
            # Use the higher resolution (usually NIR is 10m, SWIR is 20m)
            if (
                nir_array.shape[0] * nir_array.shape[1]
                > swir_array.shape[0] * swir_array.shape[1]
            ):
                swir_array = resize(swir_array, nir_array.shape, preserve_range=True)
                current_shape = nir_array.shape
            else:
                nir_array = resize(nir_array, swir_array.shape, preserve_range=True)
                current_shape = swir_array.shape
        else:
            current_shape = nir_array.shape

        # Apply scale factors if needed
        # (Sentinel-2 L1C data needs to be divided by 10000, L2A doesn't)
        # Check if the values are very large
        if np.percentile(nir_array, 95) > 1000 or np.percentile(swir_array, 95) > 1000:
            nir_array = nir_array / 10000.0
            swir_array = swir_array / 10000.0

        # Calculate NBR
        epsilon = 1e-10  # Small number to avoid division by zero
        denominator = nir_array + swir_array + epsilon
        nbr = (nir_array - swir_array) / denominator

        # Clip to valid range
        nbr = np.clip(nbr, -1.0, 1.0)

        # Resize to target shape if provided
        if target_shape is not None and target_shape != current_shape:
            print(f"Resizing NBR from {current_shape} to {target_shape}")
            nbr = resize(nbr, target_shape, preserve_range=True)

        return nbr, current_shape

    except Exception as e:
        print(f"Error calculating NBR: {str(e)}")
        traceback.print_exc()
        return None, None


def _get_pre_fire_image(fire_row, days_before=30, buffer_size=0.05):
    """
    Get a pre-fire Sentinel-2 image from approximately the same season

    Args:
        fire_row: GeoDataFrame row with fire information
        days_before: Days before fire to look for image
        buffer_size: Buffer size in degrees

    Returns:
        pystac.Item or None: Pre-fire Sentinel-2 image
    """
    try:
        from processors.sentinel import _search_planetary_computer, _select_best_image

        # Extract location data
        fire_point = fire_row.geometry
        fire_date = pd.to_datetime(fire_row.acq_date)

        # Create buffer
        area_of_interest = fire_point.buffer(buffer_size).__geo_interface__

        # Define search window for pre-fire image (45-15 days before fire)
        date_end = (fire_date - timedelta(days=15)).strftime("%Y-%m-%d")
        date_start = (fire_date - timedelta(days=days_before)).strftime("%Y-%m-%d")

        time_of_interest = f"{date_start}/{date_end}"

        # Search the Planetary Computer catalog
        items = _search_planetary_computer(
            area_of_interest, time_of_interest, 20
        )  # Allow higher cloud cover

        if not items:
            print(
                f"No suitable pre-fire images found between {date_start} and {date_end}"
            )

            # Try with a larger window
            date_start = (fire_date - timedelta(days=days_before * 2)).strftime(
                "%Y-%m-%d"
            )
            time_of_interest = f"{date_start}/{date_end}"
            items = _search_planetary_computer(area_of_interest, time_of_interest, 30)

            if not items:
                print("No suitable pre-fire images found in extended window")
                return None

        # Select the best image
        selected_item = _select_best_image(items)

        print(f"Selected pre-fire image from {selected_item.datetime.date()}")
        return selected_item

    except Exception as e:
        print(f"Error getting pre-fire image: {str(e)}")
        traceback.print_exc()
        return None


def _visualize_nbr(
    nbr_array,
    title="Normalized Burn Ratio",
    description=None,
    colormap="RdYlGn",
    value_range=(-1, 1),
):
    """
    Create a visualization array from NBR data

    Args:
        nbr_array: NBR or dNBR array
        title: Plot title
        description: Additional description text
        colormap: Matplotlib colormap name
        value_range: Data range for normalization

    Returns:
        numpy.ndarray: RGB visualization array
    """
    try:
        # Create figure with appropriate size
        aspect_ratio = nbr_array.shape[1] / nbr_array.shape[0]
        fig = plt.figure(figsize=(10, 10 / aspect_ratio))

        # Plot the NBR array
        im = plt.imshow(
            nbr_array, cmap=colormap, vmin=value_range[0], vmax=value_range[1]
        )

        # Add colorbar
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("NBR Value")

        # Add title
        plt.title(title, fontweight="bold")

        # Add description if provided
        if description:
            plt.figtext(
                0.5,
                0.01,
                description,
                ha="center",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
            )

        # Remove axes
        plt.axis("off")

        # Render to array
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Clean up
        plt.close(fig)

        # Return normalized array
        return img_array.astype(float) / 255.0

    except Exception as e:
        print(f"Error visualizing NBR: {str(e)}")
        traceback.print_exc()
        return None


def _visualize_classified_dnbr(
    dnbr_array, title="Burn Severity Classification", description=None
):
    """
    Create a classified burn severity visualization from dNBR

    Args:
        dnbr_array: dNBR array
        title: Plot title
        description: Additional description text

    Returns:
        numpy.ndarray: RGB visualization array
    """
    try:
        # Create a classified version (USGS classification scheme)
        # See: https://un-spider.org/advisory-support/recommended-practices/recommended-practice-burn-severity/in-detail/normalized-burn-ratio

        classified = np.zeros_like(dnbr_array)

        # Enhanced regrowth, high (dNBR < -0.25)
        classified[dnbr_array < -0.25] = 0

        # Enhanced regrowth, low (-0.25 <= dNBR < -0.1)
        classified[(dnbr_array >= -0.25) & (dnbr_array < -0.1)] = 1

        # Unburned (-0.1 <= dNBR < 0.1)
        classified[(dnbr_array >= -0.1) & (dnbr_array < 0.1)] = 2

        # Low severity (0.1 <= dNBR < 0.27)
        classified[(dnbr_array >= 0.1) & (dnbr_array < 0.27)] = 3

        # Moderate-low severity (0.27 <= dNBR < 0.44)
        classified[(dnbr_array >= 0.27) & (dnbr_array < 0.44)] = 4

        # Moderate-high severity (0.44 <= dNBR < 0.66)
        classified[(dnbr_array >= 0.44) & (dnbr_array < 0.66)] = 5

        # High severity (dNBR >= 0.66)
        classified[dnbr_array >= 0.66] = 6

        # Create custom colormap for burn severity
        colors = [
            (0.0, 0.4, 0.0),  # Dark green - Enhanced regrowth, high
            (0.0, 0.8, 0.0),  # Light green - Enhanced regrowth, low
            (0.9, 0.9, 0.9),  # Light grey - Unburned
            (1.0, 1.0, 0.0),  # Yellow - Low severity
            (1.0, 0.6, 0.0),  # Orange - Moderate-low severity
            (1.0, 0.0, 0.0),  # Red - Moderate-high severity
            (0.6, 0.0, 0.0),  # Dark red - High severity
        ]

        burn_cmap = LinearSegmentedColormap.from_list("burn_severity", colors, N=7)

        # Create figure with appropriate size
        aspect_ratio = dnbr_array.shape[1] / dnbr_array.shape[0]
        fig = plt.figure(figsize=(10, 10 / aspect_ratio))

        # Plot the classified array
        plt.imshow(classified, cmap=burn_cmap, vmin=0, vmax=6)

        # Create custom legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=colors[0], label="Enhanced regrowth, high"),
            Patch(facecolor=colors[1], label="Enhanced regrowth, low"),
            Patch(facecolor=colors[2], label="Unburned"),
            Patch(facecolor=colors[3], label="Low severity"),
            Patch(facecolor=colors[4], label="Moderate-low severity"),
            Patch(facecolor=colors[5], label="Moderate-high severity"),
            Patch(facecolor=colors[6], label="High severity"),
        ]

        # Add legend
        plt.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(1.1, 1),
            fontsize="small",
        )

        # Add title
        plt.title(title, fontweight="bold")

        # Add description if provided
        if description:
            plt.figtext(
                0.5,
                0.01,
                description,
                ha="center",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
            )

        # Remove axes
        plt.axis("off")

        # Render to array
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Clean up
        plt.close(fig)

        # Return normalized array
        return img_array.astype(float) / 255.0

    except Exception as e:
        print(f"Error visualizing classified dNBR: {str(e)}")
        traceback.print_exc()
        return None
