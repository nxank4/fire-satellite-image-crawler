import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path
import traceback


def create_direct_composite_visualization(
    fire_row,
    sentinel_item=None,
    composite_types=[
        "rgb",
        "fire",
        "nir",
        "swir",
        "nbr",
        "dnbr",
    ],  # Updated default types
    output_dir=None,
    dpi=300,
    layout=None,
    add_labels=True,  # Added missing parameter
    add_timestamp=False,
    buffer_size=0.05,
    organize_by_date=True,
    include_title=True,
    add_coordinate_grid=False,
    nasa_firms_key=None,
    figsize_per_panel=(10, 15),  # New parameter for panel size
):
    """
    Create a multi-panel visualization directly from Sentinel data without saving intermediates.

    Args:
        fire_row: GeoDataFrame row with fire information
        sentinel_item: STAC item from Sentinel-2 (if None, will attempt to fetch)
        composite_types: List of composite types to include
        output_dir: Directory to save the visualization
        dpi: Resolution for the output image (default: 300)
        layout: Custom layout as (rows, cols) tuple (default: None, auto-calculated)
        add_labels: Whether to add detailed labels to each panel (default: True)
        add_timestamp: Whether to add timestamp to output filename (default: False)
        buffer_size: Size of buffer around fire point in degrees (default: 0.05)
        organize_by_date: Whether to organize outputs by date folders (default: True)
        include_title: Whether to include title in visualization (default: True)
        add_coordinate_grid: Whether to add coordinate grid to panels (default: False)
        nasa_firms_key: NASA FIRMS API key for MWIR data
        figsize_per_panel: Size of each panel in inches (default: (5, 4))

    Returns:
        str or None: Path to the saved visualization or None if error occurred
    """
    # Set default output directory if not provided
    if output_dir is None:
        from config import OUTPUT_DIR

        output_dir = (
            Path(OUTPUT_DIR) / "composites"
        )  # Changed from visual_only to composites
    else:
        output_dir = Path(output_dir)

    # Extract fire date information if organizing by date
    if organize_by_date:
        fire_date = pd.to_datetime(fire_row.acq_date)
        fire_year = fire_date.strftime("%Y")
        fire_month = fire_date.strftime("%m")
        output_dir = output_dir / fire_year / fire_month

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate fire ID
    if organize_by_date:
        # Simpler ID if date is in folder structure
        fire_id = f"fire_{fire_row.geometry.y:.4f}_{fire_row.geometry.x:.4f}"
    else:
        # Include date in filename
        fire_id = f"fire_{fire_row.acq_date}_{fire_row.geometry.y:.4f}_{fire_row.geometry.x:.4f}"

    # Add timestamp if requested
    if add_timestamp:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filepath = (
            output_dir / f"{fire_id}_composite_{timestamp}.png"
        )  # Changed from visual to composite
    else:
        filepath = (
            output_dir / f"{fire_id}_composite.png"
        )  # Changed from visual to composite
        # Check if file already exists
        if filepath.exists():
            print(f"Composite visualization already exists at {filepath}")
            return str(filepath)

    try:
        # If sentinel_item wasn't provided, get one
        if sentinel_item is None:
            from processors.sentinel import get_sentinel_image

            sentinel_item = get_sentinel_image(
                fire_row, buffer_size=buffer_size, cloud_cover_limit=20
            )
            if sentinel_item is None:
                print("Could not obtain Sentinel image for visualization")
                return None

        # Title and description mapping for each product
        panel_info = {
            "rgb": {
                "title": "RGB True Color",
                "description": "Natural color view similar to what the human eye would see",
            },
            "nir": {
                "title": "Near Infrared (B08)",
                "description": "Vegetation appears bright red, water appears black",
            },
            "swir": {
                "title": "Short-wave IR (B12)",
                "description": "Highlights active fire fronts and hot spots",
            },
            "fire": {
                "title": "Fire Detection Composite",
                "description": "Fire-optimized composite for detecting active burn areas",
            },
            "mwir": {
                "title": "Thermal MWIR",
                "description": "Shows heat signature from Middle Wave Infrared bands",
            },
            "nbr": {
                "title": "Normalized Burn Ratio",
                "description": "Vegetation health indicator: high (green) = healthy, low (red) = stressed/burned",
            },
            "dnbr": {
                "title": "Burn Severity Classification",
                "description": "Shows burn severity based on pre-fire vs post-fire comparison",
            },
        }

        # Generate images in memory
        available_images = {}

        # RGB true color
        if "rgb" in composite_types:
            try:
                from processors.sentinel import create_rgb_array

                rgb_img = create_rgb_array(
                    sentinel_item, fire_row, buffer_size=buffer_size, auto_adjust=True
                )
                if rgb_img is not None:
                    available_images["rgb"] = rgb_img
            except Exception as e:
                print(f"Error creating RGB image: {str(e)}")

        # NIR visualization
        if "nir" in composite_types:
            try:
                from processors.nir import create_nir_array

                nir_img = create_nir_array(
                    sentinel_item, fire_row, buffer_size=buffer_size
                )
                if nir_img is not None:
                    available_images["nir"] = nir_img
            except Exception as e:
                print(f"Error creating NIR image: {str(e)}")

        # SWIR visualization
        if "swir" in composite_types:
            try:
                from processors.swir import create_swir_array

                swir_img = create_swir_array(
                    sentinel_item, fire_row, buffer_size=buffer_size
                )
                if swir_img is not None:
                    available_images["swir"] = swir_img
            except Exception as e:
                print(f"Error creating SWIR image: {str(e)}")

        # Fire composite
        if "fire" in composite_types:
            try:
                from processors.fire_composite import create_fire_composite_array

                fire_img = create_fire_composite_array(
                    sentinel_item, fire_row, buffer_size=buffer_size
                )
                if fire_img is not None:
                    available_images["fire"] = fire_img
            except Exception as e:
                print(f"Error creating fire composite image: {str(e)}")

        # MWIR visualization
        if "mwir" in composite_types:
            try:
                from processors.mwir import create_mwir_array

                mwir_img = create_mwir_array(
                    fire_row,
                    nasa_firms_key=nasa_firms_key,
                    buffer_deg=buffer_size * 10,  # Use larger buffer for MWIR
                    timeout=45,  # Increase timeout for busy networks
                )
                if mwir_img is not None:
                    available_images["mwir"] = mwir_img
            except Exception as e:
                print(f"Error creating MWIR image: {str(e)}")

        # NBR visualization
        if "nbr" in composite_types:
            try:
                from processors.nbr import create_nbr_array

                nbr_img = create_nbr_array(
                    sentinel_item,
                    fire_row,
                    buffer_size=buffer_size,
                    mode="nbr",  # Just NBR from the post-fire image
                )
                if nbr_img is not None:
                    available_images["nbr"] = nbr_img
            except Exception as e:
                print(f"Error creating NBR image: {str(e)}")

        # dNBR visualization (burn severity)
        if "dnbr" in composite_types:
            try:
                from processors.nbr import create_nbr_array

                dnbr_img = create_nbr_array(
                    sentinel_item,
                    fire_row,
                    buffer_size=buffer_size,
                    mode="dnbr_classified",  # Shows classified burn severity
                    pre_post_days=30,  # Look for image ~30 days before fire
                )
                if dnbr_img is not None:
                    available_images["dnbr"] = dnbr_img
            except Exception as e:
                print(f"Error creating dNBR image: {str(e)}")

        # Get the available types that were successfully processed
        available_types = list(available_images.keys())

        if not available_types:
            print("No visualization products could be created for composite")
            return None

        # Determine optimal layout if not specified
        n_images = len(available_types)

        if layout is not None:
            rows, cols = layout
        else:
            # Auto-calculate layout based on number of images
            if n_images <= 2:
                rows, cols = 1, n_images
            elif n_images <= 4:
                rows, cols = 2, 2
            elif n_images <= 6:
                rows, cols = 2, 3
            else:
                rows, cols = 3, 3  # Maximum default layout

        # Create figure with more flexible layout using GridSpec
        # Calculate figure size based on number of panels and per-panel size
        fig_width = figsize_per_panel[0] * cols
        fig_height = figsize_per_panel[1] * rows
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Create GridSpec with better spacing
        gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.1, hspace=0.3)

        # Add each image to the visualization
        for i, comp_type in enumerate(available_types):
            if i < rows * cols:
                # Calculate row and column position
                row, col = divmod(i, cols)

                # Create subplot with specific position
                ax = fig.add_subplot(gs[row, col])

                # Display the image with proper error handling
                try:
                    # Ensure all images have transparent background if they're arrays
                    img_data = available_images[comp_type]

                    # Handle different possible image formats
                    if len(img_data.shape) == 3 and img_data.shape[2] == 3:
                        # It's an RGB array
                        ax.imshow(img_data)
                    elif len(img_data.shape) == 2:
                        # It's a grayscale array
                        ax.imshow(img_data, cmap="gray")
                    else:
                        # Assume it's an RGBA or other format
                        ax.imshow(img_data)

                    # Set panel title with increased fontsize
                    ax.set_title(
                        panel_info.get(comp_type, {}).get("title", comp_type.upper()),
                        fontweight="bold",
                        fontsize=12,  # Increased font size
                        pad=8,  # Add padding below title
                    )

                    # Add detailed description if requested
                    if add_labels and comp_type in panel_info:
                        ax.text(
                            0.5,
                            -0.08,  # Moved slightly further below the image
                            panel_info[comp_type]["description"],
                            transform=ax.transAxes,
                            ha="center",
                            fontsize=10,  # Slightly increased font size
                            bbox=dict(
                                boxstyle="round,pad=0.4",
                                fc="white",
                                ec="lightgray",
                                alpha=0.9,
                            ),
                        )

                    # Add coordinate grid if requested
                    if add_coordinate_grid:
                        ax.grid(True, alpha=0.3, linestyle="--")
                    else:
                        ax.axis("off")
                except Exception as img_error:
                    print(f"Error displaying {comp_type} image: {str(img_error)}")
                    ax.text(
                        0.5,
                        0.5,
                        f"Error displaying {comp_type} image",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.axis("off")

        # Add overall title with more detailed information if requested
        if include_title:
            fire_date_str = pd.to_datetime(fire_row.acq_date).strftime("%Y-%m-%d")
            subtitle = f"Date: {fire_date_str} | Coordinates: {fire_row.geometry.y:.4f}°N, {fire_row.geometry.x:.4f}°E"

            if hasattr(fire_row, "confidence") and not pd.isna(fire_row.confidence):
                subtitle += f" | Confidence: {fire_row.confidence}%"

            if hasattr(fire_row, "brightness") and not pd.isna(fire_row.brightness):
                subtitle += f" | Brightness: {fire_row.brightness:.1f}K"

            plt.suptitle("Fire Event Analysis", fontsize=16, fontweight="bold", y=0.98)
            plt.figtext(0.5, 0.94, subtitle, ha="center", fontsize=12)

            # Add data source acknowledgment
            plt.figtext(
                0.99,
                0.01,
                "Data source: Sentinel-2/VIIRS",
                ha="right",
                fontsize=8,
                fontstyle="italic",
            )

            # Use tight_layout with adjusted rect parameter to allow for title space
            plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        else:
            # Use standard tight_layout if no title
            plt.tight_layout()

        # Save the visualization
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved multi-composite visualization to {filepath}")
        return str(filepath)

    except Exception as e:
        print(f"Error creating multi-composite visualization: {str(e)}")
        traceback.print_exc()
        return None
