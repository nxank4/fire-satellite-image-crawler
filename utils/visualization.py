import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path
import traceback
import numpy as np
from skimage.transform import resize
import os
from PIL import Image
import matplotlib.image as mpimg
import concurrent.futures
import multiprocessing
import time


def convert_png_to_jpg(
    directory=None,
    recursive=True,
    delete_original=False,
    dpi=200,
    jpeg_quality=85,
    pattern="*_composite.png",
    parallel=True,  # Add parallel processing
    max_workers=None,  # Auto-detect optimal number of workers
):
    """
    Convert existing PNG visualizations to JPG format to save storage space.

    Args:
        directory: Root directory to search for PNG files (default: OUTPUT_DIR/composites)
        recursive: Whether to search subdirectories recursively (default: True)
        delete_original: Whether to delete the original PNG files after conversion (default: False)
        dpi: New DPI value for the saved JPG files (default: 200)
        jpeg_quality: JPEG quality setting (0-100, higher is better quality but larger size) (default: 85)
        pattern: Filename pattern to match for conversion (default: "*_composite.png")
        parallel: Whether to use parallel processing (default: True)
        max_workers: Maximum number of worker processes (None = auto-detect)

    Returns:
        tuple: (number of files converted, total space saved in MB)
    """

    # Set default directory if not provided
    if directory is None:
        from config import OUTPUT_DIR

        directory = Path(OUTPUT_DIR) / "composites"
    else:
        directory = Path(directory)

    # Validate directory
    if not directory.exists():
        print(f"Directory {directory} does not exist")
        return 0, 0

    # Find all PNG files matching the pattern
    if recursive:
        # Use ** for recursive search in all subdirectories
        search_path = directory.glob(f"**/{pattern}")
    else:
        # Only search in the specified directory
        search_path = directory.glob(pattern)

    png_files = list(search_path)

    if not png_files:
        print(f"No PNG files matching '{pattern}' found in {directory}")
        return 0, 0

    print(f"Found {len(png_files)} PNG files to convert")

    # Determine optimal number of workers if not specified
    if max_workers is None:
        max_workers = min(
            32, multiprocessing.cpu_count() + 4
        )  # CPU count + 4 is generally optimal

    # Define the conversion function for a single file
    def convert_file(png_file):
        try:
            # Generate the JPG filename by replacing extension
            jpg_file = png_file.with_suffix(".jpg")

            # Skip if the JPG version already exists
            if jpg_file.exists():
                return (
                    0,
                    0,
                    f"JPG version of {png_file.name} already exists, skipping",
                )

            # Get original file size
            original_size = os.path.getsize(png_file)

            # Try PIL conversion (faster)
            try:
                # Open with PIL
                with Image.open(png_file) as img:
                    # Convert RGBA to RGB if needed
                    if img.mode == "RGBA":
                        # Create white background
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        # Paste using alpha as mask
                        background.paste(
                            img, mask=img.split()[3]
                        )  # 3 is the alpha channel
                        img = background
                    elif img.mode != "RGB":
                        # Convert any other mode to RGB
                        img = img.convert("RGB")

                    # Save as JPEG with optimizations
                    img.save(
                        jpg_file,
                        "JPEG",
                        quality=jpeg_quality,
                        optimize=True,
                        progressive=True,
                    )
            except Exception as e:
                # Fallback to matplotlib if PIL fails
                message = f"PIL conversion failed for {png_file.name}, trying matplotlib: {str(e)}"

                # Load image data with faster library
                img_data = mpimg.imread(png_file)

                # Convert RGBA to RGB if needed
                if len(img_data.shape) > 2 and img_data.shape[2] == 4:  # RGBA image
                    # Create RGB image with white background - vectorized for speed
                    rgb_data = (
                        np.ones(
                            (img_data.shape[0], img_data.shape[1], 3), dtype=np.uint8
                        )
                        * 255
                    )
                    # Apply alpha blending - vectorized operations
                    alpha = img_data[:, :, 3:4]
                    rgb_data = (
                        rgb_data * (1 - alpha) + img_data[:, :, :3] * alpha
                    ).astype(np.uint8)
                    img_data = rgb_data

                # Create figure without interactive display
                import matplotlib

                matplotlib.use("Agg")  # Non-interactive backend (faster)
                import matplotlib.pyplot as plt

                fig = plt.figure(
                    figsize=(img_data.shape[1] / dpi, img_data.shape[0] / dpi),
                    frameon=False,
                )
                ax = plt.Axes(fig, [0, 0, 1, 1])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(img_data)

                # Save without quality param (not supported in all versions)
                plt.savefig(
                    jpg_file,
                    dpi=dpi,
                    format="jpeg",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close(fig)

            # Get new file size and calculate savings
            new_size = os.path.getsize(jpg_file)
            saved_bytes = original_size - new_size

            # Delete original if requested
            if delete_original and jpg_file.exists():
                os.remove(png_file)
                delete_status = "and deleted original"
            else:
                delete_status = "original preserved"

            # Calculate stats for this file
            size_reduction = (saved_bytes / original_size) * 100
            message = (
                f"Converted {png_file.name}: {original_size / 1024:.1f} KB → {new_size / 1024:.1f} KB "
                f"({size_reduction:.1f}% smaller, {saved_bytes / 1024:.1f} KB saved), {delete_status}"
            )

            return (1, saved_bytes, message)

        except Exception as e:
            return (0, 0, f"Error converting {png_file.name}: {str(e)}")

    # Process files in parallel or sequentially
    files_converted = 0
    bytes_saved = 0

    if parallel and len(png_files) > 1:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(convert_file, f): f for f in png_files}

            # Process results as they complete with a progress indicator
            from tqdm import tqdm

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Converting",
            ):
                count, saved, message = future.result()
                files_converted += count
                bytes_saved += saved
                if count > 0 or "Error" in message:
                    print(message)
                results.append((count, saved, message))
    else:
        # Sequential processing with progress bar
        from tqdm import tqdm

        for png_file in tqdm(png_files, desc="Converting"):
            count, saved, message = convert_file(png_file)
            files_converted += count
            bytes_saved += saved
            print(message)

    # Print summary
    mb_saved = bytes_saved / (1024 * 1024)
    if files_converted > 0:
        print(
            f"\nConversion complete: {files_converted} files converted, {mb_saved:.2f} MB saved"
        )
        if delete_original:
            print("Original PNG files were deleted")
        else:
            print("Original PNG files were preserved")
    else:
        print("No files were converted")

    return files_converted, mb_saved


def batch_optimize_images(
    directory=None,
    recursive=True,
    target_format="jpg",
    current_format="png",
    dpi=200,  # Reduced DPI for faster processing
    jpeg_quality=85,
    delete_originals=False,
    parallel=True,  # Add parallel processing option
    max_workers=None,  # Auto-detect optimal number of workers
):
    """
    Batch optimize all visualization images by converting to JPG and reducing size.

    This function is useful for reclaiming disk space from previously generated
    visualizations.

    Args:
        directory: Root directory to search for image files (default: OUTPUT_DIR)
        recursive: Whether to search subdirectories recursively (default: True)
        target_format: Target format to convert to (default: "jpg")
        current_format: Current format to convert from (default: "png")
        dpi: New DPI value for the saved image files (default: 180)
        jpeg_quality: JPEG quality setting (0-100) (default: 85)
        delete_originals: Whether to delete the original files after conversion (default: False)
        parallel: Whether to use parallel processing (default: True)
        max_workers: Maximum number of worker processes (None = auto-detect)

    Returns:
        tuple: (number of files converted, total space saved in MB, percent saved)
    """
    from pathlib import Path
    import os
    import time

    # Set default directory if not provided
    if directory is None:
        from config import OUTPUT_DIR

        directory = Path(OUTPUT_DIR)
    else:
        directory = Path(directory)

    print(f"Starting batch optimization in {directory}")
    print(f"Converting {current_format.upper()} files to {target_format.upper()}")
    print(f"DPI: {dpi}, JPEG Quality: {jpeg_quality}")
    print(f"Delete originals: {delete_originals}")
    print(f"Parallel processing: {'enabled' if parallel else 'disabled'}")

    start_time = time.time()

    # Calculate initial total size (use faster method)
    def fast_get_dir_size(path):
        """Calculate directory size more efficiently"""
        total = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += fast_get_dir_size(entry.path)
        return total

    total_size_before = 0
    if recursive:
        total_size_before = fast_get_dir_size(directory)
    else:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file():
                    total_size_before += entry.stat().st_size

    # Convert composites
    files_converted, mb_saved = convert_png_to_jpg(
        directory=directory,
        recursive=recursive,
        delete_original=delete_originals,
        dpi=dpi,
        jpeg_quality=jpeg_quality,
        pattern=f"*_composite.{current_format}",
        parallel=parallel,
        max_workers=max_workers,
    )

    # Convert regular images
    files_converted2, mb_saved2 = convert_png_to_jpg(
        directory=directory,
        recursive=recursive,
        delete_original=delete_originals,
        dpi=dpi,
        jpeg_quality=jpeg_quality,
        pattern=f"*_visual.{current_format}",
        parallel=parallel,
        max_workers=max_workers,
    )

    # Calculate final stats
    total_files = files_converted + files_converted2
    total_mb_saved = mb_saved + mb_saved2

    # Calculate size after conversion (faster method)
    total_size_after = 0
    if recursive:
        total_size_after = fast_get_dir_size(directory)
    else:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file():
                    total_size_after += entry.stat().st_size

    # Final stats
    total_mb_before = total_size_before / (1024 * 1024)
    total_mb_after = total_size_after / (1024 * 1024)
    percent_saved = (
        ((total_size_before - total_size_after) / total_size_before) * 100
        if total_size_before > 0
        else 0
    )

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 50)
    print(f"Batch optimization complete in {elapsed_time:.1f} seconds")
    print(f"Total files processed: {total_files}")
    print(f"Storage before: {total_mb_before:.2f} MB")
    print(f"Storage after: {total_mb_after:.2f} MB")
    print(f"Total space saved: {total_mb_saved:.2f} MB ({percent_saved:.1f}%)")
    print("=" * 50)

    return total_files, total_mb_saved, percent_saved


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
    dpi=180,  # Reduced from 200 to save space
    layout=None,
    add_labels=True,
    add_timestamp=False,
    buffer_size=0.05,
    organize_by_date=True,
    include_title=True,
    add_coordinate_grid=False,
    nasa_firms_key=None,
    figsize_per_panel=(4, 3.5),  # More compact panel size
    output_format="jpg",  # Use JPG for better compression
    jpeg_quality=85,  # Good balance between quality and size
):
    """
    Create a multi-panel visualization directly from Sentinel data without saving intermediates.

    Args:
        fire_row: GeoDataFrame row with fire information
        sentinel_item: STAC item from Sentinel-2 (if None, will attempt to fetch)
        composite_types: List of composite types to include
        output_dir: Directory to save the visualization
        dpi: Resolution for the output image (default: 180)
        layout: Custom layout as (rows, cols) tuple (default: None, auto-calculated)
        add_labels: Whether to add detailed labels to each panel (default: True)
        add_timestamp: Whether to add timestamp to output filename (default: False)
        buffer_size: Size of buffer around fire point in degrees (default: 0.05)
        organize_by_date: Whether to organize outputs by date folders (default: True)
        include_title: Whether to include title in visualization (default: True)
        add_coordinate_grid: Whether to add coordinate grid to panels (default: False)
        nasa_firms_key: NASA FIRMS API key for MWIR data
        figsize_per_panel: Size of each panel in inches (default: (4, 3.5))
        output_format: File format ('jpg' or 'png')
        jpeg_quality: JPEG quality setting (0-100, higher is better quality but larger)

    Returns:
        str or None: Path to the saved visualization or None if error occurred
    """
    # Set default output directory if not provided
    if output_dir is None:
        from config import OUTPUT_DIR

        output_dir = Path(OUTPUT_DIR) / "composites"
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
        filepath = output_dir / f"{fire_id}_composite_{timestamp}.{output_format}"
    else:
        filepath = output_dir / f"{fire_id}_composite.{output_format}"
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

        # Helper function to downsize images for memory and storage optimization
        def optimize_image(img):
            # Only resize if image is very large (over 800 pixels in any dimension)
            h, w = img.shape[:2]
            if h > 800 or w > 800:
                # Calculate new size keeping aspect ratio
                max_dim = 800
                if h > w:
                    new_h = max_dim
                    new_w = int(w * max_dim / h)
                else:
                    new_w = max_dim
                    new_h = int(h * max_dim / w)

                # Resize and ensure proper type
                if len(img.shape) == 3:  # RGB
                    new_shape = (new_h, new_w, img.shape[2])
                else:  # Grayscale
                    new_shape = (new_h, new_w)

                img = resize(img, new_shape, preserve_range=True)

                # Ensure uint8 type for proper display
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)

            return img

        # RGB true color
        if "rgb" in composite_types:
            try:
                from processors.sentinel import create_rgb_array

                rgb_img = create_rgb_array(
                    sentinel_item, fire_row, buffer_size=buffer_size, auto_adjust=True
                )
                if rgb_img is not None:
                    available_images["rgb"] = optimize_image(rgb_img)
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
                    available_images["nir"] = optimize_image(nir_img)
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
                    available_images["swir"] = optimize_image(swir_img)
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
                    available_images["fire"] = optimize_image(fire_img)
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
                    available_images["mwir"] = optimize_image(mwir_img)
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
                    available_images["nbr"] = optimize_image(nbr_img)
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
                    available_images["dnbr"] = optimize_image(dnbr_img)
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

        # Add extra space for title if needed
        if include_title:
            fig_height += 0.6  # Add space for title

        fig = plt.figure(figsize=(fig_width, fig_height))

        # Create GridSpec with better spacing and centering
        top = 0.85 if include_title else 0.95
        gs = gridspec.GridSpec(
            rows,
            cols,
            figure=fig,
            wspace=0.2,  # More breathing room between columns
            hspace=0.4,  # More space for labels between rows
            top=top,
            bottom=0.05,
            left=0.05,
            right=0.95,
        )

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

                    # Set panel title with concise fontsize
                    ax.set_title(
                        panel_info.get(comp_type, {}).get("title", comp_type.upper()),
                        fontweight="bold",
                        fontsize=10,  # Smaller font to save space
                        pad=5,  # Less padding
                    )

                    # Add detailed description if requested - more compact
                    if add_labels and comp_type in panel_info:
                        ax.text(
                            0.5,
                            -0.05,  # Positioned closer to image
                            panel_info[comp_type]["description"],
                            transform=ax.transAxes,
                            ha="center",
                            fontsize=8,  # Smaller font
                            bbox=dict(
                                boxstyle="round,pad=0.3",
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

            plt.suptitle("Fire Event Analysis", fontsize=14, fontweight="bold", y=0.98)
            plt.figtext(0.5, 0.94, subtitle, ha="center", fontsize=10)  # Smaller font

            # Add data source acknowledgment - more compact
            plt.figtext(
                0.98,
                0.01,
                "Data: Sentinel-2/VIIRS",
                ha="right",
                fontsize=7,  # Smaller font
                fontstyle="italic",
            )

        if output_format.lower() in ["jpg", "jpeg"]:
            # First save with basic parameters only
            plt.savefig(
                filepath, dpi=dpi, format="jpeg", bbox_inches="tight", pad_inches=0
            )
            plt.close()

            # Then use PIL to optimize with quality
            try:
                with Image.open(filepath) as img:
                    img = img.convert("RGB")  # Ensure RGB mode
                    img.save(
                        filepath,
                        "JPEG",
                        quality=jpeg_quality,
                        optimize=True,
                        progressive=True,
                    )
                print(f"Saved optimized multi-composite visualization to {filepath}")
            except Exception as e:
                print(f"Warning: Could not optimize JPEG quality: {str(e)}")
        else:
            # For PNG or other formats, use simpler parameters
            plt.savefig(
                filepath, dpi=dpi, format="png", bbox_inches="tight", pad_inches=0
            )
            plt.close()
            print(f"Saved multi-composite visualization to {filepath}")

    except Exception as e:
        print(f"Error creating multi-composite visualization: {str(e)}")
        traceback.print_exc()
        return None
