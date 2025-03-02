"""
Fire Satellite Image Crawler - Main Module

This script processes satellite imagery for wildfire events,
creating various visualizations for analysis.
"""

import argparse
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
from tqdm import tqdm

# Import processor modules
from processors.fire_data import filter_fire_data, load_fire_data
from processors.sentinel import get_sentinel_image
from utils.visualization import create_direct_composite_visualization
from config import OUTPUT_DIR, CACHE_DIR

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")

# Load environment variables including NASA_FIRMS_KEY
load_dotenv()
nasa_firms_key = os.getenv("NASA_FIRMS_KEY")

if not nasa_firms_key:
    print(
        "WARNING: NASA_FIRMS_KEY environment variable is not set. MWIR data will not be available."
    )


def save_image(
    sentinel_item,
    fire_row,
    asset_key="visual",
    output_dir=None,
    buffer_size=0.05,
    dpi=300,
    organize_by_date=True,  # New parameter for date-based organization
):
    """
    Save an image from a Sentinel STAC item.

    Args:
        sentinel_item: STAC item from Sentinel-2
        fire_row: GeoDataFrame row with fire information
        asset_key: The key for the asset to save (default: "visual")
        output_dir: Directory to save the image
        buffer_size: Buffer size in degrees around the point
        dpi: Resolution for the output image
        organize_by_date: Whether to organize outputs by date folders (default: True)

    Returns:
        str or None: Path to the saved image or None if failed
    """
    try:
        # Set up output directory with date-based organization if requested
        output_dir = Path(output_dir)

        # Extract fire date information if organizing by date
        if organize_by_date:
            import pandas as pd

            fire_date = pd.to_datetime(fire_row.acq_date)
            fire_year = fire_date.strftime("%Y")
            fire_month = fire_date.strftime("%m")
            output_dir = output_dir / fire_year / fire_month

        # Create output directory
        output_dir.mkdir(exist_ok=True, parents=True)

        # Generate fire ID for consistent naming
        if organize_by_date:
            # Simpler ID if date is in folder structure
            fire_id = f"fire_{fire_row.geometry.y:.4f}_{fire_row.geometry.x:.4f}"
        else:
            # Include date in filename
            fire_id = f"fire_{fire_row.acq_date}_{fire_row.geometry.y:.4f}_{fire_row.geometry.x:.4f}"

        # Create output path
        output_path = output_dir / f"{fire_id}_{asset_key}.png"

        # Check if file already exists
        if output_path.exists():
            print(f"Image already exists at {output_path}")
            return str(output_path)

        # For visual assets, use create_rgb_array
        if asset_key == "visual" or asset_key in ["true-color", "true_color"]:
            from processors.sentinel import create_rgb_array

            image_array = create_rgb_array(
                sentinel_item, fire_row, buffer_size=buffer_size, auto_adjust=True
            )

        # For other specific bands, process appropriately
        elif asset_key == "nir":
            from processors.nir import create_nir_array

            image_array = create_nir_array(
                sentinel_item, fire_row, buffer_size=buffer_size
            )

        elif asset_key == "swir":
            from processors.swir import create_swir_array

            image_array = create_swir_array(
                sentinel_item, fire_row, buffer_size=buffer_size
            )

        elif asset_key == "fire_composite":
            from processors.fire_composite import create_fire_composite_array

            image_array = create_fire_composite_array(
                sentinel_item, fire_row, buffer_size=buffer_size
            )

        # For raw assets from the STAC item
        else:
            # Check if the asset exists in the STAC item
            if asset_key not in sentinel_item.assets:
                print(f"Asset key '{asset_key}' not found in STAC item")
                return None

            # Create a grayscale visualization from a single band
            import planetary_computer as pc
            import rasterio
            from rasterio import features, windows, warp
            import numpy as np

            # Get the asset URL
            asset_href = pc.sign(sentinel_item.assets[asset_key].href)

            # Create buffer around point
            point_buffer = fire_row.geometry.buffer(buffer_size)
            aoi_bounds = features.bounds(point_buffer.__geo_interface__)

            # Open the asset and read data
            with rasterio.open(asset_href) as ds:
                warped_aoi_bounds = warp.transform_bounds(
                    "epsg:4326", ds.crs, *aoi_bounds
                )
                aoi_window = windows.from_bounds(
                    *warped_aoi_bounds, transform=ds.transform
                )
                band_data = ds.read(1, window=aoi_window)

                # Normalize for visualization
                p_low = np.percentile(band_data, 2)
                p_high = np.percentile(band_data, 98)

                if p_high > p_low:
                    normalized = np.clip((band_data - p_low) / (p_high - p_low), 0, 1)
                else:
                    normalized = np.zeros_like(band_data, dtype=float)

                # Create a grayscale RGB array
                image_array = np.stack([normalized, normalized, normalized], axis=2)

        # Check if we have a valid image array
        if image_array is None:
            print(f"Failed to create image array for {fire_id} {asset_key}")
            return None

        # Save the image
        import matplotlib.pyplot as plt

        # Get image dimensions for figure size
        height, width = image_array.shape[:2]
        fig_width = width / dpi
        fig_height = height / dpi

        # Create figure and save image
        fig = plt.figure(figsize=(fig_width, fig_height), frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Display the image
        ax.imshow(image_array)

        # Save image
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        print(f"Saved {asset_key} image to {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"Error saving {asset_key} image: {str(e)}")
        traceback.print_exc()
        return None


def parse_arguments():
    """Parse command line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(
        description="Process satellite imagery for fire locations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of fire records to process in each batch",
    )
    parser.add_argument(
        "--cloud-limit",
        type=int,
        default=10,
        help="Maximum cloud cover percentage (0-100)",
    )
    parser.add_argument(
        "--search-days",
        type=int,
        default=10,
        help="Days before/after fire date to search for images",
    )
    parser.add_argument(
        "--buffer-size",
        type=float,
        default=0.05,
        help="Buffer size around fire points in degrees",
    )

    # Filtering options
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Process only a sample of N records (0 for all)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching of Sentinel images"
    )
    parser.add_argument(
        "--filter-high-confidence",
        action="store_true",
        help="Filter out high confidence fire detections",
    )
    parser.add_argument(
        "--start-idx", type=int, default=0, help="Start processing from this index"
    )
    parser.add_argument(
        "--date-range", type=str, help="Filter by date range (YYYY-MM-DD:YYYY-MM-DD)"
    )
    parser.add_argument(
        "--region",
        type=str,
        help="Filter by geographic region: 'country:COUNTRYNAME' or 'bbox:west,south,east,north'",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Base directory for output files",
    )
    parser.add_argument(
        "--create-composites",
        action="store_true",
        help="Create multi-panel composite visualizations for each fire",
    )
    parser.add_argument(
        "--nasa-firms-key",
        type=str,
        default="",
        help="NASA FIRMS API key for accessing MWIR data",
    )
    parser.add_argument(
        "--organize-by-date",
        action="store_true",
        default=True,
        help="Organize outputs in Year/Month folders (default: True)",
    )
    parser.add_argument(
        "--no-date-folders",
        action="store_false",
        dest="organize_by_date",
        help="Don't organize outputs in date-based folders",
    )

    # Check if run with arguments
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        # Use defaults if no arguments provided
        args = parser.parse_args([])
        print("Using default settings. Run with --help for customization options.")

    return args


def setup_directories(base_dir):
    """Create all required output directories."""
    directories = {
        "original": Path(base_dir) / "original",
        "visualizations": Path(base_dir) / "visualizations",
    }

    # Create directories if they don't exist
    for dir_path in directories.values():
        dir_path.mkdir(exist_ok=True, parents=True)

    # Ensure cache directory exists
    Path(CACHE_DIR).mkdir(exist_ok=True, parents=True)

    return directories


def process_fire_record(fire_row, args, dirs):
    """
    Process a single fire record, saving only the original Sentinel image and the final composite visualization.
    """
    try:
        # Get date components for organization
        fire_date = pd.to_datetime(fire_row.acq_date)
        fire_year = fire_date.strftime("%Y")
        fire_month = fire_date.strftime("%m")

        # Generate fire ID for consistent naming (using shorter format with date in folders)
        fire_id = f"fire_{fire_row.geometry.y:.4f}_{fire_row.geometry.x:.4f}"

        # Compose the path for the composite file with date folders
        vis_dir = dirs["visualizations"] / fire_year / fire_month
        vis_dir.mkdir(exist_ok=True, parents=True)
        composite_file = vis_dir / f"{fire_id}_composite.png"

        # Check if composite already exists
        if composite_file.exists():
            print(f"Composite visualization already exists for {fire_id}")
            return {"composite": str(composite_file)}, False  # Already processed

        # Get Sentinel image with custom parameters
        sentinel_item = get_sentinel_image(
            fire_row,
            buffer_size=args.buffer_size,
            cloud_cover_limit=args.cloud_limit,
            search_days=args.search_days,
            use_cache=not args.no_cache,
        )

        if not sentinel_item:
            return None, False

        results = {}

        # Save original visual image from Sentinel (raw data preservation)
        # Now using date-based organization (organize_by_date=True)
        visual_path = save_image(
            sentinel_item,
            fire_row,
            asset_key="visual",
            output_dir=dirs["original"],
            buffer_size=args.buffer_size,
            dpi=300,
            organize_by_date=True,
        )
        results["visual"] = visual_path

        # Create the composite visualization directly
        # Already uses date-based organization by default
        composite_path = create_direct_composite_visualization(
            fire_row,
            sentinel_item=sentinel_item,
            composite_types=[
                "rgb",
                "fire",
                "nir",
                "swir",
                "nbr",
                "dnbr",
            ],
            output_dir=dirs["visualizations"],
            add_timestamp=False,
            buffer_size=args.buffer_size,
            organize_by_date=True,  # Explicitly set for consistency
            nasa_firms_key=nasa_firms_key,
        )
        results["composite"] = composite_path

        # Only return success if at least one output was created
        success = bool(visual_path or composite_path)
        return results, success

    except Exception as e:
        print(f"Error processing fire record: {str(e)}")
        traceback.print_exc()
        return None, False


def process_batch(batch, args, dirs):
    """Process a batch of fire records with progress tracking."""
    processed_count = 0
    success_count = 0

    for _, fire_row in tqdm(list(batch.iterrows()), desc="Processing"):
        results, success = process_fire_record(fire_row, args, dirs)

        if success:
            success_count += 1

        if results is not None:
            processed_count += 1

    return processed_count, success_count


def main():
    """Main function to process all fire data with batching and customization options."""
    start_time = time.time()

    try:
        # Parse command line arguments
        args = parse_arguments()

        # Print selected options
        print(f"\n{'=' * 60}")
        print("Fire Satellite Image Crawler")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")
        print("Processing with options:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Cloud cover limit: {args.cloud_limit}%")
        print(f"  Search days: {args.search_days} days before/after")
        print(f"  Buffer size: {args.buffer_size} degrees")
        print(f"  Caching: {'disabled' if args.no_cache else 'enabled'}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Create composites: {'yes' if args.create_composites else 'no'}")

        # Setup directory structure
        dirs = setup_directories(args.output_dir)

        # Load fire data
        print("\nLoading fire data...")
        fires = load_fire_data()

        # Apply filters
        fires = filter_fire_data(fires, args)

        if len(fires) == 0:
            print("No fire records to process after applying filters.")
            return

        # Process fires in batches
        batch_size = args.batch_size
        total_processed = 0
        total_success = 0

        print(f"\nProcessing {len(fires)} fire records in batches of {batch_size}:")

        for batch_start in range(0, len(fires), batch_size):
            batch_end = min(batch_start + batch_size, len(fires))
            print(
                f"\nBatch {batch_start // batch_size + 1}: records {batch_start + 1}-{batch_end} of {len(fires)}"
            )

            # Get the current batch
            batch = fires.iloc[batch_start:batch_end]

            # Process the batch
            processed, success = process_batch(batch, args, dirs)
            total_processed += processed
            total_success += success

            # Print batch summary
            print(
                f"Batch {batch_start // batch_size + 1} complete: {success} successful, {batch_end - batch_start - processed} skipped"
            )

        # Calculate and display execution time
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"\n{'=' * 60}")
        print(f"Processing complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        )
        print(f"Processed {total_processed} fire records ({total_success} successful)")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"\n{'=' * 60}")
        print(
            f"\nProcessing interrupted by user at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"Elapsed time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        print("Progress up to this point has been saved.")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\nError in main processing: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
