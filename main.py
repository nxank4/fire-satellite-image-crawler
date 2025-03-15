"""
MODIS imagery processing script for fire hotspot regions using Google Earth Engine
Author: lunovian
Last updated: 2025-03-15
"""

import concurrent
import ee
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os
import pandas as pd
from rich.console import Console
from rich.progress import track
import tempfile
import requests
import sys
import time
import matplotlib
import argparse
import concurrent.futures
import signal

console = Console()

# Initialize Earth Engine
try:
    ee.Initialize(project="ee-nxan2911-fire")
    print("Earth Engine initialized successfully")
except Exception as e:
    print(f"Error initializing Earth Engine: {str(e)}")
    print(
        "Make sure you have authenticated with Earth Engine (run 'earthengine authenticate' in terminal)"
    )
    sys.exit(1)


def load_fire_data(csv_path):
    """Load fire data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} fire hotspots from {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading fire data: {str(e)}")
        return None


def filter_fire_data(df, min_confidence=50, max_hotspots=10):
    """Filter fire data to get high-confidence hotspots"""
    # Filter by confidence
    filtered = df[df["confidence"] >= min_confidence]

    # Sort by brightness (descending)
    sorted_df = filtered.sort_values(by="brightness", ascending=False)

    # Take top N hotspots
    result = sorted_df.head(max_hotspots)

    print(f"Filtered to {len(result)} high-confidence fire hotspots")
    return result


def save_visualization_as_png(fig, output_dir, prefix="modis_visualization"):
    """Save matplotlib figure as PNG"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{prefix}_{timestamp}.png")
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {filepath}")
    return filepath


def download_ee_image(url, output_path, max_retries=3, retry_delay=2):
    """
    Download an image from a URL and save it to the specified path.
    Includes retry logic for resilience against temporary network issues.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)  # Added timeout
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                return True
            elif response.status_code == 429:  # Too Many Requests
                console.print(
                    "[yellow]Rate limited (429). Waiting longer before retry...[/yellow]"
                )
                time.sleep(retry_delay * 5)  # Wait longer for rate limiting
            else:
                console.print(
                    f"[yellow]Failed to download image: HTTP {response.status_code}. Attempt {attempt + 1}/{max_retries}[/yellow]"
                )
                time.sleep(retry_delay)
        except requests.RequestException as e:
            console.print(
                f"[yellow]Network error during download: {str(e)}. Attempt {attempt + 1}/{max_retries}[/yellow]"
            )
            time.sleep(retry_delay)

    console.print(
        f"[bold red]Failed to download image after {max_retries} attempts[/bold red]"
    )
    return False


def get_modis_composite(start_date, end_date, roi, satellite="Terra"):
    """Get a MODIS composite image for the specified time range, region of interest and satellite"""
    # Select the appropriate MODIS collection based on satellite
    if satellite.lower() == "aqua":
        # Aqua MODIS
        daily_collection = "MODIS/061/MYD09GQ"  # Daily 250m
        eight_day_collection = "MODIS/061/MYD09A1"  # 8-day 500m
    else:
        # Default to Terra MODIS
        daily_collection = "MODIS/061/MOD09GQ"  # Daily 250m
        eight_day_collection = "MODIS/061/MOD09A1"  # 8-day 500m

    # Try to get daily data first (higher temporal resolution)
    daily_modis = ee.ImageCollection(daily_collection).filterDate(start_date, end_date)

    # If we have daily data, use it; otherwise fall back to 8-day composite
    if daily_modis.size().getInfo() > 0:
        print(f"Using daily {satellite} MODIS data")
        composite = daily_modis.median()
    else:
        print(f"No daily data available, using 8-day {satellite} MODIS composite")
        eight_day_modis = ee.ImageCollection(eight_day_collection).filterDate(
            start_date, end_date
        )
        composite = eight_day_modis.median()

    # Clip to the region of interest
    return composite.clip(roi)


def visualize_fire_hotspot(
    row, buffer_km=100, days_before=20, output_dir="output", fast_mode=False
):
    """Generate visualization for a single fire hotspot"""
    lat = row["latitude"]
    lon = row["longitude"]
    brightness = row["brightness"]
    confidence = row["confidence"]
    acq_date = row["acq_date"]

    # Get satellite type from data (or default to Terra if not available)
    satellite = row["satellite"] if "satellite" in row else "Terra"

    # Create square region of interest
    # Convert buffer distance from km to degrees (approximately)
    # 1 degree of latitude is approximately 111 km
    buffer_deg = buffer_km / 111.0  # Convert buffer to degrees

    # Create a perfect square in meters projected around the point
    # This ensures a proper square regardless of latitude
    point = ee.Geometry.Point([lon, lat])
    square = point.buffer(buffer_km * 1000, maxError=1).bounds()

    # For visualization/reference, also calculate the approximate lat/lon bounds
    lat_buffer = buffer_deg  # degrees of latitude
    # Adjust longitude buffer based on latitude to maintain square shape
    lon_buffer = buffer_deg / np.cos(np.radians(lat))

    # Calculate lat/lon bounds for fire location overlay (smaller box)
    fire_buffer_km = 5  # Use a smaller 5km buffer to highlight the fire location
    fire_buffer_deg = fire_buffer_km / 111.0
    fire_lat_buffer = fire_buffer_deg
    fire_lon_buffer = fire_buffer_deg / np.cos(np.radians(lat))

    # Use the properly projected square for ROI
    roi = square

    # Create date range (days before to day of acquisition)
    end_date = datetime.strptime(acq_date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=days_before)

    # Format dates for Earth Engine
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Create separate folders for raw and annotated images
    raw_dir = os.path.join(output_dir, "raw_modis")
    annotated_dir = os.path.join(output_dir, "annotated")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    try:
        print(f"Processing fire hotspot at ({lat}, {lon}) from {acq_date}...")
        print(f"Using {satellite} MODIS data with {buffer_km}km square AOI")

        # Define visualization parameters based on available bands
        visualization_done = False

        # Try using MOD09GA/MYD09GA dataset first (500m resolution with good RGB bands)
        try:
            # Use higher quality MOD09GA/MYD09GA collections for better RGB composite
            if satellite.lower() == "aqua":
                ga_collection = "MODIS/061/MYD09GA"
            else:
                ga_collection = "MODIS/061/MOD09GA"

            ga_modis = ee.ImageCollection(ga_collection).filterDate(
                start_date_str, end_date_str
            )

            if ga_modis.size().getInfo() > 0:
                print(f"Using higher quality {satellite} MODIS GA product (500m)")

                # Use median composite to reduce cloud effects
                ga_composite = ga_modis.median().clip(roi)

                # For better visualization, try to stretch the image
                ga_composite = ga_composite.unitScale(
                    0, 3000
                )  # Scale values between 0-1
                ga_composite = ga_composite.multiply(
                    255
                )  # Scale to 0-255 for better display

                # Get available bands
                available_bands = ga_composite.bandNames().getInfo()
                print(f"GA product bands: {available_bands}")

                # First try to use fire-enhanced visualization
                fire_vis_params, fire_note = get_fire_enhanced_visualization(
                    ga_composite, available_bands, fast_mode
                )

                if fire_vis_params:
                    visualization_params = fire_vis_params
                    note = fire_note
                elif (
                    "sur_refl_b01" in available_bands
                    and "sur_refl_b04" in available_bands
                    and "sur_refl_b03" in available_bands
                ):
                    # True color RGB (red, green, blue)
                    visualization_params = {
                        "bands": ["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"],
                        "min": 0,
                        "max": 255,  # Now scaled to 0-255
                        "gamma": 1.2,  # Slightly lower gamma for better contrast
                    }
                    note = "True color RGB (bands 1,4,3)"
                    print("Using true color RGB visualization")
                else:
                    # False color if true RGB not available
                    first_band = available_bands[0]
                    visualization_params = {
                        "bands": [first_band, first_band, first_band],
                        "min": 0,
                        "max": 255,
                        "gamma": 1.4,
                    }
                    note = f"Grayscale using band: {first_band}"
                    print(f"Using grayscale with band: {first_band}")

                # Get thumbnail URL using better quality data - larger dimensions for better detail
                thumbnail_dimensions = 1024 if fast_mode else 2048

                # ... in the thumbnail URL generation ...
                thumbnail_url = ga_composite.getThumbURL(
                    {
                        **visualization_params,
                        "dimensions": thumbnail_dimensions,
                        "format": "jpg"
                        if fast_mode
                        else "png",  # Use JPEG for faster downloads in fast mode
                    }
                )
                visualization_done = True

                # Add info about the visualization
                note = f"{note} - {satellite} MODIS data (500m resolution, {buffer_km}km square)"

                # Save the raw composite for additional processing
                composite = ga_composite

            else:
                print(f"No {ga_collection} data available for the specified date range")

        except Exception as e:
            print(
                f"Error using enhanced MODIS data: {str(e)}. Falling back to standard products."
            )

        # Download the thumbnail to raw directory
        img_file = os.path.join(
            raw_dir,
            f"{satellite.lower()}_modis_{start_date_str}_to_{end_date_str}.png",
        )

        # After downloading and displaying the image (around line 351):
        if download_ee_image(thumbnail_url, img_file):
            print(f"Raw MODIS image saved to: {img_file}")

            # Create enhanced versions directory
            enhanced_dir = os.path.join(output_dir, "enhanced")
            os.makedirs(enhanced_dir, exist_ok=True)

            # Save additional fire-enhanced views if bands are available
            try:
                if all(
                    band in available_bands
                    for band in ["sur_refl_b07", "sur_refl_b02", "sur_refl_b01"]
                ):
                    # Fire detection view (7-2-1)
                    fire_view_url = ga_composite.getThumbURL(
                        {
                            "bands": ["sur_refl_b07", "sur_refl_b02", "sur_refl_b01"],
                            "min": 0,
                            "max": 255,
                            "gamma": 1.0,
                            "dimensions": thumbnail_dimensions,
                            "format": "jpg" if fast_mode else "png",
                        }
                    )

                    fire_view_file = os.path.join(
                        enhanced_dir,
                        f"{satellite.lower()}_fire721_{start_date_str}_to_{end_date_str}.png",
                    )
                    download_ee_image(fire_view_url, fire_view_file)
                    print(f"Fire-enhanced view (7-2-1) saved to: {fire_view_file}")

                if all(
                    band in available_bands
                    for band in ["sur_refl_b02", "sur_refl_b07", "sur_refl_b03"]
                ):
                    # Smoke detection view (2-7-3)
                    smoke_view_url = ga_composite.getThumbURL(
                        {
                            "bands": ["sur_refl_b02", "sur_refl_b07", "sur_refl_b03"],
                            "min": 0,
                            "max": 255,
                            "gamma": 1.0,
                            "dimensions": thumbnail_dimensions,
                            "format": "jpg" if fast_mode else "png",
                        }
                    )

                    smoke_view_file = os.path.join(
                        enhanced_dir,
                        f"{satellite.lower()}_smoke273_{start_date_str}_to_{end_date_str}.png",
                    )
                    download_ee_image(smoke_view_url, smoke_view_file)
                    print(f"Smoke detection view (2-7-3) saved to: {smoke_view_file}")
            except Exception as e:
                print(f"Error saving enhanced visualizations: {str(e)}")

            # Create a figure with metadata
            fig, ax = plt.subplots(figsize=(12, 10))  # Larger figure size

            # Display the image
            img = plt.imread(img_file)
            ax.imshow(img)

            # Draw a red box around the fire location (center of image)
            height, width = img.shape[:2]
            center_x, center_y = width / 2, height / 2

            # Calculate the size of the fire overlay box (proportional to the fire_buffer)
            box_size_x = (fire_lon_buffer / (lon_buffer * 2)) * width
            box_size_y = (fire_lat_buffer / (lat_buffer * 2)) * height

            # Draw the fire location box
            from matplotlib.patches import Rectangle

            fire_rect = Rectangle(
                (center_x - box_size_x, center_y - box_size_y),
                2 * box_size_x,
                2 * box_size_y,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(fire_rect)

            # Add fire highlight with yellow glow if using fire-enhanced visualization
            if "fire" in note.lower():
                # Add a semi-transparent yellow overlay to highlight the fire area
                fire_highlight = Rectangle(
                    (center_x - box_size_x * 1.2, center_y - box_size_y * 1.2),
                    2 * box_size_x * 1.2,
                    2 * box_size_y * 1.2,
                    linewidth=2,
                    edgecolor="yellow",
                    facecolor="yellow",
                    alpha=0.15,
                )
                ax.add_patch(fire_highlight)

                # Add informational text about the fire visualization
                ax.text(
                    0.5,
                    0.95,
                    "FIRE ENHANCED VIEW: Active fire appears bright in this band combination",
                    color="yellow",
                    fontweight="bold",
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    bbox=dict(facecolor="black", alpha=0.5),
                    fontsize=12,
                )

            # Add a marker at the fire center
            ax.plot(center_x, center_y, "rx", markersize=10, markeredgewidth=2)

            # Add metadata as text
            ax.set_title(f"{satellite} MODIS Fire Hotspot ({lat}, {lon})", fontsize=16)

            # Add "FIRE LOCATION" label near the box
            ax.text(
                center_x,
                center_y - box_size_y - 5,
                "FIRE LOCATION",
                color="red",
                fontweight="bold",
                ha="center",
                fontsize=10,
            )

            # Add more detailed metadata
            metadata_text = (
                f"Date: {acq_date}\n"
                f"Satellite: {satellite}\n"
                f"Brightness: {brightness:.1f}K\n"
                f"Confidence: {confidence}%\n"
                f"Period: {start_date_str} to {end_date_str}\n"
                f"Visualization: {note}\n"
                f"Buffer zone: {buffer_km}km"
            )

            ax.text(
                0.02,
                0.02,
                metadata_text,
                transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=0.8),
                fontsize=12,
            )

            # If we're using a special visualization, add explanation
            if "fire enhanced" in note.lower():
                fire_legend = (
                    "In this visualization:\n"
                    "• Bright spots: Active fire\n"
                    "• Red/pink: Recently burned areas\n"
                    "• Green: Healthy vegetation\n"
                    "• Blue/gray: Smoke plumes"
                )

                ax.text(
                    0.98,
                    0.1,
                    fire_legend,
                    transform=ax.transAxes,
                    bbox=dict(facecolor="black", alpha=0.7),
                    fontsize=10,
                    color="white",
                    ha="right",
                )

            ax.axis("off")

            # Save the annotated figure to annotated directory
            annotated_file = os.path.join(
                annotated_dir,
                f"{satellite.lower()}_annotated_{start_date_str}_to_{end_date_str}.png",
            )
            plt.savefig(annotated_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"Annotated visualization saved to: {annotated_file}")
            return annotated_file

    except Exception as e:
        print(f"Error processing hotspot at ({lat}, {lon}): {str(e)}")

    return None


def get_fire_enhanced_visualization(ga_composite, available_bands, fast_mode=False):
    """
    Create a visualization that enhances fire and smoke visibility.
    Returns visualization parameters and a description note.
    """
    # Define all the band combinations we want to try, in order of preference
    visualization_options = [
        {
            "bands": ["sur_refl_b07", "sur_refl_b02", "sur_refl_b01"],
            "description": "Fire enhanced (SWIR, NIR, Red - bands 7,2,1)",
            "note": "Using fire-enhanced visualization (SWIR, NIR, Red)",
        },
        {
            "bands": ["sur_refl_b07", "sur_refl_b06", "sur_refl_b04"],
            "description": "Fire detection (SWIR, SWIR, Green - bands 7,6,4)",
            "note": "Using alternate fire detection visualization (SWIR, SWIR, Green)",
        },
        {
            "bands": ["sur_refl_b02", "sur_refl_b07", "sur_refl_b03"],
            "description": "Smoke detection (NIR, SWIR, Blue - bands 2,7,3)",
            "note": "Using smoke detection visualization (NIR, SWIR, Blue)",
        },
    ]

    # Try each option in order
    for option in visualization_options:
        if all(band in available_bands for band in option["bands"]):
            visualization_params = {
                "bands": option["bands"],
                "min": 0,
                "max": 255,
                "gamma": 1.0,
            }
            console.print(f"[green]{option['note']}[/green]")
            return visualization_params, option["description"]

    return None, None


def create_summary_visualization(image_paths, output_dir):
    """Create a summary visualization of multiple fire hotspots"""
    if not image_paths:
        print("No images to create summary visualization")
        return None

    # Determine grid layout
    n = len(image_paths)
    cols = min(3, n)
    rows = (n + cols - 1) // cols  # Ceiling division

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])  # Make axes indexable for a single subplot
    axes = axes.flatten()  # Flatten to make indexing easier

    # Add each image to the grid
    for i, img_path in enumerate(image_paths):
        if i < len(axes):
            try:
                # Extract location from filename
                img_name = os.path.basename(os.path.dirname(img_path))

                # Load and display image
                img = plt.imread(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"Hotspot {i + 1}: {img_name}")
                axes[i].axis("off")
            except Exception as e:
                print(f"Error adding image {img_path} to summary: {str(e)}")
                axes[i].text(
                    0.5,
                    0.5,
                    "Image load error",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].axis("off")

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis("off")

    # Save the summary figure
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"fire_summary_{timestamp}.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Summary visualization saved to: {summary_path}")
    return summary_path


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Process fire hotspot data from MODIS/VIIRS satellites"
    )
    parser.add_argument(
        "--max-hotspots",
        type=int,
        default=10,
        help="Maximum number of hotspots to process. Set to 0 to process all hotspots.",
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=90,
        help="Minimum confidence threshold for hotspot detection (0-100)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data_csv/fire_archive_M-C61_589118.csv",
        help="Path to the CSV file containing fire hotspot data",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for processing (default: 4)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable fast mode (fewer retries, smaller images)",
    )
    args = parser.parse_args()

    matplotlib.use("Agg")

    # Set paths
    csv_path = os.path.join(args.csv_path)
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Load fire data
    console.print("[bold yellow]Loading fire hotspot data...[/bold yellow]")
    fire_data = load_fire_data(csv_path)
    if fire_data is None:
        console.print("[bold red]Failed to load fire data. Exiting.[/bold red]")
        return

    # Filter to get high-confidence hotspots
    console.print(
        "[bold yellow]Filtering for high-confidence fire hotspots...[/bold yellow]"
    )

    # Determine max_hotspots (use all if set to 0)
    max_hotspots = len(fire_data) if args.max_hotspots == 0 else args.max_hotspots

    # Filter the data based on command line arguments
    top_hotspots = filter_fire_data(
        fire_data, min_confidence=args.min_confidence, max_hotspots=max_hotspots
    )

    total_hotspots = len(top_hotspots)
    console.print(
        f"[bold green]Processing {total_hotspots} fire hotspots...[/bold green]"
    )

    image_paths = []

    # Define a simplified processing function for parallel execution
    def process_hotspot(row_data):
        row_index, (_, row) = row_data  # Correctly unpack the tuple from iterrows()

        # Use adaptive buffer size based on fire brightness
        brightness = row["brightness"]

        # Calculate initial buffer size
        initial_buffer = (
            min(200 + (brightness - 300) * 0.5, 300) if brightness > 300 else 200
        )

        # Use fewer options in fast mode
        if args.fast_mode:
            buffer_options = [initial_buffer]
            day_options = [20]
        else:
            buffer_options = [
                initial_buffer,
                initial_buffer * 1.5,
                initial_buffer * 0.75,
            ]
            day_options = [20, 30, 10]

        for buffer_size in buffer_options:
            for days in day_options:
                console.print(
                    f"[cyan]Hotspot {row_index + 1}/{total_hotspots}: Trying {int(buffer_size)}km buffer with {days} days lookback[/cyan]"
                )

                img_path = visualize_fire_hotspot(
                    row,
                    buffer_km=int(buffer_size),
                    days_before=days,
                    output_dir=output_dir,
                    fast_mode=args.fast_mode,
                )

                if img_path:
                    console.print(
                        f"[green]Hotspot {row_index + 1}/{total_hotspots}: Successfully visualized fire at ({row['latitude']}, {row['longitude']})[/green]"
                    )
                    return img_path

        console.print(
            f"[bold red]Hotspot {row_index + 1}/{total_hotspots}: Failed to visualize fire at ({row['latitude']}, {row['longitude']})[/bold red]"
        )
        return None

    # Process hotspots (in parallel if workers > 1)
    if args.workers > 1:
        # Setup for parallel processing
        executor = None
        should_exit = False
        futures_to_process = {}

        # Define handlers for graceful shutdown
        def signal_handler(sig, frame):
            nonlocal should_exit
            if not should_exit:
                console.print(
                    "\n[bold red]Received interrupt signal. Shutting down gracefully...[/bold red]"
                )
                console.print(
                    "[yellow]Waiting for current tasks to complete (This may take a moment)...[/yellow]"
                )
                console.print(
                    "[yellow]Press Ctrl+C again to force immediate exit (may corrupt files)[/yellow]"
                )
                should_exit = True

                # Cancel pending tasks but let running ones complete
                for future in list(futures_to_process.keys()):
                    if not future.running():
                        future.cancel()

                signal.signal(signal.SIGINT, force_exit_handler)

        def force_exit_handler(sig, frame):
            console.print(
                "\n[bold red]Forced exit requested. Terminating immediately.[/bold red]"
            )
            sys.exit(1)

        # Set up signal handlers
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            # Prepare data for processing
            task_data = [
                (i, row_tuple) for i, row_tuple in enumerate(top_hotspots.iterrows())
            ]

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers
            ) as executor:
                # Submit initial batch of tasks (limit concurrency to avoid overwhelming EE API)
                batch_size = min(args.workers * 2, len(task_data))
                initial_tasks = task_data[:batch_size]
                remaining_tasks = task_data[batch_size:]

                # Submit initial batch
                futures_to_process = {
                    executor.submit(process_hotspot, item): item
                    for item in initial_tasks
                }

                # Process results as they complete and submit new tasks
                while futures_to_process and not should_exit:
                    # Wait for the next result
                    done, not_done = concurrent.futures.wait(
                        futures_to_process,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                        timeout=1.0,  # Small timeout to check should_exit periodically
                    )

                    # Process completed futures
                    for future in done:
                        item = futures_to_process.pop(future)

                        try:
                            img_path = future.result()
                            if img_path:
                                image_paths.append(img_path)

                                # Add progress information
                                idx = item[0]
                                completed = len(image_paths)
                                console.print(
                                    f"[green]Progress: {completed}/{total_hotspots} hotspots processed "
                                    f"({completed / total_hotspots * 100:.1f}%)[/green]"
                                )

                        except Exception as e:
                            console.print(
                                f"[bold red]Error processing hotspot {item[0] + 1}: {str(e)}[/bold red]"
                            )

                        # Submit a new task if available
                        if remaining_tasks and not should_exit:
                            new_item = remaining_tasks.pop(0)
                            futures_to_process[
                                executor.submit(process_hotspot, new_item)
                            ] = new_item

                if should_exit:
                    console.print(
                        "[yellow]Graceful shutdown in progress. Saving completed work...[/yellow]"
                    )

        except KeyboardInterrupt:
            console.print(
                "\n[bold red]KeyboardInterrupt caught in outer block. Exiting...[/bold red]"
            )

        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)

    else:
        # Sequential processing with progress tracking
        try:
            for i, row_tuple in enumerate(
                track(
                    list(top_hotspots.iterrows()),
                    description="Processing hotspots",
                    total=total_hotspots,
                )
            ):
                img_path = process_hotspot((i, row_tuple))
                if img_path:
                    image_paths.append(img_path)

                    # Show occasional progress
                    if (i + 1) % 5 == 0 or i + 1 == total_hotspots:
                        console.print(
                            f"[green]Progress: {len(image_paths)}/{i + 1} successful out of {total_hotspots} total[/green]"
                        )

        except KeyboardInterrupt:
            console.print(
                "\n[bold red]Processing interrupted. Saving completed work...[/bold red]"
            )

    # Create summary visualization with whatever we have completed
    if image_paths:
        console.print("[bold green]Creating summary visualization...[/bold green]")
        try:
            create_summary_visualization(image_paths, output_dir)
        except Exception as e:
            console.print(f"[bold red]Error creating summary: {str(e)}[/bold red]")

    console.print(
        f"[bold green]Processing complete! {len(image_paths)}/{total_hotspots} hotspots successfully processed.[/bold green]"
    )


if __name__ == "__main__":
    console = Console()
    console.print(
        "[bold blue]Starting Fire Hotspot Analysis with Google Earth Engine[/bold blue]"
    )
    main()
