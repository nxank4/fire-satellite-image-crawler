import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import traceback
from matplotlib.colors import LinearSegmentedColormap
from dotenv import load_dotenv
import os


# Define constants
FIRMS_SOURCE = "VIIRS_NOAA20_NRT"
DEFAULT_BUFFER_DEG = 0.5  # 0.5 degrees is about 55km at equator
DEFAULT_DAYS_RANGE = 3  # Days before and after the fire date


def create_mwir(
    fire_row,
    nasa_firms_key=None,
    days_range=DEFAULT_DAYS_RANGE,
    buffer_deg=DEFAULT_BUFFER_DEG,
    timeout=30,  # Add timeout parameter
):
    """
    Process MWIR (Mid-Wave Infrared) fire data from NASA FIRMS.

    Args:
        fire_row: GeoDataFrame row with fire information
        nasa_firms_key: NASA FIRMS API key
        days_range: Number of days before and after fire date to search
        buffer_deg: Buffer size in degrees around the point
        timeout: Request timeout in seconds (default: 30)

    Returns:
        tuple: (DataFrame with fire points, metadata dict) or (None, None) if failed
    """
    try:
        # Extract fire date and location
        lat = fire_row.geometry.y
        lon = fire_row.geometry.x
        fire_date = pd.to_datetime(fire_row.acq_date)

        # Define search range
        start_date = (fire_date - pd.Timedelta(days=days_range)).strftime("%Y-%m-%d")
        end_date = (fire_date + pd.Timedelta(days=days_range)).strftime("%Y-%m-%d")

        # Use area coordinates (buffer around the point)
        west = lon - buffer_deg
        south = lat - buffer_deg
        east = lon + buffer_deg
        north = lat + buffer_deg
        area = f"{west},{south},{east},{north}"

        # Prepare metadata
        metadata = {
            "source": FIRMS_SOURCE,
            "fire_date": fire_row.acq_date,
            "query_date_range": f"{start_date} to {end_date}",
            "coordinates": (lat, lon),
            "bounds": [west, south, east, north],
            "area_deg": buffer_deg * 2,
            "area_km": buffer_deg * 111,  # Approximate km per degree
        }

        # Check if API key is provided
        if not nasa_firms_key:
            print("Error: No NASA FIRMS API key provided")
            metadata["error"] = "No API key provided"
            return None, metadata

        # Use only VIIRS_NOAA20_NRT
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{nasa_firms_key}/{FIRMS_SOURCE}/{area}/1/{start_date}/{end_date}"
        print(f"Retrieving MWIR data from {FIRMS_SOURCE}...")

        try:
            # Make the request with timeout
            response = requests.get(url, timeout=timeout)

            if response.status_code != 200:
                error_msg = f"Error fetching FIRMS data: {response.status_code}, {response.text}"
                print(error_msg)
                metadata["error"] = error_msg
                return None, metadata

            if "Invalid" in response.text or "Error" in response.text:
                error_msg = f"API error: {response.text}"
                print(error_msg)
                metadata["error"] = error_msg
                return None, metadata

            # Save to temporary file
            temp_file = f"temp_firms_{FIRMS_SOURCE}.csv"
            with open(temp_file, "w") as f:
                f.write(response.text)

            # Read the data
            try:
                df = pd.read_csv(temp_file)
                if len(df) > 0:
                    print(
                        f"Successfully retrieved {len(df)} fire points from {FIRMS_SOURCE}"
                    )
                    metadata["points_found"] = len(df)
                    metadata["has_data"] = True

                    # Store basic statistics if frp column exists
                    if "frp" in df.columns:
                        metadata["min_frp"] = float(df["frp"].min())
                        metadata["max_frp"] = float(df["frp"].max())
                        metadata["mean_frp"] = float(df["frp"].mean())
                        metadata["total_frp"] = float(df["frp"].sum())

                    return df, metadata
                else:
                    print("No fire detections found for this area and time range")
                    metadata["points_found"] = 0
                    metadata["has_data"] = False
                    return df, metadata

            except Exception as read_err:
                print(f"Error reading FIRMS data: {str(read_err)}")
                metadata["error"] = f"Data parsing error: {str(read_err)}"
                return None, metadata

        except requests.exceptions.Timeout:
            error_msg = "Connection to NASA FIRMS API timed out. The server might be busy or your connection is slow."
            print(error_msg)
            metadata["error"] = error_msg
            return None, metadata

        except requests.exceptions.ConnectionError:
            error_msg = "Connection error when accessing NASA FIRMS API. Please check your internet connection."
            print(error_msg)
            metadata["error"] = error_msg
            return None, metadata

        except requests.exceptions.RequestException as e:
            error_msg = f"Request error when accessing NASA FIRMS API: {str(e)}"
            print(error_msg)
            metadata["error"] = error_msg
            return None, metadata

    except Exception as e:
        print(f"Error processing MWIR data: {str(e)}")
        traceback.print_exc()
        return None, None


def create_mwir_array(
    fire_row,
    nasa_firms_key=None,
    days_range=DEFAULT_DAYS_RANGE,
    buffer_deg=DEFAULT_BUFFER_DEG,
    use_visual_background=False,
    visual_array=None,
    figsize=(10, 8),
    dpi=100,
    timeout=30,  # Add timeout parameter
):
    """
    Create an MWIR visualization array for direct use in composite visualizations.

    Args:
        fire_row: GeoDataFrame row with fire information
        nasa_firms_key: NASA FIRMS API key
        days_range: Number of days before and after fire date to search
        buffer_deg: Buffer size in degrees around the point
        use_visual_background: Whether to use RGB visual image as background
        visual_array: Optional preloaded RGB array to use as background
        figsize: Figure size in inches (will determine output array aspect ratio)
        dpi: DPI for the output array (determines resolution)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        numpy.ndarray: RGB array (height, width, 3) normalized to 0-1 range or None if failed
    """
    try:
        # Extract location for plotting
        lat = fire_row.geometry.y
        lon = fire_row.geometry.x

        # Define plot bounds
        west = lon - buffer_deg
        south = lat - buffer_deg
        east = lon + buffer_deg
        north = lat + buffer_deg

        # Get MWIR data
        df, metadata = create_mwir(
            fire_row,
            nasa_firms_key=nasa_firms_key,
            days_range=days_range,
            buffer_deg=buffer_deg,
            timeout=timeout,  # Pass timeout
        )

        # Check if we got an API error
        api_error = None
        if metadata and "error" in metadata:
            api_error = metadata["error"]
            if "Invalid MAP_KEY" in api_error:
                api_error = (
                    "Invalid NASA FIRMS API key. Please check your configuration."
                )
            elif "timed out" in api_error:
                api_error = "Connection to NASA FIRMS API timed out.\nServer might be busy or network issues."
            elif "Connection error" in api_error:
                api_error = (
                    "Network connection issue.\nPlease check your internet connection."
                )

        # Create figure with off-screen rendering
        fig = plt.figure(figsize=figsize)

        # Create background
        if use_visual_background and visual_array is not None:
            plt.imshow(visual_array, extent=[west, east, south, north])
            visual_used = True
        else:
            # Use a light gray background
            plt.gca().set_facecolor("#f0f0f0")
            visual_used = False

        # If we have an API error, display it
        if api_error:
            # Create central error message
            plt.text(
                0.5,
                0.5,
                f"NASA FIRMS API Error:\n{api_error}",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            )

            plt.title(
                "FIRMS Fire Detections - Error",
                fontweight="bold",
            )

        # Otherwise proceed with normal processing
        else:
            # Get fire date range for title
            start_date = (
                metadata["query_date_range"].split(" to ")[0]
                if metadata and "query_date_range" in metadata
                else "?"
            )
            end_date = (
                metadata["query_date_range"].split(" to ")[1]
                if metadata and "query_date_range" in metadata
                else "?"
            )

            # Handle case with no data
            if df is None or len(df) == 0:
                plt.annotate(
                    "No active fire detections in this period",
                    xy=(0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                )
            else:
                # Find column names - different FIRMS products have different column names
                lon_col = next(
                    (col for col in ["longitude", "lon"] if col in df.columns), None
                )
                lat_col = next(
                    (col for col in ["latitude", "lat"] if col in df.columns), None
                )
                frp_col = next(
                    (col for col in ["frp", "FRP"] if col in df.columns), None
                )

                if not lon_col or not lat_col:
                    plt.text(
                        0.5,
                        0.5,
                        "Error: Could not identify coordinate columns",
                        ha="center",
                        va="center",
                        fontsize=14,
                        color="red",
                    )
                else:
                    # Plot fire points
                    if frp_col in df.columns:
                        # Define custom colormap for fire intensity
                        fire_colors = LinearSegmentedColormap.from_list(
                            "fire_colors", ["red", "orange", "yellow", "white"], N=256
                        )

                        # Size and color by Fire Radiative Power
                        size = df[frp_col].values * 5
                        sc = plt.scatter(
                            df[lon_col],
                            df[lat_col],
                            s=size,
                            c=df[frp_col],
                            cmap=fire_colors,
                            marker="o",
                            alpha=0.7,
                            edgecolor="k",
                            linewidth=0.5,
                        )

                        # Add colorbar
                        cbar = plt.colorbar(sc, shrink=0.7)
                        cbar.set_label("Fire Radiative Power (MW)")
                    else:
                        # Use fixed size if no FRP data
                        plt.scatter(
                            df[lon_col],
                            df[lat_col],
                            s=50,
                            c="red",
                            marker="o",
                            alpha=0.7,
                            edgecolor="k",
                        )

            # Mark the original fire location
            plt.scatter([lon], [lat], c="blue", marker="x", s=100, linewidth=2)

            # Add labels
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")

            # Add title with white background if on visual image
            title_bg = (
                dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
                if visual_used
                else None
            )
            plt.title(
                f"FIRMS Fire Detections ({FIRMS_SOURCE})\n{start_date} to {end_date}",
                bbox=title_bg,
            )

        # Set plot limits
        plt.xlim(west, east)
        plt.ylim(south, north)

        # Add grid if not using visual background
        if not visual_used:
            plt.grid(alpha=0.3)

        # Render figure to numpy array
        fig.canvas.draw()

        # Convert canvas to numpy array
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Normalize to 0-1 range for matplotlib
        image_normalized = image.astype(float) / 255.0

        plt.close()

        return image_normalized

    except Exception as e:
        print(f"Error creating MWIR array: {str(e)}")
        traceback.print_exc()

        # Create error image
        fig = plt.figure(figsize=figsize)
        plt.text(
            0.5,
            0.5,
            f"Error processing MWIR data:\n{str(e)}",
            ha="center",
            va="center",
            fontsize=12,
            color="red",
            wrap=True,
        )
        plt.axis("off")

        # Render error message to array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return image.astype(float) / 255.0
