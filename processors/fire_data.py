import os
import glob
import pickle
import pandas as pd
import geopandas as gpd
from datetime import datetime
from config import CACHE_DIR


def load_fire_data(directory="fire_info", use_cache=True, force_reload=False):
    """
    Load fire data from CSV files with intelligent caching.

    Args:
        directory (str): Directory containing fire data CSV files
        use_cache (bool): Whether to use cached data if available
        force_reload (bool): Force reload from CSV files even if cache exists

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing fire data with point geometries

    Raises:
        FileNotFoundError: If directory does not exist
        ValueError: If no CSV files found in the directory
    """
    # Set up cache file path
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "fire_data.pkl")

    # Try to load from cache if allowed and not forcing reload
    if use_cache and not force_reload and os.path.exists(cache_file):
        try:
            cache_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            with open(cache_file, "rb") as f:
                fire_gdf = pickle.load(f)
                print(
                    f"Loaded {len(fire_gdf)} fire records from cache (created {cache_mod_time.strftime('%Y-%m-%d %H:%M:%S')})"
                )
                return fire_gdf
        except Exception as e:
            print(f"Error loading from cache: {str(e)}")
            # Continue to load from CSV files

    # Load from CSV files
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found.")

    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in '{directory}' directory.")

    print(f"Found {len(csv_files)} CSV files in {directory}")

    # Load and combine all CSVs with progress tracking
    dfs = []
    total_records = 0

    for i, file in enumerate(csv_files):
        try:
            df = pd.read_csv(file)

            # Validate required columns
            required_cols = ["longitude", "latitude", "acq_date"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(
                    f"Warning: File {os.path.basename(file)} missing required columns: {', '.join(missing_cols)}"
                )
                continue

            # Basic data cleaning
            df = df.dropna(subset=["longitude", "latitude", "acq_date"])

            # Standardize date format if needed
            if not pd.api.types.is_datetime64_dtype(df["acq_date"]):
                try:
                    df["acq_date"] = pd.to_datetime(df["acq_date"])
                except Exception as e:
                    print(
                        f"Warning: Could not parse dates in file {os.path.basename(file)}: {str(e)}"
                    )

            dfs.append(df)
            total_records += len(df)
            print(
                f"Processed {i + 1}/{len(csv_files)} files: {os.path.basename(file)} ({len(df)} records)"
            )

        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

    if not dfs:
        raise ValueError("No valid data found in any CSV files.")

    # Combine all dataframes
    print(f"Combining {len(dfs)} dataframes with {total_records} total records...")
    combined_df = pd.concat(dfs, ignore_index=True)

    # Convert to GeoDataFrame
    fire_gdf = gpd.GeoDataFrame(
        combined_df,
        geometry=gpd.points_from_xy(combined_df.longitude, combined_df.latitude),
        crs="EPSG:4326",
    )

    # Add some useful derived columns
    fire_gdf["year"] = fire_gdf["acq_date"].dt.year
    fire_gdf["month"] = fire_gdf["acq_date"].dt.month
    fire_gdf["day"] = fire_gdf["acq_date"].dt.day

    # Remove duplicates if any
    initial_count = len(fire_gdf)
    fire_gdf = fire_gdf.drop_duplicates(subset=["longitude", "latitude", "acq_date"])
    if len(fire_gdf) < initial_count:
        print(f"Removed {initial_count - len(fire_gdf)} duplicate records")

    # Cache the result if caching is enabled
    if use_cache:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(fire_gdf, f)
            print(f"Cached {len(fire_gdf)} fire records to {cache_file}")
        except Exception as e:
            print(f"Error caching fire data: {str(e)}")

    print(f"Successfully loaded {len(fire_gdf)} fire records")
    return fire_gdf


def filter_fire_data(fires, args):
    """Apply filters to fire data based on command line arguments."""
    total_fires = len(fires)
    filters_applied = []

    # Filter by confidence if requested
    if "confidence" in fires.columns:
        if args.filter_high_confidence:
            # Filter to keep only high confidence fires
            initial_count = len(fires)
            fires = fires[fires["confidence"].str.lower() == "h"]
            filters_applied.append(
                f"high confidence filter: {initial_count - len(fires)} records removed, keeping only high confidence ('h') fires"
            )
        elif args.filter_high_confidence:
            # Remove low confidence fires
            initial_count = len(fires)
            fires = fires[~fires["confidence"].str.lower().isin(["l", "low"])]
            filters_applied.append(
                f"low confidence filter: {initial_count - len(fires)} records removed"
            )

    # Apply date range filter if specified
    if args.date_range:
        try:
            initial_count = len(fires)
            start_date, end_date = args.date_range.split(":")
            fires = fires[
                (fires["acq_date"] >= start_date) & (fires["acq_date"] <= end_date)
            ]
            filters_applied.append(
                f"date range {start_date} to {end_date}: {initial_count - len(fires)} records removed"
            )
        except Exception as e:
            print(
                f"Error parsing date range (should be YYYY-MM-DD:YYYY-MM-DD): {str(e)}"
            )

    # Apply region filter if specified
    if args.region:
        try:
            initial_count = len(fires)
            if args.region.startswith("country:"):
                country_name = args.region.split(":", 1)[1].strip()
                # This requires having country information in the data
                if "country" in fires.columns:
                    fires = fires[fires["country"].str.lower() == country_name.lower()]
                    filters_applied.append(
                        f"country filter '{country_name}': {initial_count - len(fires)} records removed"
                    )
                else:
                    print(
                        "Warning: Country filtering requested but 'country' column not found in data"
                    )
            elif args.region.startswith("bbox:"):
                bbox_str = args.region.split(":", 1)[1].strip()
                west, south, east, north = map(float, bbox_str.split(","))
                fires = fires[
                    (fires.geometry.x >= west)
                    & (fires.geometry.x <= east)
                    & (fires.geometry.y >= south)
                    & (fires.geometry.y <= north)
                ]
                filters_applied.append(
                    f"bbox filter [{west},{south},{east},{north}]: {initial_count - len(fires)} records removed"
                )
        except Exception as e:
            print(f"Error applying region filter: {str(e)}")

    # Apply sampling if requested
    if args.sample > 0:
        initial_count = len(fires)
        sample_size = min(args.sample, len(fires))
        fires = fires.sample(
            sample_size, random_state=42
        )  # Set seed for reproducibility
        filters_applied.append(
            f"sampling: using {sample_size} of {initial_count} records"
        )

    # Skip records if start_idx is specified
    if args.start_idx > 0:
        initial_count = len(fires)
        fires = fires.iloc[args.start_idx :]
        filters_applied.append(
            f"starting from index {args.start_idx}: {initial_count - len(fires)} records skipped"
        )

    # Print summary of filters applied
    print("Data filtering summary:")
    print(f"  Starting with {total_fires} fire records")
    for filter_desc in filters_applied:
        print(f"  Applied {filter_desc}")
    print(f"  Final dataset: {len(fires)} records")

    return fires
