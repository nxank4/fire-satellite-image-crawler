import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
import multiprocessing as mp
from functools import partial

# Suppress specific matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Define the mode names as in your example
modename = [
    "all_bands",  # 0
    "all_bands_aerosol",  # 1
    "rgb",  # 2
    "rgb_aerosol",  # 3
    "swir",  # 4
    "swir_aerosol",  # 5
    "nbr",  # 6
    "nbr_aerosol",  # 7
    "ndvi",  # 8
    "ndvi_aerosol",  # 9
    "rgb_swir_nbr_ndvi",  # 10
    "rgb_swir_nbr_ndvi_aerosol",
]  # 11

# Max values for normalization as provided in your example
Max_values = np.array(
    [
        4.58500000e03,
        8.96800000e03,
        1.03440000e04,
        1.01840000e04,
        1.67280000e04,
        1.65260000e04,
        1.63650000e04,
        1.31360000e04,
        8.61900000e03,
        6.21700000e03,
        1.56140000e04,
        1.55500000e04,
        7.12807226e00,
    ]
)


# Functions for creating different composites as per your example
def create_rgb_composite(patch_data):
    rgb_patch = patch_data[[3, 2, 1], :, :]
    return rgb_patch


def create_rgb_aerosol_composite(patch_data):
    rgb_aerosol_patch = patch_data[[3, 2, 1, -1], :, :]
    return rgb_aerosol_patch


def create_swir_composite(patch_data):
    swir_composite = patch_data[[11, 7, 3], :, :]
    return swir_composite


def create_swir_aerosol_composite(patch_data):
    swir_aerosol_composite = patch_data[[11, 7, 3, -1], :, :]
    return swir_aerosol_composite


def create_nbr_composite(patch_data):
    nbr = (patch_data[7, :, :] - patch_data[11, :, :]) / (
        patch_data[7, :, :] + patch_data[11, :, :] + 1e-10
    )
    nbr_composite = np.stack((nbr, patch_data[3, :, :], patch_data[2, :, :]), axis=0)
    return nbr_composite


def create_nbr_aerosol_composite(patch_data):
    nbr = (patch_data[7, :, :] - patch_data[11, :, :]) / (
        patch_data[7, :, :] + patch_data[11, :, :] + 1e-10
    )
    nbr_aerosol_composite = np.stack(
        (nbr, patch_data[3, :, :], patch_data[2, :, :], patch_data[-1, :, :]), axis=0
    )
    return nbr_aerosol_composite


def create_ndvi_composite(patch_data):
    ndvi = (patch_data[7, :, :] - patch_data[3, :, :]) / (
        patch_data[7, :, :] + patch_data[3, :, :] + 1e-10
    )
    ndvi_composite = np.stack((ndvi, patch_data[3, :, :], patch_data[2, :, :]), axis=0)
    return ndvi_composite


def create_ndvi_aerosol_composite(patch_data):
    ndvi = (patch_data[7, :, :] - patch_data[3, :, :]) / (
        patch_data[7, :, :] + patch_data[3, :, :] + 1e-10
    )
    ndvi_aerosol_composite = np.stack(
        (ndvi, patch_data[3, :, :], patch_data[2, :, :], patch_data[-1, :, :]), axis=0
    )
    return ndvi_aerosol_composite


def create_rgb_swir_nbr_ndvi_composite(patch_data):
    nbr = (patch_data[7, :, :] - patch_data[11, :, :]) / (
        patch_data[7, :, :] + patch_data[11, :, :] + 1e-10
    )
    ndvi = (patch_data[7, :, :] - patch_data[3, :, :]) / (
        patch_data[7, :, :] + patch_data[3, :, :] + 1e-10
    )
    rgb_swir_nbr_ndvi_composite = np.stack(
        (
            patch_data[3, :, :],
            patch_data[2, :, :],
            patch_data[1, :, :],
            patch_data[11, :, :],
            nbr,
            ndvi,
        ),
        axis=0,
    )
    return rgb_swir_nbr_ndvi_composite


def create_rgb_swir_nbr_ndvi_aerosol_composite(patch_data):
    nbr = (patch_data[7, :, :] - patch_data[11, :, :]) / (
        patch_data[7, :, :] + patch_data[11, :, :] + 1e-10
    )
    ndvi = (patch_data[7, :, :] - patch_data[3, :, :]) / (
        patch_data[7, :, :] + patch_data[3, :, :] + 1e-10
    )
    rgb_swir_nbr_ndvi_aerosol_composite = np.stack(
        (
            patch_data[3, :, :],
            patch_data[2, :, :],
            patch_data[1, :, :],
            patch_data[11, :, :],
            nbr,
            ndvi,
            patch_data[-1, :, :],
        ),
        axis=0,
    )
    return rgb_swir_nbr_ndvi_aerosol_composite


def save_composite_image(data, output_path, label=None, clip_percentile=98):
    """
    Save composite data as PNG image with proper color representation.
    If label is provided, saves both original and label-overlaid versions.
    Label values: 1 for fire (red), -1 for no fire (green)

    Args:
        data: numpy array with shape [channels, height, width]
        output_path: path to save the image
        label: optional label data for overlay
        clip_percentile: percentile for clipping values (for better visualization)
    """
    # Create base directory structure
    output_path = Path(output_path)
    orig_dir = output_path.parent / "original"
    overlay_dir = output_path.parent / "overlay"
    orig_dir.mkdir(parents=True, exist_ok=True)
    if label is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the base image data
    if data.shape[0] >= 3:
        rgb_data = data[:3, :, :].copy()
        rgb_data = np.nan_to_num(rgb_data, nan=0.0, posinf=1.0, neginf=0.0)
        rgb_data = np.transpose(rgb_data, (1, 2, 0))

        # Normalize RGB data
        for i in range(rgb_data.shape[2]):
            p_low = np.percentile(rgb_data[:, :, i], 2)
            p_high = np.percentile(rgb_data[:, :, i], clip_percentile)
            rgb_data[:, :, i] = np.clip(rgb_data[:, :, i], p_low, p_high)
            channel_min = rgb_data[:, :, i].min()
            channel_max = rgb_data[:, :, i].max()
            if channel_max > channel_min:
                rgb_data[:, :, i] = (rgb_data[:, :, i] - channel_min) / (
                    channel_max - channel_min
                )
    else:
        rgb_data = data[0, :, :]
        rgb_data = np.nan_to_num(rgb_data, nan=0.0)
        p_low = np.percentile(rgb_data, 2)
        p_high = np.percentile(rgb_data, clip_percentile)
        rgb_data = np.clip(rgb_data, p_low, p_high)
        img_min, img_max = rgb_data.min(), rgb_data.max()
        if img_max > img_min:
            rgb_data = (rgb_data - img_min) / (img_max - img_min)

    # Save original image
    plt.figure(figsize=(10, 10))
    if data.shape[0] >= 3:
        plt.imshow(rgb_data)
    else:
        plt.imshow(rgb_data, cmap="viridis")
    plt.axis("off")
    plt.savefig(orig_dir / output_path.name, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    # Create and save overlay if label is provided
    if label is not None:
        plt.figure(figsize=(10, 10))
        if data.shape[0] >= 3:
            plt.imshow(rgb_data)
        else:
            plt.imshow(rgb_data, cmap="viridis")

        # Create separate masks for fire and no-fire
        fire_mask = np.ma.masked_where(label != 1, label)
        no_fire_mask = np.ma.masked_where(label != -1, label)

        # Overlay fire (red) and no-fire (green) with transparency
        plt.imshow(fire_mask, cmap=plt.cm.colors.ListedColormap(["red"]), alpha=0.3)
        plt.imshow(
            no_fire_mask, cmap=plt.cm.colors.ListedColormap(["green"]), alpha=0.3
        )

        plt.axis("off")
        plt.savefig(
            overlay_dir / output_path.name, bbox_inches="tight", pad_inches=0, dpi=300
        )
        plt.close()


def process_npz_file(npz_path, output_folder, mode, normalize=True, clip_percentile=98):
    """
    Process a single NPZ file according to the specified mode and save as PNG
    Only processes files that contain fire pixels (label=1)
    """
    try:
        # Load NPZ data
        npz_data = np.load(npz_path)

        # Check if the file contains expected keys
        if "image" not in npz_data or "aerosol" not in npz_data:
            print(
                f"Warning: {npz_path} does not contain expected keys. Available keys: {list(npz_data.keys())}"
            )
            return False

        # Check if file has fire pixels before processing
        label = npz_data.get("label", None)
        if label is None or not np.any(label == 1):
            return False  # Skip files without fire pixels

        # Get the image data and convert to reflectance
        image = npz_data["image"].astype(np.float32) / 10000.0
        aerosol = npz_data["aerosol"]

        # Get filename without extension for output naming
        base_name = Path(npz_path).stem

        # Process data according to selected mode
        if mode == 0:
            # All bands - select bands 3,2,1 for RGB visualization
            data = image
            if data.shape[0] >= 3:
                # Reorder the first 3 bands as RGB for visualization
                viz_data = (
                    image[[3, 2, 1], :, :] if image.shape[0] > 3 else image[:3, :, :]
                )
            else:
                viz_data = image
            composite_name = "all_bands"
        else:
            # Combine image and aerosol
            data = np.concatenate([image, aerosol[np.newaxis, ...]], axis=0)

            # Apply normalization if requested
            if normalize and data.size > 0:
                # Since we already divided by 10000 above, adjust the Max_values
                adjusted_max_values = Max_values / 10000.0

                # Only normalize if the data and max_values shapes match
                if data.shape[0] <= len(adjusted_max_values):
                    norm_factors = adjusted_max_values[
                        : data.shape[0], np.newaxis, np.newaxis
                    ]
                    data = np.divide(
                        data,
                        norm_factors,
                        out=np.zeros_like(data),
                        where=norm_factors != 0,
                    )

            # Create composite based on mode
            if mode == 1:
                composite_name = "all_bands_aerosol"
                viz_data = data
            elif mode == 2:
                viz_data = create_rgb_composite(data)
                composite_name = "rgb"
            elif mode == 3:
                viz_data = create_rgb_aerosol_composite(data)
                composite_name = "rgb_aerosol"
            elif mode == 4:
                viz_data = create_swir_composite(data)
                composite_name = "swir"
            elif mode == 5:
                viz_data = create_swir_aerosol_composite(data)
                composite_name = "swir_aerosol"
            elif mode == 6:
                viz_data = create_nbr_composite(data)
                composite_name = "nbr"
            elif mode == 7:
                viz_data = create_nbr_aerosol_composite(data)
                composite_name = "nbr_aerosol"
            elif mode == 8:
                viz_data = create_ndvi_composite(data)
                composite_name = "ndvi"
            elif mode == 9:
                viz_data = create_ndvi_aerosol_composite(data)
                composite_name = "ndvi_aerosol"
            elif mode == 10:
                viz_data = create_rgb_swir_nbr_ndvi_composite(data)
                composite_name = "rgb_swir_nbr_ndvi"
            elif mode == 11:
                viz_data = create_rgb_swir_nbr_ndvi_aerosol_composite(data)
                composite_name = "rgb_swir_nbr_ndvi_aerosol"
            else:
                print(f"Unsupported mode: {mode}")
                return False

        # Save the composite image
        output_path = output_folder / f"{base_name}_{composite_name}.png"

        # Get label if it exists
        label = npz_data.get("label", None)

        # Save both original and overlay versions
        save_composite_image(
            viz_data, output_path, label=label, clip_percentile=clip_percentile
        )

        return True

    except Exception as e:
        print(f"Error processing {npz_path}: {str(e)}")
        return False


def worker(args):
    """Worker function for multiprocessing"""
    npz_file, output_folder, mode, normalize, clip_percentile = args
    try:
        success = process_npz_file(
            npz_file, output_folder, mode, normalize, clip_percentile
        )
        return (1, 0) if success else (0, 1)
    except Exception as e:
        print(f"\nError processing {npz_file.name}: {str(e)}")
        return (0, 1)


def process_npz_folder(
    input_folder, output_base=None, mode=2, normalize=True, clip_percentile=98
):
    """
    Recursively process .npz files in the specified folder and its subfolders using multiprocessing
    Only processes files containing fire pixels
    """
    folder_path = Path(input_folder)
    print(f"Processing files in mode: {modename[mode]}")

    npz_files = list(folder_path.rglob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {folder_path}")
        return

    print(f"Found {len(npz_files)} .npz files, checking for fire pixels...")

    if output_base is None:
        output_base = folder_path
    else:
        output_base = Path(output_base)

    output_folder = output_base / f"png_output_{modename[mode]}"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for multiprocessing
    num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    print(f"Using {num_cores} CPU cores")

    # Create argument tuples for each file
    work_args = [
        (npz_file, output_folder, mode, normalize, clip_percentile)
        for npz_file in npz_files
    ]

    # Process files in parallel with progress bar
    successful = 0
    failed = 0

    with mp.Pool(num_cores) as pool:
        results = list(
            tqdm(
                pool.imap(worker, work_args),
                total=len(npz_files),
                desc="Converting files",
            )
        )

    # Sum up results
    successful = sum(r[0] for r in results)
    failed = sum(r[1] for r in results)

    # Print summary
    print("\nConversion complete:")
    print(f"- Successfully converted: {successful} files")
    print(f"- Failed: {failed} files")
    print(f"- Total: {len(npz_files)} files")
    print(f"Output directory: {output_folder}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Convert .npz files to PNG images with various processing modes."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to folder containing .npz files (will search recursively)",
    )
    parser.add_argument(
        "--output", type=str, help="Base output folder (optional)", default=None
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=range(12),
        default=2,
        help="Processing mode (0-11)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Disable normalization",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=98,
        help="Percentile for clipping values (default: 98)",
    )

    args = parser.parse_args()

    # Process the folder
    process_npz_folder(
        args.input_folder, args.output, args.mode, args.normalize, args.clip
    )


if __name__ == "__main__":
    main()
