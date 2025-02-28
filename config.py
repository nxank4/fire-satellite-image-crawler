"""
Configuration settings for the fire satellite image crawler.
"""

# Default image collection IDs
SENTINEL_2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"  # Updated from deprecated S2_SR
LANDSAT_8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
MODIS_COLLECTION = "MODIS/061/MOD14A1"  # MODIS fire product
VIIRS_COLLECTION = "NASA/LANCE/NOAA20_VIIRS"


# Default output directory
DEFAULT_OUTPUT_DIR = "./output"

# Default image parameters
DEFAULT_IMAGE_SCALE = 30  # 30 meters per pixel
MAX_CLOUD_COVER = 20  # Maximum cloud cover percentage
DEFAULT_TIME_WINDOW_DAYS = (
    7  # Default time window to search for images after fire detection
)

# Visualization parameters
VIS_PARAMS = {
    "sentinel": {
        "rgb": {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000},
        "fire": {
            "bands": ["B12", "B8A", "B4"],
            "min": 0,
            "max": 3000,
        },  # SWIR, NIR, RED
    },
    "landsat8": {
        "rgb": {"bands": ["SR_B4", "SR_B3", "SR_B2"], "min": 0, "max": 30000},
        "fire": {
            "bands": ["SR_B7", "SR_B5", "SR_B4"],
            "min": 0,
            "max": 30000,
        },  # SWIR2, NIR, RED
    },
}
