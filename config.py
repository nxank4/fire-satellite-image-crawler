import os
from dotenv import load_dotenv

# Directory structure
OUTPUT_DIR = "fire_imagery"
VISUAL_DIR = os.path.join(OUTPUT_DIR, "original")
CACHE_DIR = "cache"

# Create all required directories
for directory in [
    OUTPUT_DIR,
    VISUAL_DIR,
    CACHE_DIR,
]:
    os.makedirs(directory, exist_ok=True)

# Default parameters
DEFAULT_BUFFER_SIZE = 0.05
DEFAULT_CLOUD_COVER_LIMIT = 10
DEFAULT_SEARCH_DAYS = 10
DEFAULT_BATCH_SIZE = 100

# FIRMS API configuration
FIRMS_SOURCES = ["VIIRS_NOAA20_NRT"]

# Load environment variables
load_dotenv()

# Get API keys from environment
FIRMS_API_KEY = os.environ.get("FIRMS_API_KEY", "")
PLANETARY_COMPUTER_API_KEY = os.environ.get("PLANETARY_COMPUTER_API_KEY", "")
