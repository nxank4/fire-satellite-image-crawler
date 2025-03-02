# Fire Satellite Image Crawler

This tool fetches and processes satellite imagery data related to fire events using NASA's Earth API and Google Earth Engine.

## Features

- Search for active fires using MODIS fire products
- Get high-resolution satellite imagery from Sentinel-2 or Landsat 8
- Apply specialized fire detection indices to enhance visualization
- Filter by date range, region, and fire detection confidence
- Automatic cloud cover filtering for best quality
- Batch download with progress tracking

## Setup

### Requirements

- Python 3.7 or higher
- Google Earth Engine account
- NASA API key

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/fire-satellite-image-crawler.git
   cd fire-satellite-image-crawler
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Setup environment variables:
   Create a `.env` file in the project root with:
   ```
   NASA_API_KEY=your_nasa_api_key
   PROJECT-ID=your_gee_project_id
   ```

### Google Earth Engine Authentication

Before running the script for the first time, you need to authenticate with Google Earth Engine:

## Usage

### Basic Usage

Run the main script:

```bash
python main.py --start_date 2023-01-01 --end_date 2023-01-31 --satellite sentinel
```

### Custom Region

Search in a specific region:

```bash
python main.py --start_date 2023-07-01 --end_date 2023-08-01 --region -120.5,37.7
```

### Advanced Usage

```python
from fire_crawler import FireSatelliteImageCrawler

# Initialize the crawler
crawler = FireSatelliteImageCrawler(mpc_api_key="your_api_key_here")

# Search for fires in California during summer 2023
images = crawler.search_and_download_fire_images(
    start_date="2023-06-01",
    end_date="2023-09-01",
    bbox=[-124.4, 32.5, -114.1, 42.0],  # California bounding box
    satellite="sentinel",
    confidence="high",
    output_dir="./california_fires"
)
```

## Parameters

- `start_date`/`end_date`: Date range for fire search in 'YYYY-MM-DD' format
- `satellite`: 'sentinel' (default) or 'landsat8'
- `confidence`: Fire detection confidence - 'high' (default), 'nominal', or 'low'
- `region`: Geographic region specified as 'lon,lat' (creates a 100km box around this point)
- `max_cloud_cover`: Maximum cloud cover percentage (default: 20)
- `days_after_fire`: Number of days after fire detection to look for imagery (default: 7)
- `visualization`: Visualization style - 'rgb' (natural color) or 'fire' (fire-enhanced)
- `output_dir`: Directory to save downloaded images (default: ./output)

## Microsoft Planetary Computer

This tool uses the Microsoft Planetary Computer, which provides a vast catalog of environmental data including:

- Sentinel-2 multispectral imagery
- Landsat 8 imagery
- MODIS fire products
- VIIRS fire detections
- And many more datasets

To learn more, visit the [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) website.

## Troubleshooting

### Common Errors

#### 1. No fire points found

This typically happens when:
- No fires occurred in the specified region/time range
- The date range is too narrow or in an inactive fire season
- The region is not correctly specified

**Solutions:**
- Try a wider date range 
- Try a different region
- Lower the confidence level with `--confidence nominal`

#### 2. No images downloaded

This can occur when:
- No suitable satellite imagery exists for the detected fire points
- Cloud cover is too high over all fire locations
- The satellite didn't capture the area during the specified time window

**Solutions:**
- Increase `--days_after_fire` to look for imagery in a wider time window
- Increase `--max_cloud_cover` to accept images with more cloud coverage
- Try a different satellite with `--satellite landsat8`

#### 3. Authentication errors

If you see authentication errors:
- Check that your Microsoft Planetary Computer API key is valid
- Make sure the API key is correctly specified in config.py or via --mpc_api_key

## Examples

### Find Recent Fires with Custom Parameters

```bash
python main.py \
  --start_date 2023-01-01 \
  --end_date 2023-02-01 \
  --region -120.5,37.7 \
  --satellite sentinel \
  --max_cloud_cover 30 \
  --days_after_fire 14 \
  --visualization fire
```