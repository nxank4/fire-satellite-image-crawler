# Fire Satellite Image Crawler

A Python tool that uses Microsoft Planetary Computer to find and download high-quality satellite images of fire events.

## Features

- Search for active fires using MODIS fire products
- Get high-resolution satellite imagery from Sentinel-2 or Landsat 8
- Apply specialized fire detection indices to enhance visualization
- Filter by date range, region, and fire detection confidence
- Automatic cloud cover filtering for best quality
- Batch download with progress tracking

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fire-satellite-image-crawler.git
cd fire-satellite-image-crawler
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Microsoft Planetary Computer authentication:
   - Sign up for an API key at [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
   - Add your API key to the `config.py` file or use the `--mpc_api_key` parameter

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