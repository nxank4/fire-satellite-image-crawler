# Fire Satellite Image Crawler

A Python tool that uses Google Earth Engine to find and download high-quality satellite images of fire events.

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

3. Set up Google Earth Engine authentication:
   - Visit [Google Earth Engine](https://earthengine.google.com/) and sign up
   - Install the Earth Engine command line tool: `pip install earthengine-api`
   - Authenticate with: `earthengine authenticate`

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
from gee_fire_crawler import FireSatelliteImageCrawler

# Initialize the crawler
crawler = FireSatelliteImageCrawler()

# Search for fires in California during summer 2023
california = ee.FeatureCollection('TIGER/2018/States') \
    .filter(ee.Filter.eq('NAME', 'California'))

images = crawler.search_and_download_fire_images(
    start_date="2023-06-01",
    end_date="2023-09-01",
    region=california.geometry(),
    satellite="sentinel",
    confidence="high",
    output_dir="./california_fires"
)
```

## Parameters

- `start_date`/`end_date`: Date range for fire search in 'YYYY-MM-DD' format
- `satellite`: 'sentinel' (default) or 'landsat8'
- `confidence`: Fire detection confidence - 'high' (default), 'nominal', or 'low'
- `region`: Geographic

## Troubleshooting

### Common Errors

#### 1. "Error in map" or "EEException" during image processing

This typically happens when:
- Earth Engine can't find valid satellite imagery for the detected fire points
- You're querying a very large area or time range
- There's an issue with the specific satellite data

**Solutions:**
- Try a smaller date range (e.g., 1-2 weeks instead of a month)
- Specify a more focused region rather than global search
- Try a different satellite (e.g., `--satellite landsat8` instead of sentinel)
- Ensure you're searching for past dates, not future dates

#### 2. No images downloaded

This can occur when:
- No fires were detected in your region/time range
- Fires were detected but satellite imagery was too cloudy
- The coordinates for the region are invalid

**Solutions:**
- Verify your region has active fires in the requested time period
- Increase the allowed cloud cover percentage in config.py
- Use a larger region or longer time window
- Try a different fire confidence level (e.g., `--confidence nominal`)