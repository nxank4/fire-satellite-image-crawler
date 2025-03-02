import hashlib


def get_cache_key(fire_date, lat, lon, cloud_cover_limit, search_days):
    """Generate a unique cache key for a Sentinel query."""
    key_str = f"{fire_date}_{lat:.4f}_{lon:.4f}_{cloud_cover_limit}_{search_days}"
    return hashlib.md5(key_str.encode()).hexdigest()
