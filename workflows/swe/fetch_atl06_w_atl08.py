#!/usr/bin/env python
"""Fetch ICESat-2 ATL06 data for the LIS domain via SlideRule, with daily S3 caching."""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import argparse

import geopandas as gpd
import xarray as xr
from sliderule import icesat2
import boto3

from lis_grid import build_lis_area_definition, load_lis_grid

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# Default temporal search window
T0_DEFAULT = "2019-01-01T00:00:00Z"
T1_DEFAULT = "2019-01-07T23:59:59Z"

# S3 configuration
S3_BUCKET = "airborne-smce-prod-user-bucket"
S3_PREFIX = "JOIN/ICESAT-2/ATL06_MOSAIC"


def lis_domain_polygon(lis_grid: xr.Dataset) -> list[dict]:
    """Return the LIS domain as a SlideRule polygon (list of lon/lat dicts, CCW, closed)."""
    lat = lis_grid["lat"].values
    lon = lis_grid["lon"].values

    lat_min = float(lat.min())
    lat_max = float(lat.max())
    lon_min = float(lon.min())
    lon_max = float(lon.max())

    # Counter-clockwise, first == last
    return [
        {"lon": lon_min, "lat": lat_min},
        {"lon": lon_max, "lat": lat_min},
        {"lon": lon_max, "lat": lat_max},
        {"lon": lon_min, "lat": lat_max},
        {"lon": lon_min, "lat": lat_min},
    ]


def parse_iso_datetime(iso_str: str) -> datetime:
    """Parse ISO 8601 datetime string to datetime object."""
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))


def get_s3_path(date: datetime) -> str:
    """Generate S3 path for a given date: YYYY/MM/DD/atl06_YYYYMMDD.parquet"""
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    filename = date.strftime("atl06_%Y%m%d.parquet")
    return f"s3://{S3_BUCKET}/{S3_PREFIX}/{year}/{month}/{day}/{filename}"


def upload_to_s3(local_path: Path, s3_path: str) -> None:
    """Upload a local file to S3."""
    s3_client = boto3.client("s3")
    
    # Parse S3 path
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]
    
    log.info("Uploading %s to s3://%s/%s", local_path, bucket, key)
    s3_client.upload_file(str(local_path), bucket, key)
    log.info("Successfully uploaded to S3")


def fetch_atl06_for_day(
    polygon: list[dict],
    date: datetime,
    temp_dir: Path,
) -> gpd.GeoDataFrame | None:
    """Fetch ATL06 data for a single day from SlideRule."""
    # Set time window for this day (00:00:00 to 23:59:59 UTC)
    t0 = date.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"
    t1 = date.replace(hour=23, minute=59, second=59, microsecond=0).isoformat() + "Z"
    
    log.info("Fetching ATL06 data for %s (t0=%s, t1=%s)", date.date(), t0, t1)
    
    parms: dict = {
        "poly": polygon,
        "t0": t0,
        "t1": t1,
        "atl08_class": "atl08_ground",
        "srt": icesat2.SRT_LAND,
        "cnf": 4,
        "len": 40,
        "res": 20,
        "samples": {
            "mosaic": {
                "asset": "usgs3dep-10meter-dem",
                "radius": 10.0,
                "zonal_stats": True
            }
        }
    }
    
    try:
        gdf: gpd.GeoDataFrame = icesat2.atl06p(parms)
        
        if len(gdf) == 0:
            log.info("No ATL06 observations found for %s", date.date())
            return None
        
        log.info("SlideRule returned %d ATL06 observations for %s", len(gdf), date.date())
        
        # Extract latitude and longitude from the geometry column
        gdf['lon'] = gdf.geometry.x
        gdf['lat'] = gdf.geometry.y
        
        return gdf
    
    except Exception as e:
        log.error("Error fetching ATL06 data for %s: %s", date.date(), e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch ICESat-2 ATL06 data for the LIS domain and store daily files in S3"
    )
    parser.add_argument(
        "--lis-path",
        default="s3://airborne-smce-prod-user-bucket/JOIN/lis_input_NMP_1000m_missouri.nc",
        help="Local path or s3:// URI to the LIS NetCDF file.",
    )
    parser.add_argument(
        "--t0",
        default=T0_DEFAULT,
        help=f"Start time for SlideRule query (ISO 8601, default: {T0_DEFAULT}).",
    )
    parser.add_argument(
        "--t1",
        default=T1_DEFAULT,
        help=f"End time for SlideRule query (ISO 8601, default: {T1_DEFAULT}).",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("/tmp/icesat2_daily"),
        help="Temporary directory for storing daily parquet files before S3 upload.",
    )
    ns = parser.parse_args()

    # Load LIS grid
    lis_grid = load_lis_grid(ns.lis_path)
    
    # Build LIS bounding polygon for SlideRule query
    polygon = lis_domain_polygon(lis_grid)
    
    # Parse date range
    t0_dt = parse_iso_datetime(ns.t0)
    t1_dt = parse_iso_datetime(ns.t1)
    
    # Create temporary directory
    ns.temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize SlideRule client
    log.info("Initializing SlideRule client …")
    icesat2.init(verbose=True)
    
    # Iterate through each day in the date range
    current_date = t0_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    while current_date <= t1_dt:
        # Fetch data for this day
        gdf = fetch_atl06_for_day(polygon, current_date, ns.temp_dir)
        
        if gdf is not None and len(gdf) > 0:
            # Save to temporary local file
            temp_file = ns.temp_dir / current_date.strftime("atl06_%Y%m%d.parquet")
            gdf.to_parquet(temp_file)
            log.info("Saved daily data to %s", temp_file)
            
            # Upload to S3
            s3_path = get_s3_path(current_date)
            upload_to_s3(temp_file, s3_path)
            
            # Clean up temporary file
            temp_file.unlink()
            log.info("Cleaned up temporary file")
        
        # Move to next day
        current_date += timedelta(days=1)
    
    log.info("Completed fetching ATL06 data for date range %s to %s", ns.t0, ns.t1)


if __name__ == "__main__":
    main()