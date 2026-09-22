#!/usr/bin/env python
"""Fetch ICESat-2 ATL06 data for the LIS domain via SlideRule, with daily S3 caching."""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import xarray as xr
from sliderule import icesat2
import boto3
from botocore.exceptions import ClientError

from lis_grid import build_lis_area_definition, load_lis_grid

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# Default temporal search window
T0_DEFAULT = "2019-04-01T00:00:00Z"
T1_DEFAULT = "2019-06-30T23:59:59Z"

# S3 configuration
S3_BUCKET = "airborne-smce-prod-user-bucket"
S3_PREFIX = "JOIN/ICESAT-2/ATL06_MOSAIC"

# Parallelization configuration
MAX_WORKERS = 10  # Number of parallel requests


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


def s3_file_exists(s3_path: str) -> bool:
    """Check if a file exists in S3."""
    # Parse S3 path
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]
    
    try:
        s3_client = boto3.client("s3")
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise


def upload_to_s3(local_path: Path, s3_path: str, overwrite: bool = False, max_retries: int = 3) -> bool:
    """Upload a local file to S3 with token refresh on expiration.
    
    Returns:
        True if file was uploaded, False if file already exists and overwrite is False
    """
    # Check if file already exists
    if not overwrite and s3_file_exists(s3_path):
        log.info("File already exists in S3 and overwrite is False: %s", s3_path)
        return False
    
    # Parse S3 path
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]
    
    for attempt in range(max_retries):
        try:
            # Create a fresh S3 client for each upload to ensure fresh credentials
            s3_client = boto3.client("s3")
            
            log.info("Uploading %s to s3://%s/%s (attempt %d/%d)", local_path, bucket, key, attempt + 1, max_retries)
            s3_client.upload_file(str(local_path), bucket, key)
            log.info("Successfully uploaded to S3")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ExpiredToken' and attempt < max_retries - 1:
                log.warning("Token expired, refreshing credentials and retrying (attempt %d/%d)", attempt + 1, max_retries)
                # Force boto3 to refresh credentials by creating a new session
                import importlib
                importlib.reload(boto3)
                continue
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                log.warning("Upload failed, retrying (attempt %d/%d): %s", attempt + 1, max_retries, e)
                continue
            else:
                raise


def fetch_atl06_for_day(
    polygon: list[dict],
    date: datetime,
    temp_dir: Path,
    overwrite: bool = False,
) -> tuple[datetime, int]:
    """Fetch ATL06 data for a single day from SlideRule and save to S3.
    
    Returns:
        Tuple of (date, number of observations)
    """
    s3_path = get_s3_path(date)
    if not overwrite and s3_file_exists(s3_path):
        log.info("Skipping %s because file already exists in S3: %s", date.date(), s3_path)
        return (date, 0)

    # Set time range for the day (00:00:00 to 23:59:59 UTC)
    t0 = date.strftime("%Y-%m-%dT00:00:00Z")
    t1 = date.strftime("%Y-%m-%dT23:59:59Z")
    
    log.info("Fetching ATL06 data for %s", date.date())
    
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
            return (date, 0)
        
        log.info("SlideRule returned %d ATL06 observations for %s", len(gdf), date.date())
        
        # Extract latitude and longitude from the geometry column
        gdf['lon'] = gdf.geometry.x
        gdf['lat'] = gdf.geometry.y
        
        # Save to temporary local file
        temp_file = temp_dir / date.strftime("atl06_%Y%m%d.parquet")
        gdf.to_parquet(temp_file)
        log.info("Saved daily data to %s", temp_file)
        
        # Upload to S3
        uploaded = upload_to_s3(temp_file, s3_path, overwrite=overwrite)
        
        # Clean up temporary file
        temp_file.unlink()
        log.info("Cleaned up temporary file for %s", date.date())
        
        return (date, len(gdf))
    
    except Exception as e:
        log.error("Error fetching ATL06 data for %s: %s", date.date(), e)
        return (date, 0)


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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Maximum number of parallel workers (default: {MAX_WORKERS}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in S3. If not set, existing files will be skipped.",
    )
    ns = parser.parse_args()

    # Load LIS grid
    log.info("Loading LIS grid from %s", ns.lis_path)
    lis_grid = load_lis_grid(ns.lis_path)
    
    # Build LIS bounding polygon for SlideRule query
    polygon = lis_domain_polygon(lis_grid)
    
    # Create temporary directory
    ns.temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize SlideRule client
    log.info("Initializing SlideRule client 3")
    icesat2.init(verbose=True)
    
    # Parse start and end times
    t0_dt = parse_iso_datetime(ns.t0)
    t1_dt = parse_iso_datetime(ns.t1)
    
    # Generate list of all dates to process
    current_date = t0_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = t1_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    dates_to_process = []
    while current_date <= end_date:
        dates_to_process.append(current_date)
        current_date += timedelta(days=1)
    
    log.info("Processing %d days with up to %d parallel workers", len(dates_to_process), ns.max_workers)
    
    # Process days in parallel
    total_observations = 0
    days_processed = 0
    
    with ThreadPoolExecutor(max_workers=ns.max_workers) as executor:
        # Submit all tasks
        future_to_date = {
            executor.submit(fetch_atl06_for_day, polygon, date, ns.temp_dir, ns.overwrite): date
            for date in dates_to_process
        }
        
        # Process completed tasks
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            try:
                result_date, num_obs = future.result()
                if num_obs > 0:
                    total_observations += num_obs
                    days_processed += 1
                    log.info("Completed %s: %d observations", result_date.date(), num_obs)
                else:
                    log.info("Completed %s: no observations or skipped", result_date.date())
            except Exception as e:
                log.error("Task for %s generated an exception: %s", date.date(), e)
    
    log.info("Completed processing ATL06 data for date range %s to %s", ns.t0, ns.t1)
    log.info("Processed %d days with %d total observations", days_processed, total_observations)


if __name__ == "__main__":
    main()