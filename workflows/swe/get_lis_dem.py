#!/usr/bin/env python
"""
Standalone module to extract USGS 3DEP 10m DEM for a given LIS grid using SlideRule.
Uses ThreadPoolExecutor for parallel API requests with progress tracking.
Saves the flattened DEM data as a Parquet file.
"""

import argparse
import logging
import os
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sliderule import raster
from lis_grid import load_lis_grid
from s3_utils import _is_s3, make_fs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def _fetch_chunk(start_idx: int, chunk: list) -> tuple:
    """Helper function to fetch a single chunk from SlideRule."""
    try:
        result = raster.sample("usgs3dep-10meter-dem", chunk)
        
        vals = []
        if isinstance(result, pd.DataFrame) and not result.empty:
            num_cols = result.select_dtypes(include=['number'])
            if not num_cols.empty:
                vals = num_cols.iloc[:, 0].values
        elif isinstance(result, list) and result:
            if isinstance(result[0], dict):
                keys = list(result[0].keys())
                vals = [item.get("value", item.get(keys[0], np.nan)) for item in result]
            else:
                vals = result
        
        vals = np.array(vals, dtype=np.float32) if len(vals) > 0 else np.array([])
        return start_idx, vals
    except Exception as e:
        log.error("SlideRule API Error at index %d: %s", start_idx, e)
        return start_idx, np.array([])


def compute_and_save_dem(lis_path: str, output_parquet: str, chunk_size: int = 10000, max_workers: int = 8):
    fs = make_fs() if _is_s3(lis_path) else None
    
    log.info("Loading LIS grid from %s", lis_path)
    lis_grid = load_lis_grid(lis_path, fs=fs)
    
    log.info("Starting Parallel SlideRule 3DEP DEM extraction...")
    
    # Flatten the 2D lat/lon arrays from the LIS grid
    lons = lis_grid.lon.values.ravel()
    lats = lis_grid.lat.values.ravel()
    
    # Create (lon, lat) pairs expected by SlideRule
    coords_array = np.column_stack((lons, lats))
    total_points = len(coords_array)
    
    # Pre-allocate array with NaNs
    dem_elevations = np.full(total_points, np.nan, dtype=np.float32)
    
    # Prepare chunks
    chunks = []
    for i in range(0, total_points, chunk_size):
        end_idx = min(i + chunk_size, total_points)
        chunks.append((i, coords_array[i:end_idx].tolist()))
        
    total_chunks = len(chunks)
    log.info("Prepared %d chunks (%d total points). Fetching using %d threads...", total_chunks, total_points, max_workers)

    # Variables for tracking progress
    completed_chunks = 0
    # Log progress roughly every 5% of completion
    log_interval = max(1, total_chunks // 20)

    # Execute requests in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the thread pool
        future_to_chunk = {executor.submit(_fetch_chunk, start_idx, chunk): start_idx for start_idx, chunk in chunks}
        
        # Process them as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            start_idx, vals = future.result()
            
            valid_len = min(len(vals), total_points - start_idx)
            if valid_len > 0:
                dem_elevations[start_idx : start_idx + valid_len] = vals[:valid_len]
                
            completed_chunks += 1
            
            # Print progress
            if completed_chunks % log_interval == 0 or completed_chunks == total_chunks:
                percent = (completed_chunks / total_chunks) * 100
                log.info("Progress: %d / %d chunks completed (%.1f%%)", completed_chunks, total_chunks, percent)

    valid_mask = (dem_elevations > -9000) & (~np.isnan(dem_elevations))
    log.info("Finished. Successfully fetched %d valid DEM elevations.", np.count_nonzero(valid_mask))

    # Create a DataFrame ensuring the order perfectly matches the flattened LIS grid
    df = pd.DataFrame({
        "lon": lons,
        "lat": lats,
        "dem": dem_elevations
    })

    # Ensure output directory exists
    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    log.info("Saving DEM data to %s", out_path)
    df.to_parquet(out_path, engine="pyarrow")
    log.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LIS DEM via SlideRule and save to Parquet.")
    parser.add_argument("--lis-path", required=True, help="Path to the LIS input NetCDF file (local or s3://).")
    parser.add_argument("--output", default="./_data/dem/lis_dem.parquet", help="Output Parquet path.")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Number of points per SlideRule request.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel threads to use.")
    
    # Inject AWS credentials for obstore if needed
    import boto3
    creds = boto3.Session().get_credentials()
    if creds:
        frozen_creds = creds.get_frozen_credentials()
        os.environ["AWS_ACCESS_KEY_ID"] = frozen_creds.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = frozen_creds.secret_key
        if frozen_creds.token: os.environ["AWS_SESSION_TOKEN"] = frozen_creds.token
        os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

    ns = parser.parse_args()
    compute_and_save_dem(ns.lis_path, ns.output, chunk_size=ns.chunk_size, max_workers=ns.workers)