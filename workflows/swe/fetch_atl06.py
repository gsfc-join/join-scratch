#!/usr/bin/env python
"""Fetch ICESat-2 ATL06 data for the LIS domain via SlideRule, with local caching."""

import sys
import logging
from pathlib import Path
import argparse

import geopandas as gpd
import xarray as xr
from sliderule import sliderule

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


def load_or_fetch_atl06(
    polygon: list[dict],
    t0: str,
    t1: str,
    cache_path: Path,
    force_download: bool = False,
) -> gpd.GeoDataFrame:
    """Return ATL06 GeoDataFrame from local Parquet cache or SlideRule.

    If *force_download* is True, always re-queries SlideRule and overwrites the
    cache even if a valid cache file already exists.
    """
    if not force_download and cache_path.exists():
        log.info("Loading ATL06 data from cache: %s", cache_path)
        gdf = gpd.read_parquet(cache_path)
        log.info("Loaded %d observations from cache", len(gdf))
        return gdf

    if force_download and cache_path.exists():
        log.info("--force-download set; ignoring existing cache at %s", cache_path)

    log.info("Initializing SlideRule client …")
    sliderule.init()

    parms: dict = {
        "poly": polygon,
        "t0": t0,
        "t1": t1,
    }

    log.info("Requesting ATL06 data from SlideRule (t0=%s, t1=%s) …", t0, t1)
    gdf: gpd.GeoDataFrame = sliderule.run("atl06x", parms)
    log.info("SlideRule returned %d ATL06 observations", len(gdf))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Saving ATL06 data to cache: %s", cache_path)
    gdf.to_parquet(cache_path)

    return gdf

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch ICESat-2 ATL06 fetch"
    )
    parser.add_argument(
        "--lis-path",
        default=f"s3://airborne-smce-prod-user-bucket/JOIN/lis_input_NMP_1000m_missouri.nc",
        help="Local path or s3:// URI to the LIS NetCDF file.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("_data/icesat2/atl06_benchmark.parquet"),
        help="Path to the cached ATL06 Parquet file (default: _data/icesat2/atl06_benchmark.parquet).",
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
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write timestamped report file (default: print to stdout only).",
    )
    ns = parser.parse_args()

    lis_grid = load_lis_grid(ns.lis_path)
    lis_area = build_lis_area_definition(ns.lis_path)

    # Build LIS bounding polygon for SlideRule query
    lat = lis_grid["lat"].values
    lon = lis_grid["lon"].values
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    polygon = [
        {"lon": lon_min, "lat": lat_min},
        {"lon": lon_max, "lat": lat_min},
        {"lon": lon_max, "lat": lat_max},
        {"lon": lon_min, "lat": lat_max},
        {"lon": lon_min, "lat": lat_min},
    ]

    parms: dict = {
        "poly": polygon,
        "t0": ns.t0,
        "t1": ns.t1,
    }

    ns.cache_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []

    if ns.cache_path.exists():
        ns.cache_path.unlink()
        log.info("removed existing Parquet cache %s", ns.cache_path)

    log.info("Initialising SlideRule client …")
    from sliderule import sliderule as _sliderule
    _sliderule.init()

    cache_path = ns.cache_path

    gdf: gpd.GeoDataFrame = _sliderule.run("atl06x", parms)
    gdf.to_parquet(cache_path)

if __name__ == "__main__":
    main()
