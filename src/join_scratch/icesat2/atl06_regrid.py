#!/usr/bin/env python
"""Grid ICESat-2 ATL06 snow height data to the LIS model grid.

Retrieves ATL06 data via the SlideRule Earth Python client, caches the result
locally as a Parquet file, then grids the point observations to the LIS Lambert
Conformal pixel grid by averaging all ATL06 h_mean values within each pixel.
Pixels with no observations are set to NaN, resulting in a spatially sparse
output that matches the grid dimensions and NetCDF structure of the AMSR2,
VIIRS, and CEDA regridded outputs.
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import xarray as xr
from sliderule import sliderule

from join_scratch.storage import (
    StorageConfig,
    add_storage_args,
    storage_config_from_namespace,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[3]
_DATA_OUT = _ROOT / "_data" / "icesat2"

# LIS file path relative to storage root
LIS_RELPATH = "lis_input_NMP_1000m_missouri.nc"

# Default temporal search window
T0_DEFAULT = "2019-01-01T00:00:00Z"
T1_DEFAULT = "2019-01-07T23:59:59Z"

# Local cache file for the SlideRule GeoDataFrame
CACHE_PATH = _DATA_OUT / "atl06_2019-01-01_2019-01-07.parquet"

# Output NetCDF file
OUTPUT_PATH = _DATA_OUT / "atl06_gridded_2019-01-01_2019-01-07.nc"


# ---------------------------------------------------------------------------
# LIS grid helpers
# ---------------------------------------------------------------------------


def load_lis_grid(storage: StorageConfig) -> xr.Dataset:
    """Load lat/lon coordinate variables from the LIS input file.

    Returns a Dataset with ``lat`` and ``lon`` variables (and their bounds),
    flipped to north-first row ordering consistent with other modules.
    """
    log.info("Loading LIS grid from storage (%s)", storage.storage_type)
    with storage.open(LIS_RELPATH) as f:
        ds = xr.open_dataset(f, engine="h5netcdf")
        ds = ds[["lat", "lon"]]
        ds = ds.isel(north_south=slice(None, None, -1))
        ds.load()
    return ds


def build_lis_lcc_crs(storage: StorageConfig) -> tuple[pyproj.CRS, dict]:
    """Build a pyproj CRS for the LIS Lambert Conformal grid.

    Returns (crs, grid_info) where grid_info contains:
      x_min, y_min, dx_m, dy_m, nx, ny
    """
    with storage.open(LIS_RELPATH) as f:
        ds = xr.open_dataset(f, engine="h5netcdf")
        ds.load()

    attrs = ds.attrs
    sw_lat = float(attrs["SOUTH_WEST_CORNER_LAT"])
    sw_lon = float(attrs["SOUTH_WEST_CORNER_LON"])
    dx_m = float(attrs["DX"]) * 1000.0
    dy_m = float(attrs["DY"]) * 1000.0
    truelat1 = float(attrs["TRUELAT1"])
    truelat2 = float(attrs["TRUELAT2"])
    standard_lon = float(attrs["STANDARD_LON"])
    ny, nx = ds["lat"].shape

    crs = pyproj.CRS.from_dict(
        {
            "proj": "lcc",
            "lat_1": truelat1,
            "lat_2": truelat2,
            "lon_0": standard_lon,
            "lat_0": (truelat1 + truelat2) / 2.0,
            "datum": "WGS84",
            "units": "m",
        }
    )

    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_sw, y_sw = transformer.transform(sw_lon, sw_lat)

    # SW corner is pixel centre; compute pixel edge for half-pixel offset
    x_min = x_sw - dx_m / 2.0
    y_min = y_sw - dy_m / 2.0

    grid_info = dict(x_min=x_min, y_min=y_min, dx_m=dx_m, dy_m=dy_m, nx=nx, ny=ny)
    return crs, grid_info


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


# ---------------------------------------------------------------------------
# SlideRule retrieval & caching
# ---------------------------------------------------------------------------


def retrieve_atl06(
    polygon: list[dict],
    t0: str = T0_DEFAULT,
    t1: str = T1_DEFAULT,
) -> gpd.GeoDataFrame:
    """Query SlideRule for ATL06 snow height data within the given polygon and time range.

    Uses the standard ATL06 subset endpoint (``atl06x``), which returns the
    processed land-ice surface height product.  The ``h_mean`` column contains
    the along-track mean height for each 40 m segment.
    """
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
    return gdf


def load_or_fetch_atl06(
    polygon: list[dict],
    cache_path: Path = CACHE_PATH,
    force_download: bool = False,
    t0: str = T0_DEFAULT,
    t1: str = T1_DEFAULT,
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

    gdf = retrieve_atl06(polygon, t0=t0, t1=t1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Saving ATL06 data to cache: %s", cache_path)
    gdf.to_parquet(cache_path)

    return gdf


# ---------------------------------------------------------------------------
# Gridding
# ---------------------------------------------------------------------------


def grid_atl06(
    gdf: gpd.GeoDataFrame,
    crs: pyproj.CRS,
    grid_info: dict,
) -> np.ndarray:
    """Project ATL06 points to the LIS CRS and compute per-pixel mean h_mean.

    Points outside the LIS domain are silently dropped.  Pixels with no
    observations receive NaN.

    Returns a float32 array of shape (ny, nx) in north-first row order
    (row 0 = northernmost row), consistent with the LIS grid after the
    north-first flip applied in :func:`load_lis_grid`.
    """
    x_min: float = grid_info["x_min"]
    y_min: float = grid_info["y_min"]
    dx_m: float = grid_info["dx_m"]
    dy_m: float = grid_info["dy_m"]
    nx: int = grid_info["nx"]
    ny: int = grid_info["ny"]

    if len(gdf) == 0:
        log.warning("ATL06 GeoDataFrame is empty; returning all-NaN grid")
        return np.full((ny, nx), np.nan, dtype=np.float32)

    # Project to LIS CRS
    log.info("Projecting %d ATL06 points to LIS CRS …", len(gdf))
    gdf_lcc = gdf.to_crs(crs)

    xs = np.asarray(gdf_lcc.geometry.x.values, dtype=np.float64)
    ys = np.asarray(gdf_lcc.geometry.y.values, dtype=np.float64)
    h = gdf["h_li"].values.astype(np.float64)

    # Map projected coordinates to pixel indices
    # LIS stores rows south-first in the file; after the north-first flip,
    # row 0 corresponds to the maximum y value (northernmost).
    col = np.floor((xs - x_min) / dx_m).astype(int)
    # y_min is south edge; row index in south-first space: row_s = (y - y_min) / dy
    # After north-first flip: row_n = (ny - 1) - row_s
    row_s = np.floor((ys - y_min) / dy_m).astype(int)
    row_n = (ny - 1) - row_s

    # Keep only points that fall inside the grid
    valid = (col >= 0) & (col < nx) & (row_n >= 0) & (row_n < ny) & np.isfinite(h)
    col = col[valid]
    row_n = row_n[valid]
    h = h[valid]
    log.info("%d points fall within the LIS domain", valid.sum())

    # Accumulate sum and count per pixel
    flat_idx = row_n * nx + col
    h_sum = np.zeros(ny * nx, dtype=np.float64)
    count = np.zeros(ny * nx, dtype=np.int32)
    np.add.at(h_sum, flat_idx, h)
    np.add.at(count, flat_idx, 1)

    # Compute mean; leave as NaN where count == 0
    with np.errstate(invalid="ignore"):
        h_mean = np.where(count > 0, h_sum / count, np.nan).reshape(ny, nx)

    n_filled = int((count > 0).sum())
    log.info(
        "Gridded %d points into %d/%d pixels (%.2f%% fill)",
        valid.sum(),
        n_filled,
        ny * nx,
        100.0 * n_filled / (ny * nx),
    )
    return h_mean.astype(np.float32)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_output(
    h_mean: np.ndarray,
    lis_grid: xr.Dataset,
    out_path: Path,
) -> Path:
    """Write the gridded h_mean array to a NetCDF file.

    The output shares dimension names and coordinate variables with the LIS
    grid (``north_south``, ``east_west``, ``lat``, ``lon``).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            "h_li": xr.DataArray(
                h_mean,
                dims=["north_south", "east_west"],
                attrs={
                    "long_name": "ICESat-2 ATL06 land-ice surface height",
                    "units": "meters",
                    "comment": (
                        "Mean of ATL06 h_li values within each LIS pixel. "
                        "NaN indicates no ICESat-2 observations in this pixel."
                    ),
                },
            )
        },
        coords={
            "lat": lis_grid["lat"],
            "lon": lis_grid["lon"],
        },
    )

    encoding = {"h_li": {"dtype": "float32", "_FillValue": np.float32("nan")}}
    log.info("Writing output to %s", out_path)
    ds.to_netcdf(out_path, engine="h5netcdf", encoding=encoding)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid ICESat-2 ATL06 snow height data to the LIS model grid."
    )
    add_storage_args(parser)
    parser.add_argument(
        "--force-download",
        action="store_true",
        default=False,
        help=(
            "Re-query SlideRule and overwrite the local Parquet cache even if "
            "it already exists."
        ),
    )
    ns = parser.parse_args()
    storage = storage_config_from_namespace(ns)

    # Load LIS grid for coordinates and domain polygon
    lis_grid = load_lis_grid(storage)
    polygon = lis_domain_polygon(lis_grid)
    log.info(
        "LIS domain bounding box: lon [%.3f, %.3f], lat [%.3f, %.3f]",
        min(p["lon"] for p in polygon),
        max(p["lon"] for p in polygon),
        min(p["lat"] for p in polygon),
        max(p["lat"] for p in polygon),
    )

    # Build LIS CRS and pixel grid parameters (reads LIS file again for attrs)
    crs, grid_info = build_lis_lcc_crs(storage)

    # Retrieve or load cached ATL06 data
    gdf = load_or_fetch_atl06(
        polygon,
        cache_path=CACHE_PATH,
        force_download=ns.force_download,
    )

    # Grid to LIS pixels
    h_mean = grid_atl06(gdf, crs, grid_info)

    # Write output
    out_path = write_output(h_mean, lis_grid, OUTPUT_PATH)
    log.info("Done: %s", out_path)


if __name__ == "__main__":
    main()
