"""Shared LIS grid utilities for SWE workflow scripts."""

import logging
from pathlib import Path

import pyproj
import xarray as xr
from pyresample.geometry import AreaDefinition

log = logging.getLogger(__name__)

LIS_RELPATH = "lis_input_NMP_1000m_missouri.nc"


def load_lis_grid(path: Path | str, fs=None) -> xr.Dataset:
    """Load the LIS input file and return a dataset with lat/lon/lat_b/lon_b.

    The LIS file stores rows south-first (row 0 = southernmost).  This
    function flips the north_south (and north_south_b) dimensions so that
    row 0 = northernmost, matching the convention used by pyresample
    AreaDefinition and consistent with all regrid outputs.

    Parameters
    ----------
    path:
        Path to the LIS input NetCDF file.
    fs:
        Optional fsspec filesystem object.  If provided, opens via FSFile.
        If None, opens locally.
    """
    log.info("Loading LIS grid from %s", path)
    if fs is not None:
        from satpy.readers.core.remote import FSFile

        f = FSFile(path, fs=fs)
        ds = xr.open_dataset(f, engine="h5netcdf")
    else:
        ds = xr.open_dataset(path, engine="h5netcdf")

    ds = ds[["lat", "lon", "lat_b", "lon_b"]]
    ds = ds.isel(
        north_south=slice(None, None, -1),
        north_south_b=slice(None, None, -1),
    )
    ds.load()
    return ds


def build_lis_area_definition(path: Path | str, fs=None) -> AreaDefinition:
    """Construct a pyresample AreaDefinition for the LIS Lambert Conformal grid.

    All parameters are read from the global attributes of the LIS input file:
      - MAP_PROJECTION: LAMBERT CONFORMAL
      - SOUTH_WEST_CORNER_LAT/LON (pixel centres)
      - TRUELAT1/TRUELAT2, STANDARD_LON
      - DX/DY (km), grid shape from the lat/lon variable dimensions

    Parameters
    ----------
    path:
        Path to the LIS input NetCDF file.
    fs:
        Optional fsspec filesystem object.  If provided, opens via FSFile.
        If None, opens locally.
    """
    log.info("Building LIS AreaDefinition from %s", path)
    if fs is not None:
        from satpy.readers.core.remote import FSFile

        f = FSFile(path, fs=fs)
        ds = xr.open_dataset(f, engine="h5netcdf")
    else:
        ds = xr.open_dataset(path, engine="h5netcdf")
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

    x_min = x_sw - dx_m / 2
    y_min = y_sw - dy_m / 2
    x_max = x_min + nx * dx_m
    y_max = y_min + ny * dy_m

    return AreaDefinition(
        "lis_lcc",
        "LIS Lambert Conformal 1 km",
        "lis_lcc",
        crs.to_dict(),
        nx,
        ny,
        (x_min, y_min, x_max, y_max),
    )
