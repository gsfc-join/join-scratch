"""Shared LIS grid utilities for SWE workflow scripts."""

import hashlib
import logging
import pickle
from pathlib import Path

import pyproj
import xarray as xr
from pyresample.geometry import AreaDefinition

log = logging.getLogger(__name__)

LIS_RELPATH = "lis_input_NMP_1000m_missouri.nc"


def _open_lis_ds(path, fs=None) -> xr.Dataset:
    """Open a LIS NetCDF as an xarray Dataset, supporting S3 URIs via obstore.

    Parameters
    ----------
    path:
        Local path or ``s3://`` URI to the LIS NetCDF file.
    fs:
        Optional obstore ``FsspecStore`` (or any fsspec-compatible object).
        If *None* and *path* starts with ``s3://``, one is created automatically
        via :func:`s3_utils.make_fs`.  If *None* and path is local, opens
        directly.
    """
    path_str = str(path)
    if path_str.startswith("s3://"):
        if fs is None:
            from s3_utils import make_fs
            fs = make_fs()
        fobj = fs.open(path_str)
        ds = xr.open_dataset(fobj, engine="h5netcdf")
    elif fs is not None:
        from satpy.readers.core.remote import FSFile
        f = FSFile(path_str, fs=fs)
        ds = xr.open_dataset(f, engine="h5netcdf")
    else:
        ds = xr.open_dataset(path_str, engine="h5netcdf")
    return ds


def load_lis_grid(path: Path | str, fs=None) -> xr.Dataset:
    """Load the LIS input file and return a dataset with lat/lon/lat_b/lon_b.

    The LIS file stores rows south-first (row 0 = southernmost).  This
    function flips the north_south (and north_south_b) dimensions so that
    row 0 = northernmost, matching the convention used by pyresample
    AreaDefinition and consistent with all regrid outputs.

    Parameters
    ----------
    path:
        Local path or ``s3://`` URI to the LIS input NetCDF file.
    fs:
        Optional obstore ``FsspecStore``.  If *None* and *path* is an S3 URI,
        a store is created automatically.
    """
    log.info("Loading LIS grid from %s", path)
    ds = _open_lis_ds(path, fs=fs)

    ds = ds[["lat", "lon", "lat_b", "lon_b"]]
    ds = ds.isel(
        north_south=slice(None, None, -1),
        north_south_b=slice(None, None, -1),
    )
    ds.load()
    return ds


def build_lis_area_definition(
    path: Path | str,
    fs=None,
    cache_dir: Path | str | None = "_data/weights",
    overwrite: bool = False,
) -> AreaDefinition:
    """Construct a pyresample AreaDefinition for the LIS Lambert Conformal grid.

    All parameters are read from the global attributes of the LIS input file:
      - MAP_PROJECTION: LAMBERT CONFORMAL
      - SOUTH_WEST_CORNER_LAT/LON (pixel centres)
      - TRUELAT1/TRUELAT2, STANDARD_LON
      - DX/DY (km), grid shape from the lat/lon variable dimensions

    The result is pickled to *cache_dir* so subsequent calls are near-instant.
    Pass *overwrite=True* to force recomputation.

    Parameters
    ----------
    path:
        Local path or ``s3://`` URI to the LIS input NetCDF file.
    fs:
        Optional obstore ``FsspecStore``.  If *None* and *path* is an S3 URI,
        a store is created automatically.
    cache_dir:
        Local directory for the pickle cache.  Pass *None* to disable caching.
    overwrite:
        If True, recompute and overwrite any existing cache file.
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        key = hashlib.md5(str(path).encode()).hexdigest()[:12]
        cache_path = Path(cache_dir) / f"lis-area-def-{key}.pkl"
        if cache_path.exists() and not overwrite:
            log.info("Reusing cached LIS AreaDefinition from %s", cache_path)
            with cache_path.open("rb") as f:
                return pickle.load(f)

    log.info("Building LIS AreaDefinition from %s", path)
    ds = _open_lis_ds(path, fs=fs)
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

    area_def = AreaDefinition(
        "lis_lcc",
        "LIS Lambert Conformal 1 km",
        "lis_lcc",
        crs.to_dict(),
        nx,
        ny,
        (x_min, y_min, x_max, y_max),
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(area_def, f)
        log.info("LIS AreaDefinition cached to %s", cache_path)

    return area_def
