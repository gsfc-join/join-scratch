#!/usr/bin/env python
"""Regrid local VIIRS CGF Snow Cover files to the LIS input grid.

VIIRS VJ110A1F tiles use the MODIS Sinusoidal (SIN) projection stored in
HDFEOS HDF5 files.  Each tile is 3000×3000 pixels at 375 m resolution.
Because the source grid is a projected (non-geographic) tile, it is treated
as swath data: pixel centres are derived from the XDim/YDim projection
coordinates via pyproj, and pyresample/satpy resampler methods are used
(xESMF bilinear is not used since xESMF requires a regular lon/lat source).

Multi-tile compositing
----------------------
When more than one tile is provided, the pixel centres (lon, lat) and data
arrays for all tiles are concatenated along the row axis before building the
SwathDefinition so that a single regridding pass covers the full LIS domain.

Flag masking
------------
Valid NDSI snow cover values are 0–100.  Special flags (>100) such as 201
(no decision), 211 (night), 237 (inland water), 239 (ocean), 250 (cloud),
255 (fill) are masked to NaN before regridding.
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import pyproj
import xarray as xr
from pyproj.crs import GeographicCRS, ProjectedCRS
from pyproj.crs.coordinate_operation import SinusoidalConversion
from pyproj.crs.datum import CustomDatum, CustomEllipsoid
from pyresample.geometry import AreaDefinition, SwathDefinition
from satpy.resample.bucket import BucketAvg
from satpy.resample.ewa import DaskEWAResampler
from satpy.resample.kdtree import BilinearResampler, KDTreeResampler

from join_scratch.amsr2.amsr2_regrid import build_lis_area_definition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = ROOT / "_data-raw"
DATA_OUT = ROOT / "_data" / "viirs"
LIS_PATH = DATA_RAW / "lis_input_NMP_1000m_missouri.nc"
VIIRS_GLOB = "JOIN/VIIRS/**/*.h5"
# Satpy caches neighbour-lookup tables as zarr files; shared across methods.
SATPY_CACHE = DATA_OUT / "satpy-cache"

# ---------------------------------------------------------------------------
# VIIRS constants
# ---------------------------------------------------------------------------

# MODIS Sinusoidal projection (HDFEOS standard)
# earth_radius=6371007.181 matches the value stored in the HDFEOS metadata.
# Built programmatically to avoid lossy PROJ-string round-trips.
_SIN_CRS = ProjectedCRS(
    name="MODIS Sinusoidal",
    conversion=SinusoidalConversion(),
    geodetic_crs=GeographicCRS(
        datum=CustomDatum(
            ellipsoid=CustomEllipsoid(
                name="MODIS Sphere",
                semi_major_axis=6371007.181,
                inverse_flattening=0,
            )
        )
    ),
)

# HDF5 path to the primary variable inside the HDFEOS tile file
_HDFEOS_DATA_PATH = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields/CGF_NDSI_Snow_Cover"
_HDFEOS_XDIM_PATH = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/XDim"
_HDFEOS_YDIM_PATH = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/YDim"

# Pixel size in metres (375 m product)
VIIRS_PIXEL_M = 375.0
# radius_of_influence: slightly larger than the pixel diagonal to avoid gaps
# diagonal ≈ 375 * sqrt(2) ≈ 530 m; we use 600 m
RADIUS_OF_INFLUENCE = 600.0


# ---------------------------------------------------------------------------
# Tile loading
# ---------------------------------------------------------------------------


def _sin_to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert MODIS SIN projection coordinates (metres) to lon/lat (degrees).

    x and y can be 1-D (coordinate vectors) or 2-D (pixel centres).
    Returns (lon, lat) arrays of the same shape.
    """
    transformer = pyproj.Transformer.from_crs(_SIN_CRS, "EPSG:4326", always_xy=True)
    return transformer.transform(x, y)


def load_viirs_tile(path: Path) -> dict:
    """Load one VIIRS HDF5 tile and return a dict with keys:

    - ``data``: float32 (ny, nx) array, flag values (>100) masked to NaN
    - ``lon2d``: float64 (ny, nx) array of pixel-centre longitudes
    - ``lat2d``: float64 (ny, nx) array of pixel-centre latitudes
    - ``stem``: str, file stem used for output naming

    The sinusoidal tile XDim/YDim coordinates (in metres) are broadcast to a
    2-D meshgrid and transformed to geographic coordinates via pyproj.
    """
    log.info("Loading VIIRS tile %s", path)
    with h5py.File(path, "r") as f:
        raw = f[_HDFEOS_DATA_PATH][:]  # uint8 (ny, nx)
        x_coords = f[_HDFEOS_XDIM_PATH][:]  # float64 (nx,) in metres
        y_coords = f[_HDFEOS_YDIM_PATH][:]  # float64 (ny,) in metres

    # Mask special-flag values (valid data: 0–100)
    data = raw.astype(np.float32)
    data[data > 100] = np.nan

    # Build 2-D pixel-centre coordinate arrays
    x2d, y2d = np.meshgrid(x_coords, y_coords)
    lon2d, lat2d = _sin_to_lonlat(x2d, y2d)

    return {
        "data": data,
        "lon2d": lon2d,
        "lat2d": lat2d,
        "stem": path.stem,
    }


def load_viirs_tiles(paths: list[Path]) -> dict:
    """Load and composite multiple VIIRS tiles into a single swath.

    Tiles are stacked along the row axis (axis 0).  This produces one large
    SwathDefinition covering all provided tiles so that a single regridding
    pass serves the full LIS domain.

    Returns the same dict structure as ``load_viirs_tile`` with a composite
    ``stem`` derived from the first tile's date component.
    """
    if len(paths) == 1:
        return load_viirs_tile(paths[0])

    tiles = [load_viirs_tile(p) for p in paths]
    return {
        "data": np.concatenate([t["data"] for t in tiles], axis=0),
        "lon2d": np.concatenate([t["lon2d"] for t in tiles], axis=0),
        "lat2d": np.concatenate([t["lat2d"] for t in tiles], axis=0),
        "stem": tiles[0]["stem"],
    }


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------


def build_viirs_swath_definition(tile: dict) -> SwathDefinition:
    """Build a pyresample SwathDefinition from a loaded VIIRS tile dict.

    lon2d and lat2d are wrapped as xarray DataArrays with dims ['y', 'x']
    so that satpy resamplers (which require .dims on geometry arrays) work.
    """
    lon_da = xr.DataArray(tile["lon2d"], dims=["y", "x"])
    lat_da = xr.DataArray(tile["lat2d"], dims=["y", "x"])
    return SwathDefinition(lons=lon_da, lats=lat_da)


def _data_as_da(tile: dict) -> xr.DataArray:
    """Return the tile data as a 2-D xr.DataArray with dims ['y', 'x']."""
    return xr.DataArray(tile["data"], dims=["y", "x"])


# ---------------------------------------------------------------------------
# satpy regridding
# ---------------------------------------------------------------------------


def regrid_nearest(
    tile: dict,
    source_def: SwathDefinition,
    target_def: AreaDefinition,
    cache_dir: Path = SATPY_CACHE,
) -> np.ndarray:
    """Regrid using satpy KDTreeResampler (nearest-neighbour).

    Neighbour indices are cached to *cache_dir* on first run.
    Returns a float32 (NY, NX) array.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    resampler = KDTreeResampler(source_def, target_def)
    resampler.precompute(
        radius_of_influence=RADIUS_OF_INFLUENCE,
        cache_dir=str(cache_dir),
    )
    return np.asarray(resampler.compute(_data_as_da(tile), fill_value=np.nan))


def regrid_bilinear(
    tile: dict,
    source_def: SwathDefinition,
    target_def: AreaDefinition,
    cache_dir: Path = SATPY_CACHE,
) -> np.ndarray:
    """Regrid using satpy BilinearResampler.

    Bilinear coefficients are cached to *cache_dir* on first run.
    Returns a float32 (NY, NX) array.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    resampler = BilinearResampler(source_def, target_def)
    return np.asarray(
        resampler.resample(
            _data_as_da(tile),
            radius_of_influence=RADIUS_OF_INFLUENCE,
            fill_value=np.nan,
            cache_dir=str(cache_dir),
        )
    )


def regrid_ewa(
    tile: dict,
    source_def: SwathDefinition,
    target_def: AreaDefinition,
) -> np.ndarray:
    """Regrid using satpy DaskEWAResampler (Elliptical Weighted Averaging).

    rows_per_scan=0 disables scan-line grouping; no caching supported.
    Returns a float32 (NY, NX) array.
    """
    resampler = DaskEWAResampler(source_def, target_def)
    return np.asarray(
        resampler.resample(
            _data_as_da(tile),
            rows_per_scan=0,
            fill_value=np.nan,
        )
    )


def regrid_bucket_avg(
    tile: dict,
    source_def: SwathDefinition,
    target_def: AreaDefinition,
) -> np.ndarray:
    """Regrid using satpy BucketAvg (scatter-add average per target cell).

    BucketAvg assigns each *source* pixel to the single *target* cell its
    centre falls in, then averages all source pixels in that cell.  For VIIRS
    375 m → LIS 1 km the source is finer than the target (~7 source pixels
    per target cell), so bucket_avg is appropriate and will produce valid
    output where the tiles overlap the LIS domain.

    No caching supported.  Returns a float32 (NY, NX) array.
    """
    resampler = BucketAvg(source_def, target_def)
    return np.asarray(resampler.resample(_data_as_da(tile), fill_value=np.nan))


# ---------------------------------------------------------------------------
# File-level output (nearest-neighbour as default, writes NetCDF)
# ---------------------------------------------------------------------------


def regrid_tile_to_nc(
    tile: dict,
    source_def: SwathDefinition,
    target_def: AreaDefinition,
    lis_lons: np.ndarray,
    lis_lats: np.ndarray,
    out_dir: Path,
    method: str = "nearest",
) -> Path:
    """Regrid a composited tile and write the result to a NetCDF file.

    Only one method is used per call; default is nearest-neighbour.
    Returns the path of the written output file.
    """
    log.info("Regridding VIIRS tile(s) with method=%s …", method)
    dispatch = {
        "nearest": regrid_nearest,
        "bilinear": regrid_bilinear,
        "ewa": regrid_ewa,
        "bucket_avg": regrid_bucket_avg,
    }
    fn = dispatch[method]
    if method in ("nearest", "bilinear"):
        arr = fn(tile, source_def, target_def, SATPY_CACHE)
    else:
        arr = fn(tile, source_def, target_def)

    ds_out = xr.Dataset(
        {
            "CGF_NDSI_Snow_Cover": xr.DataArray(
                arr,
                dims=["y", "x"],
                attrs={"long_name": "CGF NDSI Snow Cover", "units": "1"},
            ),
            "lat": xr.DataArray(lis_lats, dims=["y", "x"]),
            "lon": xr.DataArray(lis_lons, dims=["y", "x"]),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tile['stem']}_{method}.nc"
    log.info("Writing output to %s", out_path)
    ds_out.to_netcdf(out_path, engine="h5netcdf")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    viirs_files = sorted(DATA_RAW.glob(VIIRS_GLOB))
    if not viirs_files:
        raise FileNotFoundError(
            f"No VIIRS files found matching '{VIIRS_GLOB}' under {DATA_RAW}"
        )
    log.info("Found %d VIIRS file(s)", len(viirs_files))

    # Group files by date (YYYYDDD encoded in filename, e.g. A2019001)
    from collections import defaultdict

    date_groups: dict[str, list[Path]] = defaultdict(list)
    for p in viirs_files:
        # Filename format: VJ110A1F.A2019001.h00v08.002.<ts>.h5
        parts = p.stem.split(".")
        date_key = parts[1] if len(parts) > 1 else p.stem
        date_groups[date_key].append(p)

    lis_area = build_lis_area_definition(LIS_PATH)
    lis_lons, lis_lats = lis_area.get_lonlats()

    for date_key, paths in sorted(date_groups.items()):
        log.info("Processing date %s (%d tile(s))", date_key, len(paths))
        tile = load_viirs_tiles(paths)
        source_def = build_viirs_swath_definition(tile)

        out_path = regrid_tile_to_nc(
            tile, source_def, lis_area, lis_lons, lis_lats, DATA_OUT
        )
        log.info("Done: %s", out_path)

    log.info("All VIIRS tiles regridded successfully.")


if __name__ == "__main__":
    main()
