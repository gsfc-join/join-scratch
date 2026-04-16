#!/usr/bin/env python
"""Regrid VIIRS CGF Snow Cover files to the LIS input grid.

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

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pyproj
import xarray as xr
from pyproj.crs import GeographicCRS, ProjectedCRS
from pyproj.crs.coordinate_operation import SinusoidalConversion
from pyproj.crs.datum import CustomDatum, CustomEllipsoid
from pyresample.ewa.dask_ewa import DaskEWAResampler
from pyresample.geometry import AreaDefinition, SwathDefinition
from satpy.resample.bucket import BucketAvg
from satpy.resample.kdtree import BilinearResampler, KDTreeResampler

# Default buffer added around the LIS domain when filtering tiles (degrees)
DOMAIN_FILTER_BUFFER_DEG: float = 2.0

from join_scratch.amsr2.amsr2_regrid import build_lis_area_definition
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
_DATA_OUT = _ROOT / "_data" / "viirs"

# Glob pattern and LIS file relative to storage root
VIIRS_GLOB = "JOIN/VIIRS/**/*.h5"
LIS_RELPATH = "lis_input_NMP_1000m_missouri.nc"

# Local output path (output always written to local disk)
SATPY_CACHE = _DATA_OUT / "satpy-cache"

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


def _lis_lonlat_bounds(area: AreaDefinition) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) for a pyresample AreaDefinition.

    Transforms all four corners of the projected extent to geographic
    coordinates to handle non-rectangular (e.g. Lambert Conformal) domains.
    """
    x_min, y_min, x_max, y_max = area.area_extent
    corners_xy = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    )
    transformer = pyproj.Transformer.from_crs(area.crs, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(corners_xy[:, 0], corners_xy[:, 1])
    return float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())


def _sin_tile_lonlat_bbox(h: int, v: int) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) for a MODIS/VIIRS SIN tile.

    Tile bounds are computed purely from (h, v) indices using the MODIS SIN
    grid geometry — no file I/O required.  See docs/VIIRS.md for the math.

    The MODIS SIN grid divides the globe into 36×18 tiles.  Each tile spans
    exactly 10° of latitude (tile_height_y = pi*R/18) and the same distance
    in metres along the x-axis (tile_width_x = 2*pi*R/36).

    Lon bounds are computed at both the southern and northern latitude edges
    because the sinusoidal projection compresses x by cos(lat); the actual
    extreme longitudes depend on which latitude edge is closer to the equator.
    """
    R = 6371007.181  # MODIS sphere radius (metres)
    tile_w = 2 * np.pi * R / 36  # ≈ 1,111,950 m
    tile_h = np.pi * R / 18  # ≈ 1,111,950 m

    x_min = (h - 18) * tile_w
    x_max = x_min + tile_w
    y_min = (9 - v - 1) * tile_h  # y increases northward; v increases southward
    y_max = y_min + tile_h

    transformer = pyproj.Transformer.from_crs(_SIN_CRS, "EPSG:4326", always_xy=True)
    # Evaluate lon at all four corners; lon extremes depend on lat (cos compression)
    corners_x = np.array([x_min, x_max, x_max, x_min])
    corners_y = np.array([y_min, y_min, y_max, y_max])
    lons, lats = transformer.transform(corners_x, corners_y)
    return float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())


def filter_tiles_by_domain(
    paths: list[str],
    area: AreaDefinition,
    buffer_deg: float = DOMAIN_FILTER_BUFFER_DEG,
) -> list[str]:
    """Return only the tiles whose bounding box overlaps the LIS domain.

    Tile bounds are derived purely from the ``hXXvYY`` tile index encoded in
    each filename using the MODIS SIN grid geometry — no file I/O is needed.

    Parameters
    ----------
    paths:
        List of VIIRS tile paths as returned by ``StorageConfig.glob``.
    area:
        The target pyresample ``AreaDefinition`` (LIS domain).
    buffer_deg:
        Extra margin added to the LIS domain bounding box on all sides
        (degrees).  Ensures tiles that only partially overlap the domain
        edge are not accidentally excluded.

    Returns
    -------
    list[str]
        Subset of *paths* that overlap (or are within *buffer_deg* of) the
        LIS domain.
    """
    import re

    lon_min, lat_min, lon_max, lat_max = _lis_lonlat_bounds(area)
    lon_min -= buffer_deg
    lat_min -= buffer_deg
    lon_max += buffer_deg
    lat_max += buffer_deg

    log.info(
        "LIS domain (lon/lat, +%.1f° buffer): %.2f–%.2f lon, %.2f–%.2f lat",
        buffer_deg,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    )

    _hv_re = re.compile(r"\.h(\d{2})v(\d{2})\.")
    kept: list[str] = []
    skipped = 0
    for path in paths:
        m = _hv_re.search(path)
        if m is None:
            log.warning("Cannot parse hXXvYY from path, including by default: %s", path)
            kept.append(path)
            continue
        h, v = int(m.group(1)), int(m.group(2))
        t_lon_min, t_lat_min, t_lon_max, t_lat_max = _sin_tile_lonlat_bbox(h, v)
        # Tiles near the antimeridian or poles can produce lon spans > 180°
        # (i.e. the tile wraps).  Treat such tiles conservatively: if their
        # lon span exceeds 180° they are not genuinely within any compact
        # mid-latitude domain, so skip them.
        lon_span = t_lon_max - t_lon_min
        if lon_span > 180.0:
            skipped += 1
            log.debug(
                "Skipping tile h%02dv%02d (antimeridian-crossing, lon span %.0f°): %s",
                h,
                v,
                lon_span,
                path,
            )
            continue
        overlaps = (
            t_lon_max >= lon_min
            and t_lon_min <= lon_max
            and t_lat_max >= lat_min
            and t_lat_min <= lat_max
        )
        if overlaps:
            kept.append(path)
        else:
            skipped += 1
            log.debug("Skipping tile h%02dv%02d (outside domain): %s", h, v, path)

    log.info(
        "Tile filter: %d / %d tiles overlap the LIS domain (%d skipped)",
        len(kept),
        len(paths),
        skipped,
    )
    return kept


# MODIS SIN grid constants (shared by multiple functions)
_SIN_R = 6371007.181  # sphere radius (metres)
_SIN_TILE_W = 2 * np.pi * _SIN_R / 36  # tile width  ≈ 1,111,950 m
_SIN_TILE_H = np.pi * _SIN_R / 18  # tile height ≈ 1,111,950 m
_SIN_PIXELS_PER_TILE = 3000  # pixels per tile per axis (375 m product)
_SIN_PIXEL_SIZE = _SIN_TILE_W / _SIN_PIXELS_PER_TILE  # ≈ 370.65 m


def build_viirs_domain_mapping(
    bbox: tuple[float, float, float, float],
    buffer_deg: float = DOMAIN_FILTER_BUFFER_DEG,
) -> xr.Dataset:
    """Build a VIIRS pixel mapping dataset for a spatial domain.

    Computes every VIIRS pixel that falls within *bbox* (with *buffer_deg*
    margin) purely from the MODIS SIN grid geometry — no file I/O required.

    The result is an ``xr.Dataset`` laid out as a 2-D composite grid in SIN
    projection.  Rows correspond to tile rows (v-index, top-to-bottom); columns
    to tile columns (h-index, left-to-right).  Within each tile block the
    3000×3000 pixel indices are unrolled in row-major order.

    Parameters
    ----------
    bbox:
        ``(lon_min, lat_min, lon_max, lat_max)`` in degrees.  The region of
        interest.  A *buffer_deg* margin is added before tile selection.
    buffer_deg:
        Extra margin (degrees) added to *bbox* before selecting tiles.

    Returns
    -------
    xr.Dataset
        Dimensions: ``y`` (rows in composite grid), ``x`` (columns).
        Coordinates: ``sin_x`` (metres), ``sin_y`` (metres) — SIN projection
        coordinates of each pixel centre; ``lon``, ``lat`` (degrees).
        Data variables:
          - ``tile``    : str, ``"hHHvVV"`` tile identifier
          - ``tile_xi`` : int16, pixel column index within the tile (0–2999)
          - ``tile_yi`` : int16, pixel row index within the tile (0–2999)

        Only pixels whose (lon, lat) fall within *bbox* (+ buffer) are
        included; pixels outside the domain are masked (NaN / empty string).
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lon_min -= buffer_deg
    lat_min -= buffer_deg
    lon_max += buffer_deg
    lat_max += buffer_deg

    # ------------------------------------------------------------------
    # Step 1: find which (h, v) tile indices overlap the buffered bbox.
    # Convert bbox lon/lat corners to SIN metres to determine the range.
    # ------------------------------------------------------------------
    to_sin = pyproj.Transformer.from_crs("EPSG:4326", _SIN_CRS, always_xy=True)
    to_geo = pyproj.Transformer.from_crs(_SIN_CRS, "EPSG:4326", always_xy=True)

    # Sample the bbox boundary densely so curved projections don't cause gaps
    n_samples = 100
    edge_lons = np.concatenate(
        [
            np.linspace(lon_min, lon_max, n_samples),  # south edge
            np.full(n_samples, lon_max),  # east edge
            np.linspace(lon_max, lon_min, n_samples),  # north edge
            np.full(n_samples, lon_min),  # west edge
        ]
    )
    edge_lats = np.concatenate(
        [
            np.full(n_samples, lat_min),
            np.linspace(lat_min, lat_max, n_samples),
            np.full(n_samples, lat_max),
            np.linspace(lat_max, lat_min, n_samples),
        ]
    )
    edge_x, edge_y = to_sin.transform(edge_lons, edge_lats)

    h_vals = np.floor(edge_x / _SIN_TILE_W + 18).astype(int)
    v_vals = np.floor(9 - edge_y / _SIN_TILE_H).astype(int)
    h_min, h_max = int(h_vals.min()), int(h_vals.max())
    v_min, v_max = int(v_vals.min()), int(v_vals.max())

    # Clamp to valid MODIS tile range
    h_min = max(0, h_min)
    h_max = min(35, h_max)
    v_min = max(0, v_min)
    v_max = min(17, v_max)

    n_tiles_x = h_max - h_min + 1
    n_tiles_y = v_max - v_min + 1
    n = _SIN_PIXELS_PER_TILE
    total_x = n_tiles_x * n
    total_y = n_tiles_y * n

    log.info(
        "Domain mapping: tiles h%02d–h%02d × v%02d–v%02d (%d×%d tiles, %d×%d pixels)",
        h_min,
        h_max,
        v_min,
        v_max,
        n_tiles_x,
        n_tiles_y,
        total_x,
        total_y,
    )

    # ------------------------------------------------------------------
    # Step 2: build full composite pixel coordinate arrays (no file I/O).
    # ------------------------------------------------------------------
    # 1-D SIN coordinate vectors for the composite grid
    # x increases left→right; y increases top→bottom (v increases southward)
    x_tile_origins = np.array([(h - 18) * _SIN_TILE_W for h in range(h_min, h_max + 1)])
    y_tile_origins = np.array(
        [(9 - v - 1) * _SIN_TILE_H for v in range(v_min, v_max + 1)]
    )

    # Pixel-centre offsets within a tile (0.5, 1.5, … 2999.5) × pixel_size
    px_offsets = (np.arange(n) + 0.5) * _SIN_PIXEL_SIZE

    # Full 1-D coordinate vectors across the composite grid
    sin_x_1d = np.concatenate([orig + px_offsets for orig in x_tile_origins])
    # y_tile_origins are in SIN (northward-positive); flip so row 0 = northernmost
    sin_y_1d = np.concatenate(
        [orig + px_offsets[::-1] for orig in y_tile_origins[::-1]]
    )

    # 2-D grids of SIN coordinates
    sin_x_2d, sin_y_2d = np.meshgrid(sin_x_1d, sin_y_1d)
    lon_2d, lat_2d = to_geo.transform(sin_x_2d, sin_y_2d)

    # ------------------------------------------------------------------
    # Step 3: compute tile name and within-tile pixel indices for every pixel.
    # ------------------------------------------------------------------
    # Global tile column / row for each composite pixel
    tile_col = np.repeat(np.arange(n_tiles_x), n)  # shape (total_x,)
    tile_row = np.repeat(np.arange(n_tiles_y), n)  # shape (total_y,)
    tile_col_2d = np.broadcast_to(tile_col[np.newaxis, :], (total_y, total_x))
    tile_row_2d = np.broadcast_to(tile_row[:, np.newaxis], (total_y, total_x))

    h_2d = (h_min + tile_col_2d).astype(np.int16)
    v_2d = (v_min + tile_row_2d).astype(np.int16)

    # Within-tile pixel indices (xi = column within tile, yi = row within tile)
    xi_local = np.tile(np.arange(n, dtype=np.int16), n_tiles_x)  # (total_x,)
    yi_local = np.tile(np.arange(n, dtype=np.int16), n_tiles_y)  # (total_y,)
    xi_2d = np.broadcast_to(xi_local[np.newaxis, :], (total_y, total_x))
    yi_2d = np.broadcast_to(yi_local[:, np.newaxis], (total_y, total_x))

    # Tile name strings: "hHHvVV"
    tile_names = np.array(
        [f"h{h:02d}v{v:02d}" for v, h in np.ndindex(n_tiles_y, n_tiles_x)],
        dtype="U8",
    ).reshape(n_tiles_y, n_tiles_x)
    tile_block = np.repeat(np.repeat(tile_names, n, axis=0), n, axis=1)

    # ------------------------------------------------------------------
    # Step 4: mask pixels outside the bbox.
    # ------------------------------------------------------------------
    in_domain = (
        (lon_2d >= lon_min)
        & (lon_2d <= lon_max)
        & (lat_2d >= lat_min)
        & (lat_2d <= lat_max)
    )

    ds = xr.Dataset(
        {
            "tile": (["y", "x"], np.where(in_domain, tile_block, "")),
            "tile_xi": (["y", "x"], np.where(in_domain, xi_2d, -1).astype(np.int16)),
            "tile_yi": (["y", "x"], np.where(in_domain, yi_2d, -1).astype(np.int16)),
        },
        coords={
            "sin_x": (["x"], sin_x_1d),
            "sin_y": (["y"], sin_y_1d),
            "lon": (["y", "x"], lon_2d.astype(np.float32)),
            "lat": (["y", "x"], lat_2d.astype(np.float32)),
        },
        attrs={
            "bbox_lon_min": lon_min,
            "bbox_lat_min": lat_min,
            "bbox_lon_max": lon_max,
            "bbox_lat_max": lat_max,
            "buffer_deg": buffer_deg,
            "h_min": h_min,
            "h_max": h_max,
            "v_min": v_min,
            "v_max": v_max,
        },
    )
    n_valid = int(in_domain.sum())
    log.info("Domain mapping: %d / %d pixels within domain", n_valid, total_x * total_y)
    return ds


def _sin_to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert MODIS SIN projection coordinates (metres) to lon/lat (degrees).

    x and y can be 1-D (coordinate vectors) or 2-D (pixel centres).
    Returns (lon, lat) arrays of the same shape.
    """
    transformer = pyproj.Transformer.from_crs(_SIN_CRS, "EPSG:4326", always_xy=True)
    return transformer.transform(x, y)


def load_viirs_tile(path: str, storage: StorageConfig) -> dict:
    """Load one VIIRS HDF5 tile and return a dict with keys:

    - ``data``: float32 (ny, nx) array, flag values (>100) masked to NaN
    - ``lon2d``: float64 (ny, nx) array of pixel-centre longitudes
    - ``lat2d``: float64 (ny, nx) array of pixel-centre latitudes
    - ``stem``: str, file stem used for output naming

    The sinusoidal tile XDim/YDim coordinates (in metres) are broadcast to a
    2-D meshgrid and transformed to geographic coordinates via pyproj.
    """
    log.info("Loading VIIRS tile %s", path)
    stem = path.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    with storage.open(path) as fobj:
        with h5py.File(fobj, "r") as f:
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
        "stem": stem,
    }


def load_viirs_tiles_subset(
    mapping: xr.Dataset,
    paths: list[str],
    storage: StorageConfig,
) -> dict:
    """Load VIIRS tile data for only the pixels that fall within the domain.

    Uses the pre-computed *mapping* (from :func:`build_viirs_domain_mapping`)
    to determine exactly which pixels to read from each tile.  No lon/lat
    computation is needed — coordinates come directly from the mapping.

    For each tile in *paths* the function:

    1. Looks up all mapping pixels whose ``tile`` variable matches the tile's
       ``hXXvYY`` identifier.
    2. Reads those pixels from the HDF5 by integer-indexing into ``raw[yi, xi]``
       — so only domain pixels are loaded from S3, no full 3000×3000 array is
       kept in memory.
    3. Collects the corresponding lon/lat values from the mapping.

    The result is a flat ``(N, 1)`` shaped dict suitable for passing to satpy
    resamplers (which require at least 2-D arrays).

    Parameters
    ----------
    mapping:
        Dataset returned by :func:`build_viirs_domain_mapping`.
    paths:
        Paths to VIIRS HDF5 tiles (may be a subset of all tiles).
    storage:
        Storage configuration for file access.

    Returns
    -------
    dict with keys:

    - ``data``  : float32 (N, 1) — VIIRS data values for domain pixels
    - ``lon2d`` : float64 (N, 1) — pixel-centre longitudes
    - ``lat2d`` : float64 (N, 1) — pixel-centre latitudes
    - ``stem``  : str, derived from the first tile's date component
    """
    import re as _re

    _hv_re = _re.compile(r"\.(h\d{2}v\d{2})\.")

    # Pre-flatten mapping arrays once — avoids repeated 2-D indexing
    tile_flat = mapping["tile"].values.ravel()  # (total_y * total_x,)
    xi_flat = mapping["tile_xi"].values.ravel()  # int16
    yi_flat = mapping["tile_yi"].values.ravel()  # int16
    lon_flat = mapping.coords["lon"].values.ravel()  # float32
    lat_flat = mapping.coords["lat"].values.ravel()  # float32

    all_data: list[np.ndarray] = []
    all_lon: list[np.ndarray] = []
    all_lat: list[np.ndarray] = []
    stem: str = ""

    for path in paths:
        m = _hv_re.search(path)
        if m is None:
            log.warning("Cannot parse hXXvYY from %s; skipping", path)
            continue
        hv = m.group(1)  # e.g. "h10v04"

        # Find mapping pixels belonging to this tile
        mask = tile_flat == hv
        if not mask.any():
            log.debug("No domain pixels for tile %s; skipping", hv)
            continue

        xi = xi_flat[mask].astype(np.intp)
        yi = yi_flat[mask].astype(np.intp)

        log.info("Loading VIIRS tile %s (%d domain pixels)", path, len(xi))
        if not stem:
            stem = path.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0]

        with storage.open(path) as fobj:
            with h5py.File(fobj, "r") as f:
                raw = f[_HDFEOS_DATA_PATH][yi, xi]  # uint8 (N_tile,)

        data = raw.astype(np.float32)
        data[data > 100] = np.nan

        all_data.append(data)
        all_lon.append(lon_flat[mask].astype(np.float64))
        all_lat.append(lat_flat[mask].astype(np.float64))

    if not all_data:
        raise ValueError(
            "No domain pixels found in any of the provided tile paths. "
            "Check that the mapping bbox overlaps the tile set."
        )

    data_1d = np.concatenate(all_data)  # (N,)
    lon_1d = np.concatenate(all_lon)  # (N,)
    lat_1d = np.concatenate(all_lat)  # (N,)

    # Reshape to (N, 1) so satpy resamplers (which expect 2-D dims) work
    return {
        "data": data_1d[:, np.newaxis],
        "lon2d": lon_1d[:, np.newaxis],
        "lat2d": lat_1d[:, np.newaxis],
        "stem": stem,
    }


def load_viirs_tiles(paths: list[str], storage: StorageConfig) -> dict:
    """Load and composite multiple VIIRS tiles into a single swath.

    Tiles are stacked along the row axis (axis 0).  This produces one large
    SwathDefinition covering all provided tiles so that a single regridding
    pass serves the full LIS domain.

    Returns the same dict structure as ``load_viirs_tile`` with a composite
    ``stem`` derived from the first tile's date component.
    """
    if len(paths) == 1:
        return load_viirs_tile(paths[0], storage)

    tiles = [load_viirs_tile(p, storage) for p in paths]
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
    """Regrid using DaskEWAResampler (Elliptical Weighted Averaging).

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
    parser = argparse.ArgumentParser(
        description="Regrid VIIRS CGF Snow Cover files to the LIS input grid."
    )
    add_storage_args(parser)
    ns = parser.parse_args()
    storage = storage_config_from_namespace(ns)

    viirs_files = storage.glob(VIIRS_GLOB)
    if not viirs_files:
        raise FileNotFoundError(
            f"No VIIRS files found matching '{VIIRS_GLOB}' in {storage.storage_location}"
        )
    log.info("Found %d VIIRS file(s)", len(viirs_files))

    lis_area = build_lis_area_definition(storage)
    lis_lons, lis_lats = lis_area.get_lonlats()

    log.info("Filtering tiles to LIS domain …")
    viirs_files = filter_tiles_by_domain(viirs_files, lis_area)
    if not viirs_files:
        raise FileNotFoundError("No VIIRS tiles overlap the LIS domain.")

    # Group files by date (YYYYDDD encoded in filename, e.g. A2019001)
    date_groups: dict[str, list[str]] = defaultdict(list)
    for p in viirs_files:
        stem = p.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        parts = stem.split(".")
        date_key = parts[1] if len(parts) > 1 else stem
        date_groups[date_key].append(p)

    lis_area = build_lis_area_definition(storage)
    lis_lons, lis_lats = lis_area.get_lonlats()

    for date_key, paths in sorted(date_groups.items()):
        log.info("Processing date %s (%d tile(s))", date_key, len(paths))
        tile = load_viirs_tiles(paths, storage)
        source_def = build_viirs_swath_definition(tile)

        out_path = regrid_tile_to_nc(
            tile, source_def, lis_area, lis_lons, lis_lats, _DATA_OUT
        )
        log.info("Done: %s", out_path)

    log.info("All VIIRS tiles regridded successfully.")


if __name__ == "__main__":
    main()
