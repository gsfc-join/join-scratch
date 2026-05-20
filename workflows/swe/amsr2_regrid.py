#!/usr/bin/env python
"""Regrid AMSR2 L3 snow depth files to the LIS input grid.

Workflow
--------
1. Create (or open) a preallocated IceChunk store on S3 containing virtual
   HTTPS references to JAXA G-Portal HDF5 files.
2. Compute (or reuse) ESMF bilinear regridding weights mapping the global
   0.1° AMSR2 equirectangular grid to the LIS 1 km Lambert Conformal grid.
3. For each requested date, read the two orbit variables and two band variables
   from the IceChunk store, apply a NaN mask to fill-valued pixels, apply the
   precomputed sparse weight matrix, and write a daily output NetCDF that
   follows CF conventions.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import icechunk
import numpy as np
import obstore
import pandas as pd
import scipy.sparse
import xarray as xr
import zarr
from obstore.store import HTTPStore
from zarr.codecs import Zlib
from virtualizarr.parsers.hdf.hdf import _construct_manifest_array

# Must be set BEFORE importing icechunk (Rust logger initialised at import time).
os.environ.setdefault("RUST_LOG", "error")

sys.path.insert(0, str(Path(__file__).parent))

from lis_grid import build_lis_area_definition, load_lis_grid
from s3_utils import make_fs, make_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
S3_BUCKET = "airborne-smce-prod-user-bucket"
S3_JOIN_PREFIX = "JOIN"

DEFAULT_LIS_PATH = f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/lis_input_NMP_1000m_missouri.nc"
DEFAULT_STORE_URI = (
    f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/icechunk-stores/GCOM-W1-AMSR2-L3-SND"
)
DEFAULT_WEIGHTS_URI = (
    f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/cached-weights"
    "/GCOM-W1-AMSR2-L3-SND/lis-1km-missouri.nc4"
)
DEFAULT_OUTPUT_PREFIX = f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/outputs"

GPORTAL_BASE = (
    "https://gportal.jaxa.jp/download/standard/GCOM-W/GCOM-W.AMSR2"
    "/L3.SND_10/2/{yyyy}/{mm}/"
)
FNAME_TMPL = "GW1AM2_{date}_01D_{orbit}_L3SGSNDHG2210210.h5"
ORBIT_CODES = ["EQMA", "EQMD"]
ORBIT_LABELS = ["Ascending", "Descending"]

NLAT, NLON = 1800, 3600
CLAT = 72  # lat chunk size

FILL_MISSING = np.int16(-32768)
FILL_NOOBS = np.int16(-32767)
SCALE_FACTOR = 0.1

TIME_UNITS = "days since 1970-01-01"
TIME_CALENDAR = "proleptic_gregorian"
TIME_EPOCH = pd.Timestamp("1970-01-01")

PREALLOC_START = pd.Timestamp("2012-06-01")
PREALLOC_END = pd.Timestamp("2030-12-31")

_LATS = np.linspace(89.95, -89.95, NLAT).astype("float32")
_LONS = np.linspace(0.05, 359.95, NLON).astype("float32")

# Pre-build day-value → slot index mapping once at module level.
_prealloc_days = pd.date_range(PREALLOC_START, PREALLOC_END, freq="D")
_PREALLOC_DAY_VALS = ((_prealloc_days - TIME_EPOCH).days.astype("int32"))
_PREALLOC_NDAYS = len(_PREALLOC_DAY_VALS)
_DAY_TO_SLOT: dict[int, int] = {
    int(d): i for i, d in enumerate(_PREALLOC_DAY_VALS)
}

# ── helper functions ───────────────────────────────────────────────────────────


def gportal_url(date: str, orbit: str) -> str:
    """Build the G-Portal HTTPS URL for one AMSR2 file.

    Parameters
    ----------
    date:
        Date string in ``YYYYMMDD`` format.
    orbit:
        Orbit code, one of ``"EQMA"`` (ascending) or ``"EQMD"`` (descending).
    """
    yyyy, mm = date[:4], date[4:6]
    return GPORTAL_BASE.format(yyyy=yyyy, mm=mm) + FNAME_TMPL.format(
        date=date, orbit=orbit
    )


def date_to_days(date: str) -> int:
    """Convert a ``YYYYMMDD`` string to integer days since 1970-01-01."""
    ts = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:]}")
    return int((ts - TIME_EPOCH).days)


def _av(v):
    """Normalise an HDF5 attribute value to a plain Python scalar or list."""
    if isinstance(v, np.ndarray):
        flat = v.flatten()
        if flat.dtype.kind in ("S", "U", "O"):
            items = [x.decode() if isinstance(x, bytes) else str(x) for x in flat]
            return items[0] if len(items) == 1 else items
        lst = flat.tolist()
        return lst[0] if len(lst) == 1 else lst
    return v.decode() if isinstance(v, bytes) else v


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/prefix`` into ``(bucket, prefix)``."""
    without_scheme = uri[len("s3://"):]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix


def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")


# ── icechunk store management ─────────────────────────────────────────────────


def _make_repo_config() -> icechunk.RepositoryConfig:
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "https://gportal.jaxa.jp/", icechunk.http_store()
        )
    )
    return config


def _s3_storage(bucket: str, prefix: str) -> icechunk.Storage:
    return icechunk.s3_storage(
        bucket=bucket,
        prefix=prefix,
        region="us-west-2",
    )


def _store_exists(bucket: str, prefix: str) -> bool:
    """Return True if the IceChunk store already has objects on S3."""
    s3 = make_store(bucket, prefix=prefix)
    pages = list(s3.list(None))
    keys = [obj["path"] for page in pages for obj in page]
    return len(keys) > 0


def create_icechunk_store(bucket: str, prefix: str) -> None:
    """Create a preallocated IceChunk store on S3."""
    log.info(
        "Creating IceChunk store at s3://%s/%s (%d time slots, %s → %s)",
        bucket,
        prefix,
        _PREALLOC_NDAYS,
        PREALLOC_START.date(),
        PREALLOC_END.date(),
    )
    storage = _s3_storage(bucket, prefix)
    config = _make_repo_config()
    repo = icechunk.Repository.create(storage, config=config)

    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="w", zarr_format=3)

    root.require_array(
        "time",
        shape=(_PREALLOC_NDAYS,),
        chunks=(512,),
        dtype="int32",
        dimension_names=["time"],
        attributes={
            "units": TIME_UNITS,
            "calendar": TIME_CALENDAR,
            "long_name": "observation date",
        },
    )
    root["time"][:] = _PREALLOC_DAY_VALS

    root.require_array(
        "orbit",
        shape=(2,),
        chunks=(2,),
        dtype=str,
        dimension_names=["orbit"],
        attributes={"long_name": "orbit direction"},
    )
    root["orbit"][:] = ORBIT_LABELS

    root.require_array(
        "band",
        shape=(2,),
        chunks=(2,),
        dtype=str,
        dimension_names=["band"],
        attributes={"long_name": "band name"},
    )
    root["band"][:] = ["snow_depth", "quality_flag"]

    root.require_array(
        "lat",
        shape=(NLAT,),
        chunks=(NLAT,),
        dtype="float32",
        dimension_names=["lat"],
        attributes={"units": "degrees_north", "long_name": "latitude"},
    )
    root["lat"][:] = _LATS

    root.require_array(
        "lon",
        shape=(NLON,),
        chunks=(NLON,),
        dtype="float32",
        dimension_names=["lon"],
        attributes={"units": "degrees_east", "long_name": "longitude"},
    )
    root["lon"][:] = _LONS

    root.require_array(
        "geophysical_data",
        shape=(_PREALLOC_NDAYS, 2, NLAT, NLON, 2),
        chunks=(1, 1, CLAT, NLON, 2),
        dtype="int16",
        fill_value=int(FILL_MISSING),
        compressors=[Zlib(level=9)],
        dimension_names=["time", "orbit", "lat", "lon", "band"],
        attributes={
            "long_name": "geophysical data",
            "units": "cm",
            "scale_factor": SCALE_FACTOR,
            "_FillValue": int(FILL_MISSING),
            "missing_value": int(FILL_NOOBS),
        },
    )

    session.commit("Initialise preallocated store")
    log.info("IceChunk store created successfully.")


def open_icechunk_repo(bucket: str, prefix: str) -> icechunk.Repository:
    """Open an existing IceChunk store for reading or writing."""
    return icechunk.Repository.open(
        _s3_storage(bucket, prefix),
        authorize_virtual_chunk_access={"https://gportal.jaxa.jp/": None},
    )


def _get_manifests(chunk_url: str):
    """Download one HDF5 file and extract its VirtualiZarr manifest."""
    from urllib.parse import urlsplit

    parts = urlsplit(chunk_url)
    base_url = f"{parts.scheme}://{parts.netloc}"
    obj_path = parts.path.lstrip("/")
    store = HTTPStore.from_url(base_url)
    buf = io.BytesIO(obstore.get(store, obj_path).bytes())
    with h5py.File(buf, "r") as f:
        ma = _construct_manifest_array(chunk_url, f["Geophysical Data"], "/")
        attrs = {k: _av(v) for k, v in f["Geophysical Data"].attrs.items()}
    return ma.manifest, attrs


def _slot_has_virtual_refs(session_store, slot: int) -> bool:
    """Return True if the given time slot already has virtual refs written."""
    # Check for the presence of the first expected chunk key (orbit 0, lat chunk 0)
    test_key = f"geophysical_data/c/{slot}/0/0/0/0"
    try:
        ref = session_store.get_virtual_ref(test_key)
        return ref is not None
    except Exception:
        return False


def insert_date(bucket: str, prefix: str, date: str) -> None:
    """Insert virtual refs for one date into the preallocated IceChunk store.

    Opens a fresh repo on every call so we always see the latest snapshot and
    avoid ``ConflictError`` from stale parent references.  If virtual refs for
    this date already exist, this is a no-op.
    """
    day_val = date_to_days(date)
    if day_val not in _DAY_TO_SLOT:
        log.warning("Date %s is outside the preallocated range — skipping.", date)
        return
    slot = _DAY_TO_SLOT[day_val]

    # Always open a fresh repo so we start from the latest committed snapshot.
    repo = open_icechunk_repo(bucket, prefix)
    session = repo.writable_session("main")
    store = session.store

    if _slot_has_virtual_refs(store, slot):
        log.info("Slot %d (%s) already populated — skipping.", slot, date)
        return

    log.info("Fetching manifests for %s and writing to slot %d …", date, slot)
    any_written = False
    for o_idx, orbit in enumerate(ORBIT_CODES):
        url = gportal_url(date, orbit)
        try:
            manifest, _ = _get_manifests(url)
        except Exception as exc:
            log.warning("Could not fetch manifest for %s %s: %s", date, orbit, exc)
            continue
        for (ci, cj, ck), ref in manifest.iter_refs():
            key = f"geophysical_data/c/{slot}/{o_idx}/{ci}/{cj}/{ck}"
            store.set_virtual_ref(
                key, ref["path"], offset=ref["offset"], length=ref["length"]
            )
        any_written = True
        log.debug("  wrote virtual refs for orbit %s (slot %d)", orbit, slot)

    if any_written:
        session.commit(f"Fill slot {slot} ({date})")
        log.info("Committed slot %d (%s).", slot, date)
    else:
        log.warning("No manifests written for %s — nothing committed.", date)


# ── ESMF weight generation ────────────────────────────────────────────────────


def _write_scrip_grid(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    mask2d: np.ndarray | None,
    path: Path,
    title: str = "grid",
) -> None:
    """Write a 2-D curvilinear grid in SCRIP NetCDF format."""
    import netCDF4 as nc  # noqa: N813

    ny, nx = lat2d.shape
    ncells = ny * nx

    lat_flat = lat2d.flatten()
    lon_flat = lon2d.flatten()

    # Compute corner coordinates (simple ±0.5 cell approximation for regular grids)
    # Corner order: SW, SE, NE, NW (counter-clockwise)
    dlat = np.gradient(lat2d, axis=0)
    dlon = np.gradient(lon2d, axis=1)

    lat_corners = np.stack(
        [
            lat_flat - dlat.flatten() / 2,
            lat_flat - dlat.flatten() / 2,
            lat_flat + dlat.flatten() / 2,
            lat_flat + dlat.flatten() / 2,
        ],
        axis=0,
    )
    lon_corners = np.stack(
        [
            lon_flat - dlon.flatten() / 2,
            lon_flat + dlon.flatten() / 2,
            lon_flat + dlon.flatten() / 2,
            lon_flat - dlon.flatten() / 2,
        ],
        axis=0,
    )

    with nc.Dataset(path, "w") as ds:
        ds.createDimension("grid_size", ncells)
        ds.createDimension("grid_corners", 4)
        ds.createDimension("grid_rank", 2)

        grid_dims = ds.createVariable("grid_dims", "i4", ("grid_rank",))
        grid_dims[:] = [nx, ny]

        grid_center_lat = ds.createVariable(
            "grid_center_lat", "f8", ("grid_size",), fill_value=False
        )
        grid_center_lat.units = "degrees"
        grid_center_lat[:] = lat_flat

        grid_center_lon = ds.createVariable(
            "grid_center_lon", "f8", ("grid_size",), fill_value=False
        )
        grid_center_lon.units = "degrees"
        grid_center_lon[:] = lon_flat

        grid_corner_lat = ds.createVariable(
            "grid_corner_lat", "f8", ("grid_size", "grid_corners"), fill_value=False
        )
        grid_corner_lat.units = "degrees"
        grid_corner_lat[:] = lat_corners.T

        grid_corner_lon = ds.createVariable(
            "grid_corner_lon", "f8", ("grid_size", "grid_corners"), fill_value=False
        )
        grid_corner_lon.units = "degrees"
        grid_corner_lon[:] = lon_corners.T

        grid_imask = ds.createVariable("grid_imask", "i4", ("grid_size",))
        grid_imask.units = "unitless"
        if mask2d is not None:
            grid_imask[:] = mask2d.flatten().astype("i4")
        else:
            grid_imask[:] = np.ones(ncells, dtype="i4")


def build_esmf_weights(
    lis_path: str,
    weights_uri: str,
    tmp_dir: Path,
    recreate: bool = False,
) -> Path:
    """Compute ESMF bilinear weights from AMSR2 → LIS grid.

    The weights are cached at *weights_uri* (S3 or local).  If the file already
    exists and *recreate* is False, it is downloaded and returned immediately.

    Parameters
    ----------
    lis_path:
        Local path or ``s3://`` URI to the LIS NetCDF file.
    weights_uri:
        Destination URI for the weights NetCDF4 file (S3 or local).
    tmp_dir:
        Scratch directory for SCRIP grid files and the ESMF output.
    recreate:
        If True, ignore any existing weights file and regenerate from scratch.

    Returns
    -------
    Local ``Path`` to the downloaded/generated weights file.
    """
    local_weights = tmp_dir / "esmf-weights.nc4"

    # -- try to reuse existing weights -----------------------------------------
    if not recreate and _is_s3(weights_uri):
        bucket, prefix = _parse_s3_uri(weights_uri)
        s3 = make_store(bucket, prefix=str(Path(prefix).parent))
        fname = Path(prefix).name
        try:
            pages = list(s3.list(None))
            keys = [obj["path"] for page in pages for obj in page]
            if fname in keys or prefix in keys or any(k.endswith(fname) for k in keys):
                log.info("Downloading existing weights from %s …", weights_uri)
                fs = make_fs()
                with fs.open(weights_uri) as fobj:
                    local_weights.write_bytes(fobj.read())
                log.info("Weights downloaded to %s.", local_weights)
                return local_weights
        except Exception as exc:
            log.warning("Could not check/download weights: %s — regenerating.", exc)
    elif not recreate and not _is_s3(weights_uri) and Path(weights_uri).exists():
        log.info("Using existing local weights file %s.", weights_uri)
        return Path(weights_uri)

    # -- generate weights with ESMF_RegridWeightGen ----------------------------
    log.info("Building AMSR2 → LIS ESMF bilinear weights …")

    # Source: AMSR2 global 0.1° equirectangular grid (lon 0..360)
    src_lats = np.linspace(89.95, -89.95, NLAT)
    src_lons = np.linspace(0.05, 359.95, NLON)
    src_lon2d, src_lat2d = np.meshgrid(src_lons, src_lats)
    src_grid = tmp_dir / "amsr2-scrip.nc"
    log.debug("Writing AMSR2 SCRIP grid to %s …", src_grid)
    _write_scrip_grid(src_lat2d, src_lon2d, None, src_grid, title="AMSR2 0.1deg")

    # Destination: LIS Lambert Conformal grid (lat/lon 2-D)
    log.info("Loading LIS grid for SCRIP destination grid …")
    lis_ds = load_lis_grid(lis_path)
    # load_lis_grid returns north-first; ESMF is fine with that.
    dst_lat2d = lis_ds["lat"].values
    dst_lon2d = lis_ds["lon"].values
    dst_grid = tmp_dir / "lis-scrip.nc"
    log.debug("Writing LIS SCRIP grid to %s …", dst_grid)
    _write_scrip_grid(dst_lat2d, dst_lon2d, None, dst_grid, title="LIS 1km LCC")

    esmf_out = tmp_dir / "esmf-weights-raw.nc"
    cmd = [
        "ESMF_RegridWeightGen",
        "--source",
        str(src_grid),
        "--destination",
        str(dst_grid),
        "--weight",
        str(esmf_out),
        "--method",
        "bilinear",
        "--src_type",
        "SCRIP",
        "--dst_type",
        "SCRIP",
        "--ignore_unmapped",
    ]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ESMF_RegridWeightGen stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"ESMF_RegridWeightGen failed (exit {result.returncode})"
        )
    log.info("ESMF weight generation complete.")
    if result.stdout.strip():
        log.debug("ESMF stdout:\n%s", result.stdout)

    # Copy raw ESMF output to local_weights path
    import shutil
    shutil.copy2(esmf_out, local_weights)

    # -- upload to S3 if requested --------------------------------------------
    if _is_s3(weights_uri):
        log.info("Uploading weights to %s …", weights_uri)
        bucket, key = _parse_s3_uri(weights_uri)
        s3_store = make_store(bucket)
        with open(local_weights, "rb") as fobj:
            data = fobj.read()
        obstore.put(s3_store, key, data)
        log.info("Weights uploaded.")

    return local_weights


def load_weights(weights_path: Path) -> scipy.sparse.csr_matrix:
    """Load ESMF weight file and return a CSR sparse matrix.

    The ESMF weight file stores the weight triplets as ``S`` (values),
    ``row`` (1-based destination indices), ``col`` (1-based source indices).
    """
    import netCDF4 as nc  # noqa: N813

    with nc.Dataset(weights_path, "r") as ds:
        S = ds.variables["S"][:]
        row = ds.variables["row"][:] - 1  # convert to 0-based
        col = ds.variables["col"][:] - 1
        n_dst = int(ds.variables["dst_grid_dims"][:].prod())
        n_src = int(ds.variables["src_grid_dims"][:].prod())

    W = scipy.sparse.csr_matrix((S, (row, col)), shape=(n_dst, n_src))
    log.info(
        "Loaded weight matrix: %d src cells → %d dst cells, %d non-zeros",
        n_src,
        n_dst,
        W.nnz,
    )
    return W


# ── NaN-masked sparse regridding ──────────────────────────────────────────────


def regrid_slice(
    data: np.ndarray,
    W: scipy.sparse.csr_matrix,
    dst_shape: tuple[int, int],
    fill_values: tuple[int, int] = (int(FILL_MISSING), int(FILL_NOOBS)),
) -> np.ndarray:
    """Regrid a single 2-D source field using precomputed ESMF weights.

    Fill-valued pixels are masked to NaN before the sparse matrix multiply so
    they cannot infect neighbouring destination cells.  Destination cells with
    no valid source contribution remain NaN.

    Parameters
    ----------
    data:
        2-D array of raw int16 source values, shape ``(NLAT, NLON)``.
        Assumed layout: row 0 = lat 89.95°N, last row = lat −89.95°S (north-first),
        columns: lon 0.05°E … 359.95°E.
    W:
        CSR sparse weight matrix, shape ``(n_dst, n_src)``.
    dst_shape:
        ``(ny_lis, nx_lis)`` — shape of the destination grid.
    fill_values:
        Raw int16 values that indicate missing/no-observation; these are
        masked to NaN before regridding.

    Returns
    -------
    Float32 array of shape *dst_shape* with regridded physical values
    (i.e. raw × SCALE_FACTOR).  Cells with no valid source data are NaN.
    """
    float_data = data.astype("float32") * SCALE_FACTOR

    # Mask fill-valued pixels so they do not pollute the weighted average.
    for fv in fill_values:
        float_data[data == fv] = np.nan

    src_flat = float_data.flatten()
    valid = np.isfinite(src_flat)

    # To avoid NaN contamination in the sparse multiply, replace NaN with 0
    # and use a separate weight-sum array to detect cells with zero valid weight.
    src_clean = np.where(valid, src_flat, 0.0)
    valid_weight = np.where(valid, 1.0, 0.0)

    dst_vals = W @ src_clean
    dst_wsum = W @ valid_weight

    # Normalise by the actual sum of weights that came from valid pixels.
    with np.errstate(invalid="ignore", divide="ignore"):
        dst_norm = np.where(dst_wsum > 0, dst_vals / dst_wsum, np.nan)

    return dst_norm.reshape(dst_shape).astype("float32")


# ── output NetCDF writing ─────────────────────────────────────────────────────


def _upload_or_save(ds: xr.Dataset, output_path: str, tmp_dir: Path) -> None:
    """Write *ds* to *output_path* (S3 URI or local path)."""
    if _is_s3(output_path):
        local_out = tmp_dir / "output.nc"
        ds.to_netcdf(local_out, engine="h5netcdf")
        bucket, key = _parse_s3_uri(output_path)
        s3_store = make_store(bucket)
        log.info("Uploading output to %s …", output_path)
        with open(local_out, "rb") as fobj:
            data = fobj.read()
        obstore.put(s3_store, key, data)
        log.info("Output uploaded.")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_path, engine="h5netcdf")
        log.info("Output written to %s.", output_path)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regrid AMSR2 L3 snow depth to the LIS 1 km grid."
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="First date to process, YYYYMMDD.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Last date to process, YYYYMMDD (inclusive). Defaults to --start-date.",
    )
    parser.add_argument(
        "--lis-path",
        default=DEFAULT_LIS_PATH,
        help=f"Local path or s3:// URI to the LIS NetCDF file. Default: {DEFAULT_LIS_PATH}",
    )
    parser.add_argument(
        "--store-uri",
        default=DEFAULT_STORE_URI,
        help=f"S3 URI for the IceChunk store. Default: {DEFAULT_STORE_URI}",
    )
    parser.add_argument(
        "--weights-uri",
        default=DEFAULT_WEIGHTS_URI,
        help=f"URI for the ESMF weights file. Default: {DEFAULT_WEIGHTS_URI}",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Output NetCDF path (local or s3://). "
            f"Defaults to {DEFAULT_OUTPUT_PREFIX}/GCOM-W1-AMSR2-L3-SND-<start>-<end>.nc"
        ),
    )
    parser.add_argument(
        "--recreate-store",
        action="store_true",
        default=False,
        help="Delete and recreate the IceChunk store even if it already exists.",
    )
    parser.add_argument(
        "--recreate-weights",
        action="store_true",
        default=False,
        help="Regenerate ESMF weights even if a cached file already exists.",
    )
    ns = parser.parse_args()

    start_date = ns.start_date
    end_date = ns.end_date if ns.end_date else start_date
    log.info("Date range: %s → %s", start_date, end_date)

    # Build list of YYYYMMDD strings for all days in range.
    date_range = pd.date_range(
        pd.Timestamp(f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"),
        pd.Timestamp(f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"),
        freq="D",
    )
    dates = [d.strftime("%Y%m%d") for d in date_range]
    log.info("Processing %d date(s): %s … %s", len(dates), dates[0], dates[-1])

    output_path = ns.output_path
    if output_path is None:
        fname = f"GCOM-W1-AMSR2-L3-SND-{start_date}-{end_date}.nc"
        output_path = f"{DEFAULT_OUTPUT_PREFIX}/{fname}"
    log.info("Output will be written to: %s", output_path)

    # ── Step 1: IceChunk store ───────────────────────────────────────────────
    store_uri = ns.store_uri
    if _is_s3(store_uri):
        bucket, prefix = _parse_s3_uri(store_uri)
        store_exists = _store_exists(bucket, prefix)
    else:
        store_exists = Path(store_uri).exists()
        bucket, prefix = None, store_uri  # used below

    if ns.recreate_store and store_exists:
        log.info("--recreate-store set: deleting existing store …")
        if _is_s3(store_uri):
            s3 = make_store(bucket, prefix=prefix)
            pages = list(s3.list(None))
            keys = [obj["path"] for page in pages for obj in page]
            for key in keys:
                obstore.delete(s3, key)
            log.info("Deleted %d objects from existing store.", len(keys))
        else:
            import shutil
            shutil.rmtree(store_uri)
        store_exists = False

    if not store_exists:
        if _is_s3(store_uri):
            create_icechunk_store(bucket, prefix)
        else:
            raise NotImplementedError("Local IceChunk stores are not supported.")
    else:
        log.info("IceChunk store already exists at %s.", store_uri)

    # ── Step 2: Populate virtual refs for requested dates ───────────────────
    if not _is_s3(store_uri):
        raise NotImplementedError("Local IceChunk stores are not supported.")

    for date in dates:
        insert_date(bucket, prefix, date)

    # ── Step 3: ESMF weights ─────────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="amsr2_regrid_") as tmp_str:
        tmp_dir = Path(tmp_str)

        weights_local = build_esmf_weights(
            lis_path=ns.lis_path,
            weights_uri=ns.weights_uri,
            tmp_dir=tmp_dir,
            recreate=ns.recreate_weights,
        )
        W = load_weights(weights_local)

        # Determine destination grid shape from the LIS file.
        lis_ds = load_lis_grid(ns.lis_path)
        lis_lat = lis_ds["lat"].values  # (ny, nx)
        lis_lon = lis_ds["lon"].values
        ny_lis, nx_lis = lis_lat.shape
        log.info("LIS grid shape: %d × %d", ny_lis, nx_lis)

        # ── Step 4: Read from IceChunk and regrid ────────────────────────────
        log.info("Opening IceChunk store for reading …")
        repo_ro = open_icechunk_repo(bucket, prefix)
        session = repo_ro.readonly_session("main")
        # Open with mask_and_scale=False so we get raw int16 values and can
        # apply NaN masking ourselves in regrid_slice before the sparse multiply.
        ic_ds = xr.open_zarr(
            session.store, consolidated=False, zarr_format=3, mask_and_scale=False
        )

        # With mask_and_scale=False, the time coordinate is raw int32
        # days-since-epoch — use isel with the pre-built slot indices.
        slots = [_DAY_TO_SLOT[date_to_days(d)] for d in dates]
        ic_sub = ic_ds.isel(time=slots)

        # Band indices: 0 = snow_depth, 1 = quality_flag
        band_names = ["snow_depth", "quality_flag"]
        orbit_names = ["Ascending", "Descending"]

        # Output arrays: (time, orbit, band, ny_lis, nx_lis)
        n_times = len(dates)
        n_orbits = 2
        n_bands = 2
        out_data = np.full(
            (n_times, n_orbits, n_bands, ny_lis, nx_lis),
            np.nan,
            dtype="float32",
        )

        log.info("Regridding %d date(s) × %d orbits × %d bands …", n_times, n_orbits, n_bands)
        for t_idx, date in enumerate(dates):
            day_val = date_to_days(date)
            log.info("  Processing date %s (slot %d) …", date, _DAY_TO_SLOT.get(day_val, -1))
            for o_idx in range(n_orbits):
                for b_idx in range(n_bands):
                    # Raw int16 slice — shape (NLAT, NLON) = (1800, 3600).
                    # regrid_slice applies fill masking before the sparse multiply.
                    raw = (
                        ic_sub["geophysical_data"]
                        .isel(time=t_idx)
                        .sel(orbit=orbit_names[o_idx], band=band_names[b_idx])
                        .values
                    )
                    out_data[t_idx, o_idx, b_idx] = regrid_slice(
                        raw, W, (ny_lis, nx_lis)
                    )

        # ── Step 5: Build output Dataset ──────────────────────────────────────
        log.info("Building output Dataset …")

        # CF-compliant time coordinate
        time_vals = pd.DatetimeIndex([
            pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]}") for d in dates
        ])

        # Flatten LIS lat/lon to 1-D centre coordinates for output.
        # LIS grid is 2-D; we use the first column/row as representative.
        lis_lat_1d = lis_lat[:, 0].astype("float32")
        lis_lon_1d = lis_lon[0, :].astype("float32")

        # Determine CRS info from the LIS AreaDefinition for CF grid_mapping.
        area_def = build_lis_area_definition(ns.lis_path, cache_dir=str(tmp_dir))
        crs_wkt = area_def.crs.to_wkt()

        output_vars: dict[str, xr.DataArray] = {}
        for o_idx, orbit_label in enumerate(orbit_names):
            for b_idx, band_label in enumerate(band_names):
                var_name = f"{band_label}_{orbit_label.lower()}"
                da = xr.DataArray(
                    out_data[:, o_idx, b_idx, :, :],
                    dims=["time", "south_north", "west_east"],
                    coords={
                        "time": time_vals,
                        "south_north": np.arange(ny_lis),
                        "west_east": np.arange(nx_lis),
                        "lat": (["south_north", "west_east"], lis_lat.astype("float32")),
                        "lon": (["south_north", "west_east"], lis_lon.astype("float32")),
                    },
                    attrs={
                        "long_name": f"AMSR2 {band_label.replace('_', ' ')} ({orbit_label} orbit)",
                        "units": "cm" if band_label == "snow_depth" else "1",
                        "grid_mapping": "crs",
                        "_FillValue": np.float32(np.nan),
                        "source": "JAXA GCOM-W1 AMSR2 L3 Snow Depth (bilinear regrid to LIS 1 km LCC)",
                    },
                )
                output_vars[var_name] = da

        out_ds = xr.Dataset(
            output_vars,
            attrs={
                "Conventions": "CF-1.8",
                "title": "AMSR2 L3 Snow Depth regridded to LIS 1 km Lambert Conformal grid",
                "source_product": "JAXA GCOM-W1/AMSR2 L3 Snow Depth (L3SGSNDHG2210210)",
                "regrid_method": "ESMF bilinear",
                "prealloc_start": str(PREALLOC_START.date()),
                "prealloc_end": str(PREALLOC_END.date()),
            },
        )

        # Attach CRS as a scalar coordinate variable (CF grid_mapping convention)
        out_ds["crs"] = xr.DataArray(
            np.int32(0),
            attrs={"crs_wkt": crs_wkt, "grid_mapping_name": "lambert_conformal_conic"},
        )

        # ── Step 6: Write output ──────────────────────────────────────────────
        _upload_or_save(out_ds, output_path, tmp_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
