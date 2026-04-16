#!/usr/bin/env python
"""Regrid CEDA ESA CCI SWE files to the LIS input grid.

The CEDA ESA CCI Snow SWE product (L3C daily) uses a global 0.1° regular
lat/lon grid (1800 × 3600) — structurally identical to AMSR2.  Both xESMF
(bilinear) and the four satpy methods are supported.

Flag masking
------------
Valid SWE values are >= 0 mm.  Special flag values (< 0) encode:
  -30 = Glacier, -20 = Mountain, -10 = Water body, -1 = no_data
These are masked to NaN before regridding.  The same masking applies to swe_std.

Caching
-------
- xESMF writes a NetCDF weights file to _data/ceda/ceda-lis-weights.nc.
  Because all CEDA daily files share the same 0.1° grid, one weights file
  covers all dates.
- satpy KDTreeResampler and BilinearResampler cache their lookup structures
  as zarr files in _data/ceda/satpy-cache/.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf
from pyresample.ewa.dask_ewa import DaskEWAResampler
from pyresample.geometry import AreaDefinition, SwathDefinition
from satpy.resample.bucket import BucketAvg
from satpy.resample.kdtree import BilinearResampler, KDTreeResampler

from join_scratch.amsr2.amsr2_regrid import (
    build_lis_area_definition,
    load_lis_grid,
)
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
_DATA_OUT = _ROOT / "_data" / "ceda"

# Glob pattern and LIS file path relative to storage root
CEDA_GLOB = "JOIN/CEDA/**/*.nc"
LIS_RELPATH = "lis_input_NMP_1000m_missouri.nc"

# Local output paths (output always written to local disk)
WEIGHTS_PATH = _DATA_OUT / "ceda-lis-weights.nc"
SATPY_CACHE = _DATA_OUT / "satpy-cache"

# CEDA 0.1° grid has ~14 km diagonal at mid-latitudes; use 15 km ROI
RADIUS_OF_INFLUENCE = 15_000.0

# Variables to regrid
CEDA_VARS = ["swe", "swe_std"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ceda(path: str, storage: StorageConfig) -> xr.Dataset:
    """Load one CEDA SWE NetCDF file, squeeze the time dimension, and mask flags.

    Flag values (< 0) are masked to NaN in both swe and swe_std.
    Returns a Dataset with lat/lon coordinates and the two SWE variables as
    2-D arrays of shape (lat, lon).
    """
    log.info("Loading CEDA file %s", path)
    with storage.open(path) as f:
        ds = xr.open_dataset(f)
        # Drop the size-1 time dimension
        ds = ds.squeeze("time", drop=True)
        # Keep only the variables we need
        ds = ds[CEDA_VARS + ["lat", "lon"]]
        # Mask flag values (< 0 are special flags, not physical values)
        for var in CEDA_VARS:
            ds[var] = ds[var].where(ds[var] >= 0)
        ds.load()
    return ds


# ---------------------------------------------------------------------------
# Geometry builder (SwathDefinition for satpy methods)
# ---------------------------------------------------------------------------


def build_ceda_swath_definition(ds: xr.Dataset) -> SwathDefinition:
    """Build a pyresample SwathDefinition from a CEDA xarray Dataset.

    The dataset has 1-D lat/lon; these are broadcast to 2-D meshgrids and
    wrapped as xarray DataArrays so that satpy resamplers work correctly.
    """
    lons_np, lats_np = np.meshgrid(ds["lon"].values, ds["lat"].values)
    lons_da = xr.DataArray(lons_np, dims=["y", "x"])
    lats_da = xr.DataArray(lats_np, dims=["y", "x"])
    return SwathDefinition(lons=lons_da, lats=lats_da)


# ---------------------------------------------------------------------------
# xESMF regridding
# ---------------------------------------------------------------------------


def compute_weights(
    source_grid: xr.Dataset,
    target_grid: xr.Dataset,
    weights_path: Path,
    method: str = "bilinear",
) -> Path:
    """Compute xESMF regridding weights and save them to *weights_path*.

    Always recomputes — never reuses an existing file.  Returns *weights_path*.
    """
    log.info("Computing xESMF %s weights …", method)
    regridder = xesmf.Regridder(source_grid, target_grid, method=method, periodic=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    regridder.to_netcdf(str(weights_path))
    log.info("xESMF weights saved to %s", weights_path)
    return weights_path


def load_regridder(
    source_grid: xr.Dataset,
    target_grid: xr.Dataset,
    weights_path: Path,
    method: str = "bilinear",
) -> xesmf.Regridder:
    """Load a pre-computed xESMF regridder from *weights_path*."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}. Run compute_weights() first."
        )
    log.info("Loading xESMF weights from %s", weights_path)
    return xesmf.Regridder(
        source_grid,
        target_grid,
        method=method,
        periodic=True,
        weights=str(weights_path),
        reuse_weights=True,
    )


# ---------------------------------------------------------------------------
# satpy regridding (one variable at a time)
# ---------------------------------------------------------------------------


def _iter_vars(ds: xr.Dataset) -> list[tuple[str, xr.DataArray]]:
    """Return (name, DataArray) pairs for each CEDA variable as 2-D ['y','x'] arrays."""
    return [
        (
            var,
            xr.DataArray(ds[var].values.astype(np.float32), dims=["y", "x"]),
        )
        for var in CEDA_VARS
    ]


def _stack_vars(slices: dict[str, np.ndarray]) -> np.ndarray:
    """Stack per-variable arrays into shape (NY, NX, n_vars) — channel-last."""
    return np.stack([slices[v] for v in CEDA_VARS], axis=-1)


def regrid_nearest(
    ds: xr.Dataset,
    source_def: SwathDefinition,
    target_def: AreaDefinition,
    cache_dir: Path = SATPY_CACHE,
) -> np.ndarray:
    """Regrid using satpy KDTreeResampler (nearest-neighbour).

    Returns float32 (NY, NX, n_vars) array.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    resampler = KDTreeResampler(source_def, target_def)
    resampler.precompute(
        radius_of_influence=RADIUS_OF_INFLUENCE,
        cache_dir=str(cache_dir),
    )
    slices = {
        name: np.asarray(resampler.compute(da, fill_value=np.nan))
        for name, da in _iter_vars(ds)
    }
    return _stack_vars(slices)


def regrid_bilinear(
    ds: xr.Dataset,
    source_def: SwathDefinition,
    target_def: AreaDefinition,
    cache_dir: Path = SATPY_CACHE,
) -> np.ndarray:
    """Regrid using satpy BilinearResampler.

    Returns float32 (NY, NX, n_vars) array.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    resampler = BilinearResampler(source_def, target_def)
    slices = {
        name: np.asarray(
            resampler.resample(
                da,
                radius_of_influence=RADIUS_OF_INFLUENCE,
                fill_value=np.nan,
                cache_dir=str(cache_dir),
            )
        )
        for name, da in _iter_vars(ds)
    }
    return _stack_vars(slices)


def regrid_ewa(
    ds: xr.Dataset,
    source_def: SwathDefinition,
    target_def: AreaDefinition,
) -> np.ndarray:
    """Regrid using DaskEWAResampler.

    Returns float32 (NY, NX, n_vars) array.
    """
    resampler = DaskEWAResampler(source_def, target_def)
    slices = {
        name: np.asarray(resampler.resample(da, rows_per_scan=0, fill_value=np.nan))
        for name, da in _iter_vars(ds)
    }
    return _stack_vars(slices)


def regrid_bucket_avg(
    ds: xr.Dataset,
    source_def: SwathDefinition,
    target_def: AreaDefinition,
) -> np.ndarray:
    """Regrid using satpy BucketAvg.

    WARNING: BucketAvg is a coarse-to-coarse or fine-to-coarse method.  It
    assigns each *source* pixel to the single *target* cell its centre falls
    in, then averages all source pixels in that cell.  For CEDA 0.1° (≈11 km)
    → LIS 1 km the source is much coarser than the target; each CEDA pixel
    maps to exactly one LIS cell, leaving ~120 neighbouring LIS cells empty.
    The result is therefore nearly all NaN and is not useful for this use
    case.  It is retained here for completeness and benchmarking comparison.

    Returns float32 (NY, NX, n_vars) array.
    """
    resampler = BucketAvg(source_def, target_def)
    slices = {
        name: np.asarray(resampler.resample(da, fill_value=np.nan))
        for name, da in _iter_vars(ds)
    }
    return _stack_vars(slices)


# ---------------------------------------------------------------------------
# File-level regridding (xESMF path, writes NetCDF)
# ---------------------------------------------------------------------------


def regrid_file(
    ceda_path: str,
    storage: StorageConfig,
    regridder: xesmf.Regridder,
    out_dir: Path,
) -> Path:
    """Apply a pre-loaded xESMF regridder to one CEDA file and write NetCDF.

    Returns the path of the written output file.
    """
    ds = load_ceda(ceda_path, storage)
    stem = ceda_path.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    log.info("Regridding %s …", stem)
    regridded = regridder(ds)
    out_path = out_dir / (stem + ".nc")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing output to %s", out_path)
    regridded.to_netcdf(out_path, engine="h5netcdf")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regrid CEDA ESA CCI SWE files to the LIS input grid."
    )
    add_storage_args(parser)
    ns = parser.parse_args()
    storage = storage_config_from_namespace(ns)

    ceda_files = storage.glob(CEDA_GLOB)
    if not ceda_files:
        raise FileNotFoundError(
            f"No CEDA files found matching '{CEDA_GLOB}' in {storage.storage_location}"
        )
    log.info("Found %d CEDA file(s)", len(ceda_files))

    lis_grid = load_lis_grid(storage)

    # All CEDA daily files share the same 0.1° grid — compute weights once
    source_ds = load_ceda(ceda_files[0], storage)
    source_grid = source_ds[["lat", "lon"]]
    compute_weights(source_grid, lis_grid, WEIGHTS_PATH)
    regridder = load_regridder(source_grid, lis_grid, WEIGHTS_PATH)

    for ceda_path in ceda_files:
        out_path = regrid_file(ceda_path, storage, regridder, _DATA_OUT)
        log.info("Done: %s", out_path)

    log.info("All CEDA files regridded successfully.")


if __name__ == "__main__":
    main()
