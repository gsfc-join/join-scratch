#!/usr/bin/env python
"""Regrid local AMSR2 snow depth files to the LIS input grid using xESMF."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "_data-raw"
DATA_OUT = ROOT / "_data" / "amsr2"
LIS_PATH = DATA_RAW / "lis_input_NMP_1000m_missouri.nc"
AMSR2_GLOB = "JOIN/AMSR2/**/*.h5"
WEIGHTS_PATH = DATA_OUT / "amsr2-lis-weights.nc"

# AMSR2 equirectangular grid parameters (0.1° resolution)
_AMSR2_LAT = np.linspace(89.95, -89.95, 1800)
_AMSR2_LON = np.linspace(-179.95, 179.95, 3600)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def load_lis_grid(path: Path) -> xr.Dataset:
    """Load the LIS input file and return a dataset containing only lat/lon."""
    log.info("Loading LIS grid from %s", path)
    ds = xr.open_dataset(path, engine="h5netcdf")
    return ds[["lat", "lon"]]


def load_amsr2(path: Path) -> xr.Dataset:
    """Load an AMSR2 HDF5 file, assign coordinates, and mask invalid values.

    Returns the full global dataset ready for regridding.
    """
    log.info("Loading AMSR2 file %s", path)
    ds = (
        xr.open_dataset(path, engine="h5netcdf", phony_dims="sort")
        .rename_dims({"phony_dim_0": "lat", "phony_dim_1": "lon"})
        .assign_coords(lat=_AMSR2_LAT, lon=_AMSR2_LON)
        .sortby(["lat", "lon"])
    )

    # Mask fill values (valid data is >= 0)
    ds = ds.where(ds["Geophysical Data"] >= 0)
    return ds


def get_regridder(
    source_grid: xr.Dataset,
    target_grid: xr.Dataset,
    weights_path: Path,
) -> xesmf.Regridder:
    """Build or load an xESMF bilinear regridder.

    If *weights_path* already exists the weights are reused; otherwise they are
    computed and saved to *weights_path*.
    """
    if weights_path.exists():
        log.info("Reusing existing weights from %s", weights_path)
        regridder = xesmf.Regridder(
            source_grid,
            target_grid,
            method="bilinear",
            periodic=True,
            weights=str(weights_path),
            reuse_weights=True,
        )
    else:
        log.info("Computing bilinear regridding weights …")
        regridder = xesmf.Regridder(
            source_grid,
            target_grid,
            method="bilinear",
            periodic=True,
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        regridder.to_netcdf(str(weights_path))
        log.info("Weights saved to %s", weights_path)

    return regridder


def regrid_file(
    amsr2_path: Path,
    regridder: xesmf.Regridder,
    out_dir: Path,
) -> Path:
    """Regrid a single AMSR2 file and write the result to *out_dir*.

    Returns the path of the written output file.
    """
    ds = load_amsr2(amsr2_path)

    log.info("Regridding %s …", amsr2_path.name)
    regridded = regridder(ds)

    out_path = out_dir / (amsr2_path.stem + ".nc")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Writing output to %s", out_path)
    regridded.to_netcdf(out_path, engine="h5netcdf")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    amsr2_files = sorted(DATA_RAW.glob(AMSR2_GLOB))
    if not amsr2_files:
        raise FileNotFoundError(
            f"No AMSR2 files found matching '{AMSR2_GLOB}' under {DATA_RAW}"
        )
    log.info("Found %d AMSR2 file(s)", len(amsr2_files))

    lis_grid = load_lis_grid(LIS_PATH)

    # Build/load the regridder using the first file's grid (all files share
    # the same equirectangular 0.1° grid, so one weight set covers all).
    source_ds = load_amsr2(amsr2_files[0])
    source_grid = source_ds[["lat", "lon"]]
    regridder = get_regridder(source_grid, lis_grid, WEIGHTS_PATH)

    for amsr2_path in amsr2_files:
        out_path = regrid_file(amsr2_path, regridder, DATA_OUT)
        log.info("Done: %s", out_path)

    log.info("All files regridded successfully.")


if __name__ == "__main__":
    main()
