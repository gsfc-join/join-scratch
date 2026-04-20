#!/usr/bin/env python
"""Regrid AMSR2 snow depth files to the LIS input grid."""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from join_scratch.datasets import Amsr2FileHandler
from join_scratch.regrid.regular_to_regular import compute_weights, load_regridder
from lis_grid import load_lis_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

AMSR2_GLOB = "**/*.h5"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regrid AMSR2 snow depth files to the LIS input grid."
    )
    parser.add_argument(
        "--lis-path",
        required=True,
        type=Path,
        help="Path to the LIS input NetCDF file.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing AMSR2 HDF5 files (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_data/amsr2"),
        help="Directory for output NetCDF files (default: _data/amsr2).",
    )
    ns = parser.parse_args()

    amsr2_files = sorted(ns.input_dir.glob(AMSR2_GLOB))
    if not amsr2_files:
        raise FileNotFoundError(
            f"No AMSR2 HDF5 files found matching '{AMSR2_GLOB}' under {ns.input_dir}"
        )
    log.info("Found %d AMSR2 file(s)", len(amsr2_files))

    lis_grid = load_lis_grid(ns.lis_path)

    # All AMSR2 files share the same 0.1° equirectangular grid — compute weights once
    first_handler = Amsr2FileHandler.from_path(amsr2_files[0])

    # Build a minimal source/target Dataset for xESMF.
    # xESMF requires lat/lon as DIMENSION names (not just coordinates), and
    # lat must be monotonically increasing (south-first) — matching the
    # original load_amsr2 which used sortby(["lat", "lon"]).
    lat_asc = np.sort(first_handler._lat)  # -89.95 → +89.95
    lon_asc = np.sort(first_handler._lon)  # -179.95 → +179.95
    source_grid = xr.Dataset(coords={"lat": lat_asc, "lon": lon_asc})

    weights_path = ns.output_dir / "amsr2-lis-weights.nc"
    compute_weights(source_grid, lis_grid, weights_path)
    regridder = load_regridder(source_grid, lis_grid, weights_path)

    ns.output_dir.mkdir(parents=True, exist_ok=True)
    for path in amsr2_files:
        handler = Amsr2FileHandler.from_path(path)
        da = handler.get_dataset()  # dims (y, x, inner), lat north→south
        # xESMF needs data with dim names matching those used for weight computation.
        # Rename y→lat, x→lon and sort lat ascending (south-first) to match source_grid.
        da = da.rename({"y": "lat", "x": "lon"}).sortby("lat")
        ds = da.to_dataset(name="Geophysical Data")
        log.info("Regridding %s …", path.stem)
        regridded = regridder(ds)
        out_path = ns.output_dir / (path.stem + ".nc")
        log.info("Writing output to %s", out_path)
        regridded.to_netcdf(out_path, engine="h5netcdf")
        log.info("Done: %s", out_path)

    log.info("All AMSR2 files regridded successfully.")


if __name__ == "__main__":
    main()
