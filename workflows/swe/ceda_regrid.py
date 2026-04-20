#!/usr/bin/env python
"""Regrid CEDA ESA CCI SWE files to the LIS input grid."""

import sys
import argparse
import logging
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from join_scratch.datasets import CedaFileHandler
from join_scratch.regrid.regular_to_regular import compute_weights, load_regridder
from lis_grid import load_lis_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

CEDA_GLOB = "**/*.nc"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regrid CEDA ESA CCI SWE files to the LIS input grid."
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
        help="Directory containing CEDA NetCDF files (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_data/ceda"),
        help="Directory for output NetCDF files (default: _data/ceda).",
    )
    ns = parser.parse_args()

    ceda_files = sorted(ns.input_dir.glob(CEDA_GLOB))
    if not ceda_files:
        raise FileNotFoundError(
            f"No CEDA NetCDF files found matching '{CEDA_GLOB}' under {ns.input_dir}"
        )
    log.info("Found %d CEDA file(s)", len(ceda_files))

    lis_grid = load_lis_grid(ns.lis_path)

    # All CEDA daily files share the same 0.1° grid — compute weights once
    first_handler = CedaFileHandler.from_path(ceda_files[0])
    first_ds = first_handler.get_dataset()

    # xESMF needs a Dataset with 1-D lat/lon dimension coordinates
    lat_vals = first_ds["lat"].values
    lon_vals = first_ds["lon"].values
    # lat/lon may be 1-D already or 2-D (take first col/row respectively)
    if lat_vals.ndim == 2:
        lat_vals = lat_vals[:, 0]
    if lon_vals.ndim == 2:
        lon_vals = lon_vals[0, :]
    source_grid = xr.Dataset(coords={"lat": lat_vals, "lon": lon_vals})

    weights_path = ns.output_dir / "ceda-lis-weights.nc"
    compute_weights(source_grid, lis_grid, weights_path)
    regridder = load_regridder(source_grid, lis_grid, weights_path)

    ns.output_dir.mkdir(parents=True, exist_ok=True)
    for path in ceda_files:
        handler = CedaFileHandler.from_path(path)
        ds = handler.get_dataset()
        # xESMF requires dim names matching the source_grid (lat/lon).
        # CedaFileHandler renames lat→y, lon→x; restore them using swap_dims
        # (rename_dims would fail because lat/lon coordinate variables still exist).
        ds = ds.swap_dims({"y": "lat", "x": "lon"})
        log.info("Regridding %s …", path.stem)
        regridded = regridder(ds)
        out_path = ns.output_dir / (path.stem + ".nc")
        log.info("Writing output to %s", out_path)
        regridded.to_netcdf(out_path, engine="h5netcdf")
        log.info("Done: %s", out_path)

    log.info("All CEDA files regridded successfully.")


if __name__ == "__main__":
    main()
