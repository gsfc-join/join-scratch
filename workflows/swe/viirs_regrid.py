#!/usr/bin/env python
"""Regrid VIIRS CGF Snow Cover files to the LIS input grid."""

import sys
import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from join_scratch.datasets import ViirsFileHandler
from join_scratch.regrid import regrid
from lis_grid import build_lis_area_definition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

VIIRS_GLOB = "**/*.h5"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regrid VIIRS CGF Snow Cover files to the LIS input grid."
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
        help="Directory containing VIIRS HDF5 files (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_data/viirs"),
        help="Directory for output NetCDF files (default: _data/viirs).",
    )
    parser.add_argument(
        "--method",
        default="nearest",
        choices=["nearest", "bilinear", "ewa", "bucket_avg"],
        help="Regridding method (default: nearest).",
    )
    ns = parser.parse_args()

    viirs_files = sorted(ns.input_dir.glob(VIIRS_GLOB))
    if not viirs_files:
        raise FileNotFoundError(
            f"No VIIRS HDF5 files found matching '{VIIRS_GLOB}' under {ns.input_dir}"
        )
    log.info("Found %d VIIRS file(s)", len(viirs_files))

    # Build LIS AreaDefinition once (fixes the double-call bug in old viirs_regrid.py)
    lis_area = build_lis_area_definition(ns.lis_path)
    lis_lons, lis_lats = lis_area.get_lonlats()

    # Group files by date (YYYYDDD encoded in filename, e.g. A2019001)
    date_groups: dict[str, list[Path]] = defaultdict(list)
    for p in viirs_files:
        parts = p.stem.split(".")
        date_key = parts[1] if len(parts) > 1 else p.stem
        date_groups[date_key].append(p)

    ns.output_dir.mkdir(parents=True, exist_ok=True)

    for date_key, paths in sorted(date_groups.items()):
        log.info("Processing date %s (%d tile(s))", date_key, len(paths))

        # Load and composite all tiles for this date
        all_data = []
        all_lons = []
        all_lats = []
        stem = None
        for path in paths:
            handler = ViirsFileHandler.from_path(path)
            da = handler.get_dataset()
            swath_def = da.attrs["area"]
            all_data.append(da.values)
            all_lons.append(swath_def.lons.values)
            all_lats.append(swath_def.lats.values)
            if stem is None:
                stem = path.stem

        from pyresample.geometry import SwathDefinition

        composite_data = np.concatenate(all_data, axis=0)
        composite_lons = np.concatenate(all_lons, axis=0)
        composite_lats = np.concatenate(all_lats, axis=0)

        lons_da = xr.DataArray(composite_lons, dims=["y", "x"])
        lats_da = xr.DataArray(composite_lats, dims=["y", "x"])
        source_def = SwathDefinition(lons=lons_da, lats=lats_da)

        composite_da = xr.DataArray(composite_data, dims=["y", "x"])

        log.info("Regridding with method=%s …", ns.method)
        regridded = regrid(composite_da, source_def, lis_area, method=ns.method)

        ds_out = xr.Dataset(
            {
                "CGF_NDSI_Snow_Cover": xr.DataArray(
                    regridded.values,
                    dims=["y", "x"],
                    attrs={"long_name": "CGF NDSI Snow Cover", "units": "1"},
                ),
                "lat": xr.DataArray(lis_lats, dims=["y", "x"]),
                "lon": xr.DataArray(lis_lons, dims=["y", "x"]),
            }
        )

        out_path = ns.output_dir / f"{stem}_{ns.method}.nc"
        log.info("Writing output to %s", out_path)
        ds_out.to_netcdf(out_path, engine="h5netcdf")
        log.info("Done: %s", out_path)

    log.info("All VIIRS tiles regridded successfully.")


if __name__ == "__main__":
    main()
