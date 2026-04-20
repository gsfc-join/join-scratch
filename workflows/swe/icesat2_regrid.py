#!/usr/bin/env python
"""Grid ICESat-2 ATL06 snow height data to the LIS model grid."""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from join_scratch.datasets import Icesat2FileHandler
from join_scratch.regrid import regrid
from lis_grid import build_lis_area_definition, load_lis_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# Default temporal search window
T0_DEFAULT = "2019-01-01T00:00:00Z"
T1_DEFAULT = "2019-01-07T23:59:59Z"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid ICESat-2 ATL06 snow height data to the LIS model grid."
    )
    parser.add_argument(
        "--lis-path",
        required=True,
        type=Path,
        help="Path to the LIS input NetCDF file.",
    )
    parser.add_argument(
        "--parquet-path",
        required=True,
        type=Path,
        help="Path to the cached ATL06 parquet file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("_data/icesat2/atl06_gridded.nc"),
        help="Output NetCDF path (default: _data/icesat2/atl06_gridded.nc).",
    )
    ns = parser.parse_args()

    # Load LIS grid for coordinates
    lis_grid = load_lis_grid(ns.lis_path)
    lis_area = build_lis_area_definition(ns.lis_path)

    # Load ATL06 data from parquet
    handler = Icesat2FileHandler.from_path(ns.parquet_path)
    da = handler.get_dataset()
    source_area = da.attrs["area"]

    log.info("Regridding %d ATL06 observations using method=mean …", len(da))
    regridded = regrid(da, source_area, lis_area, method="mean")

    # Inline write_output
    ns.output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            "h_li": xr.DataArray(
                regridded.values.astype(np.float32),
                dims=["north_south", "east_west"],
                attrs={
                    "long_name": "ICESat-2 ATL06 land-ice surface height",
                    "units": "meters",
                    "comment": (
                        "Mean of ATL06 h_li values within each LIS pixel. "
                        "NaN indicates no ICESat-2 observations in this pixel."
                    ),
                },
            )
        },
        coords={
            "lat": lis_grid["lat"],
            "lon": lis_grid["lon"],
        },
    )

    encoding = {"h_li": {"dtype": "float32", "_FillValue": np.float32("nan")}}
    log.info("Writing output to %s", ns.output_path)
    ds.to_netcdf(ns.output_path, engine="h5netcdf", encoding=encoding)
    log.info("Done: %s", ns.output_path)


if __name__ == "__main__":
    main()
