#!/usr/bin/env python
"""Visualize AMSR2 regridding results: source (native grid) vs regridded."""

import argparse
import logging
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from join_scratch.datasets import Amsr2FileHandler
from visualize_utils import plot_panel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

AMSR2_GLOB = "**/*.h5"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize AMSR2 source vs regridded side-by-side."
    )
    parser.add_argument("--lis-path", required=True, type=Path)
    parser.add_argument("--src-dir", required=True, type=Path)
    parser.add_argument("--regrid-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("_figures"))
    ns = parser.parse_args()

    ns.output_dir.mkdir(parents=True, exist_ok=True)

    # LIS grid for regridded output (flip south→north)
    lis = xr.open_dataset(ns.lis_path, engine="h5netcdf").isel(
        north_south=slice(None, None, -1)
    )
    lis_lats = lis["lat"].values
    lis_lons = lis["lon"].values

    PROJ = ccrs.LambertConformal(
        central_longitude=-100.0, standard_parallels=(39.0, 46.0)
    )
    PC = ccrs.PlateCarree()

    extent = [
        float(lis_lons.min()) - 1,
        float(lis_lons.max()) + 1,
        float(lis_lats.min()) - 1,
        float(lis_lats.max()) + 1,
    ]
    lon_min, lon_max, lat_min, lat_max = extent

    src_files = sorted(ns.src_dir.glob(AMSR2_GLOB))
    if not src_files:
        raise FileNotFoundError(f"No AMSR2 .h5 files found under {ns.src_dir}")
    log.info("Found %d source file(s)", len(src_files))

    for h5_path in src_files:
        stem = h5_path.stem
        nc_path = ns.regrid_dir / (stem + ".nc")

        if not nc_path.exists():
            log.warning("Regridded file not found, skipping: %s", nc_path)
            continue

        # Source data (native 0.1° global grid)
        handler = Amsr2FileHandler.from_path(h5_path)
        src_da = handler.get_dataset("Geophysical Data")
        src_lats = handler._lat  # descending: 89.95 → -89.95
        src_lons = handler._lon

        # Flip to ascending lat for consistent plotting
        src_da_vals = src_da.values[::-1, :, :]
        src_lats_asc = src_lats[::-1]

        lat_mask = (src_lats_asc >= lat_min) & (src_lats_asc <= lat_max)
        lon_mask = (src_lons >= lon_min) & (src_lons <= lon_max)
        src_data_crop = src_da_vals[
            np.ix_(lat_mask, lon_mask, np.arange(src_da_vals.shape[2]))
        ]
        src_lats_crop = src_lats_asc[lat_mask]
        src_lons_crop = src_lons[lon_mask]

        new = xr.open_dataset(nc_path)
        gd_n = new["Geophysical Data"].values  # (inner, ny, nx)

        n_inner = gd_n.shape[0]
        vmax = max(np.nanpercentile(src_data_crop, 99), np.nanpercentile(gd_n, 99))
        vmin = 0

        fig = plt.figure(figsize=(14, 9))
        fig.suptitle(stem, fontsize=9)

        for i in range(n_inner):
            ax_src = fig.add_subplot(2, n_inner, i + 1, projection=PROJ)
            ax_new = fig.add_subplot(2, n_inner, n_inner + i + 1, projection=PROJ)

            im = plot_panel(
                ax_src,
                src_data_crop[:, :, i],
                src_lons_crop,
                src_lats_crop,
                vmin,
                vmax,
                f"Source (native 0.1°) — inner[{i}]",
                extent,
                PC,
            )
            plt.colorbar(
                im, ax=ax_src, orientation="horizontal", pad=0.02, label="raw int16"
            )

            im = plot_panel(
                ax_new,
                gd_n[i],
                lis_lons,
                lis_lats,
                vmin,
                vmax,
                f"Regridded (LIS 1km) — inner[{i}]",
                extent,
                PC,
            )
            plt.colorbar(
                im, ax=ax_new, orientation="horizontal", pad=0.02, label="raw int16"
            )

        fig.tight_layout()
        out = ns.output_dir / (stem + "_compare.png")
        fig.savefig(out, dpi=150)
        log.info("Saved %s", out)
        plt.close(fig)


if __name__ == "__main__":
    main()
