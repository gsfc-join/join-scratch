#!/usr/bin/env python
"""Visualize CEDA regridding results: original source (top) vs regridded (bottom)."""

import argparse
import logging
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from visualize_utils import add_map_features, plot_panel, _lis_extent  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

CEDA_GLOB = "**/*.nc"
VARS = ["swe", "swe_std"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize CEDA source vs regridded SWE side-by-side."
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

    src_files = sorted(ns.src_dir.glob(CEDA_GLOB))
    if not src_files:
        raise FileNotFoundError(f"No CEDA files found under {ns.src_dir}")
    log.info("Found %d source file(s)", len(src_files))

    for src_path in src_files:
        fname = src_path.name
        new_path = ns.regrid_dir / fname

        if not new_path.exists():
            log.warning("Regridded file not found, skipping: %s", new_path)
            continue

        # Source data (native 0.1° global grid)
        src = xr.open_dataset(src_path).isel(time=0)
        src_lats = src["lat"].values
        src_lons = src["lon"].values

        lat_mask = (src_lats >= lat_min) & (src_lats <= lat_max)
        lon_mask = (src_lons >= lon_min) & (src_lons <= lon_max)
        src_lats_crop = src_lats[lat_mask]
        src_lons_crop = src_lons[lon_mask]

        new = xr.open_dataset(new_path)

        fig = plt.figure(figsize=(14, 9))
        fig.suptitle(fname, fontsize=9)

        ncols = len(VARS)
        for col, var in enumerate(VARS):
            src_data = src[var].values[np.ix_(lat_mask, lon_mask)]
            new_data = new[var].values

            vmax = float(np.nanpercentile(src_data, 99))
            vmin = 0

            ax_src = fig.add_subplot(2, ncols, col + 1, projection=PROJ)
            ax_new = fig.add_subplot(2, ncols, ncols + col + 1, projection=PROJ)

            im = plot_panel(
                ax_src,
                src_data,
                src_lons_crop,
                src_lats_crop,
                vmin,
                vmax,
                f"Source (native 0.1°) — {var}",
                extent,
                PC,
            )
            plt.colorbar(im, ax=ax_src, orientation="horizontal", pad=0.02, label="mm")

            im = plot_panel(
                ax_new,
                new_data,
                lis_lons,
                lis_lats,
                vmin,
                vmax,
                f"Regridded (LIS 1km) — {var}",
                extent,
                PC,
            )
            plt.colorbar(im, ax=ax_new, orientation="horizontal", pad=0.02, label="mm")

        fig.tight_layout()
        out = ns.output_dir / (Path(fname).stem + "_ceda.png")
        fig.savefig(out, dpi=150)
        log.info("Saved %s", out)
        plt.close(fig)


if __name__ == "__main__":
    main()
