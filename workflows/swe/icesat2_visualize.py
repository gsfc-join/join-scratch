#!/usr/bin/env python
"""Visualize ICESat-2 ATL06 regridding results: scatter points vs gridded output."""

import argparse
import logging
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from join_scratch.datasets import Icesat2FileHandler
from visualize_utils import plot_panel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ICESat-2 ATL06 source points vs regridded side-by-side."
    )
    parser.add_argument("--lis-path", required=True, type=Path)
    parser.add_argument(
        "--parquet-path",
        required=True,
        type=Path,
        help="Path to the cached ATL06 parquet file.",
    )
    parser.add_argument(
        "--regrid-path",
        required=True,
        type=Path,
        help="Path to the regridded ATL06 NetCDF file.",
    )
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

    # Load source point cloud
    handler = Icesat2FileHandler.from_path(ns.parquet_path)
    da = handler.get_dataset()
    src_area = da.attrs["area"]
    src_lons = src_area.lons.values
    src_lats = src_area.lats.values
    src_vals = da.values

    # Crop to LIS extent
    mask = (
        (src_lons >= lon_min)
        & (src_lons <= lon_max)
        & (src_lats >= lat_min)
        & (src_lats <= lat_max)
    )
    src_lons_c = src_lons[mask]
    src_lats_c = src_lats[mask]
    src_vals_c = src_vals[mask]

    # Load regridded
    new = xr.open_dataset(ns.regrid_path)
    regridded = new["h_li"].values

    valid_src = src_vals_c[np.isfinite(src_vals_c)]
    valid_new = regridded[np.isfinite(regridded)]
    if len(valid_src) == 0 and len(valid_new) == 0:
        log.warning("No valid data in LIS extent — skipping")
        return

    vmin = float(np.nanpercentile(np.concatenate([valid_src, valid_new]), 1))
    vmax = float(np.nanpercentile(np.concatenate([valid_src, valid_new]), 99))

    fig, (ax_src, ax_new) = plt.subplots(
        1, 2, subplot_kw={"projection": PROJ}, figsize=(14, 6)
    )
    fig.suptitle(f"ICESat-2 ATL06 — {ns.parquet_path.stem}", fontsize=9)

    # Source: scatter plot over map
    ax_src.set_extent(extent, crs=PC)
    from visualize_utils import add_map_features

    add_map_features(ax_src)
    sc = ax_src.scatter(
        src_lons_c,
        src_lats_c,
        c=src_vals_c,
        s=1,
        transform=PC,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax_src.set_title("Source ATL06 points", fontsize=8)
    plt.colorbar(sc, ax=ax_src, orientation="horizontal", pad=0.02, label="h_li (m)")

    # Regridded
    im = plot_panel(
        ax_new,
        regridded,
        lis_lons,
        lis_lats,
        vmin,
        vmax,
        "Regridded (LIS 1km mean)",
        extent,
        PC,
    )
    plt.colorbar(im, ax=ax_new, orientation="horizontal", pad=0.02, label="h_li (m)")

    fig.tight_layout()
    out = ns.output_dir / f"{ns.parquet_path.stem}_icesat2_compare.png"
    fig.savefig(out, dpi=150)
    log.info("Saved %s", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
