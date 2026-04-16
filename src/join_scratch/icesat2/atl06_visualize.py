#!/usr/bin/env python
"""Visualize ICESat-2 ATL06 gridded snow height results.

Produces a single PNG in _figures/ with three panels:
  1. ICESat-2 ground tracks / observation locations (point scatter in LIS domain)
  2. Gridded ATL06 h_li mean on the LIS pixel grid
  3. Histogram of valid h_li values

Coastlines and country borders are drawn via cartopy's Natural Earth dataset.
"""

import argparse
import logging
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from join_scratch.icesat2.atl06_regrid import (
    CACHE_PATH,
    OUTPUT_PATH,
    load_lis_grid,
)
from join_scratch.storage import add_storage_args, storage_config_from_namespace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = ROOT / "_figures"

PLATE_CARREE = ccrs.PlateCarree()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_map_features(ax) -> None:
    """Add coastlines, country borders, and basic formatting to a cartopy axis."""
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray", linestyle="--")
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="lightgray")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)


def _lis_extent(
    lis_grid: xr.Dataset, pad: float = 1.0
) -> tuple[float, float, float, float]:
    """Return (lon_min, lon_max, lat_min, lat_max) with optional padding."""
    lat = lis_grid["lat"].values
    lon = lis_grid["lon"].values
    return (
        float(lon.min()) - pad,
        float(lon.max()) + pad,
        float(lat.min()) - pad,
        float(lat.max()) + pad,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ICESat-2 ATL06 gridded snow height results."
    )
    add_storage_args(parser)
    parser.add_argument(
        "--cache",
        type=Path,
        default=CACHE_PATH,
        help="Path to the ATL06 Parquet cache file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to the gridded NetCDF output file.",
    )
    ns = parser.parse_args()
    storage = storage_config_from_namespace(ns)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    log.info("Loading LIS grid …")
    lis_grid = load_lis_grid(storage)
    lon_min, lon_max, lat_min, lat_max = _lis_extent(lis_grid, pad=1.0)

    log.info("Loading ATL06 cache from %s …", ns.cache)
    gdf: gpd.GeoDataFrame = gpd.read_parquet(ns.cache)

    log.info("Loading gridded output from %s …", ns.output)
    gridded = xr.open_dataset(ns.output, engine="h5netcdf")
    h_li = gridded["h_li"].values  # (north_south, east_west)
    grid_lons = lis_grid["lon"].values
    grid_lats = lis_grid["lat"].values

    # --- Subsample points for plotting (scatter is slow with 5 M points) ---
    rng = np.random.default_rng(42)
    max_pts = 100_000
    if len(gdf) > max_pts:
        idx = rng.choice(len(gdf), size=max_pts, replace=False)
        gdf_plot = gdf.iloc[idx]
        log.info("Subsampled %d / %d points for scatter plot", max_pts, len(gdf))
    else:
        gdf_plot = gdf

    pt_lons = gdf_plot.geometry.x.values
    pt_lats = gdf_plot.geometry.y.values

    # Colour scale for gridded data: 2–98th percentile of valid values
    valid = h_li[np.isfinite(h_li)]
    vmin, vmax = np.nanpercentile(valid, [2, 98]) if valid.size > 0 else (0, 1)

    # --- Figure layout: 1×3 ---
    proj = ccrs.LambertConformal(
        central_longitude=(lon_min + lon_max) / 2.0,
        central_latitude=(lat_min + lat_max) / 2.0,
    )

    fig = plt.figure(figsize=(22, 8), constrained_layout=True)
    ax1 = fig.add_subplot(1, 3, 1, projection=proj)
    ax2 = fig.add_subplot(1, 3, 2, projection=proj)
    ax3 = fig.add_subplot(1, 3, 3)

    map_extent = [lon_min, lon_max, lat_min, lat_max]

    # --- Panel 1: ICESat-2 ground tracks ---
    ax1.set_extent(map_extent, crs=PLATE_CARREE)
    _add_map_features(ax1)
    ax1.scatter(
        pt_lons,
        pt_lats,
        s=0.3,
        c="steelblue",
        alpha=0.4,
        linewidths=0,
        transform=PLATE_CARREE,
        rasterized=True,
    )
    ax1.set_title(
        f"ICESat-2 ATL06 observations\n"
        f"(subsample {max_pts:,} / {len(gdf):,} pts, Jan 1–7 2019)",
        fontsize=10,
    )

    # --- Panel 2: Gridded h_li ---
    ax2.set_extent(map_extent, crs=PLATE_CARREE)
    _add_map_features(ax2)
    pcm = ax2.pcolormesh(
        grid_lons,
        grid_lats,
        h_li,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        transform=PLATE_CARREE,
        rasterized=True,
    )
    fig.colorbar(pcm, ax=ax2, label="h_li (m above ellipsoid)", shrink=0.85, pad=0.02)
    n_valid = int(np.isfinite(h_li).sum())
    fill_pct = 100.0 * n_valid / h_li.size
    ax2.set_title(
        f"Gridded ATL06 mean h_li on LIS grid\n"
        f"({n_valid:,} / {h_li.size:,} pixels filled, {fill_pct:.2f}%)",
        fontsize=10,
    )

    # --- Panel 3: Histogram of h_li values ---
    ax3.hist(valid, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax3.set_xlabel("h_li (m above ellipsoid)", fontsize=11)
    ax3.set_ylabel("Pixel count", fontsize=11)
    ax3.set_title("Distribution of gridded h_li\n(valid pixels only)", fontsize=10)
    ax3.axvline(
        float(np.nanmedian(valid)),
        color="red",
        linewidth=1.2,
        label=f"Median: {np.nanmedian(valid):.1f} m",
    )
    ax3.legend(fontsize=9)

    out_path = FIGURES_DIR / "icesat2_atl06_gridded_2019-01-01_2019-01-07.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Figure saved to %s", out_path)


if __name__ == "__main__":
    main()
