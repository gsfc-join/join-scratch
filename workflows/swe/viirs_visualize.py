#!/usr/bin/env python
"""Visualize VIIRS CGF Snow Cover regridding results: source tiles vs regridded."""

import argparse
import logging
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from join_scratch.datasets import ViirsFileHandler
from visualize_utils import add_map_features, plot_panel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

VIIRS_GLOB = "**/*.h5"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize VIIRS source tiles vs regridded side-by-side."
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

    src_files = sorted(ns.src_dir.glob(VIIRS_GLOB))
    if not src_files:
        raise FileNotFoundError(f"No VIIRS .h5 files found under {ns.src_dir}")
    log.info("Found %d source file(s)", len(src_files))

    # Group by stem prefix (same date/tile pattern used in viirs_regrid.py)
    from collections import defaultdict

    date_groups: dict[str, list[Path]] = defaultdict(list)
    for p in src_files:
        parts = p.stem.split(".")
        date_key = parts[1] if len(parts) > 1 else p.stem
        date_groups[date_key].append(p)

    for date_key, paths in sorted(date_groups.items()):
        log.info("Processing date %s (%d tile(s))", date_key, len(paths))

        # Find the corresponding regridded NC file (named <stem>_<method>.nc)
        stem = paths[0].stem
        nc_candidates = list(ns.regrid_dir.glob(f"{stem}_*.nc"))
        if not nc_candidates:
            log.warning("No regridded file found for date %s, skipping", date_key)
            continue
        nc_path = nc_candidates[0]

        # Load all source tiles for this date (keep per-tile for correct pcolormesh)
        tiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for path in paths:
            handler = ViirsFileHandler.from_path(path)
            da = handler.get_dataset()
            swath_def = da.attrs["area"]
            tile_data = da.values.astype(float)
            tile_lons = swath_def.lons.values
            tile_lats = swath_def.lats.values
            tiles.append((tile_data, tile_lons, tile_lats))

        # Flatten for vmax computation (mask to LIS extent)
        lon_min, lon_max, lat_min, lat_max = extent
        flat_vals: list[np.ndarray] = []
        for tile_data, tile_lons, tile_lats in tiles:
            mask = (
                (tile_lons >= lon_min)
                & (tile_lons <= lon_max)
                & (tile_lats >= lat_min)
                & (tile_lats <= lat_max)
            )
            v = tile_data.copy()
            v[~mask] = np.nan
            flat_vals.append(v.ravel())

        new = xr.open_dataset(nc_path)
        regridded = new["CGF_NDSI_Snow_Cover"].values.astype(float)

        all_flat = np.concatenate(flat_vals)
        vmax = max(
            float(np.nanpercentile(all_flat[np.isfinite(all_flat)], 99))
            if np.any(np.isfinite(all_flat))
            else 1.0,
            float(np.nanpercentile(regridded, 99)),
        )
        vmin = 0.0

        fig, (ax_src, ax_new) = plt.subplots(
            1, 2, subplot_kw={"projection": PROJ}, figsize=(14, 6)
        )
        fig.suptitle(f"VIIRS CGF Snow Cover — {date_key}", fontsize=9)

        # Plot each tile separately to avoid cross-tile pcolormesh artifacts
        ax_src.set_extent(extent, crs=PC)
        im = None
        for tile_data, tile_lons, tile_lats in tiles:
            mask = (
                (tile_lons >= lon_min)
                & (tile_lons <= lon_max)
                & (tile_lats >= lat_min)
                & (tile_lats <= lat_max)
            )
            display = tile_data.copy()
            display[~mask] = np.nan
            im = ax_src.pcolormesh(
                tile_lons,
                tile_lats,
                display,
                transform=PC,
                vmin=vmin,
                vmax=vmax,
                cmap="Blues",
                shading="auto",
            )
        add_map_features(ax_src)
        ax_src.set_title("Source (sinusoidal tiles)", fontsize=8)
        plt.colorbar(im, ax=ax_src, orientation="horizontal", pad=0.02, label="NDSI")

        im = plot_panel(
            ax_new,
            regridded,
            lis_lons,
            lis_lats,
            vmin,
            vmax,
            "Regridded (LIS 1km)",
            extent,
            PC,
        )
        plt.colorbar(im, ax=ax_new, orientation="horizontal", pad=0.02, label="NDSI")

        fig.tight_layout()
        out = ns.output_dir / f"{stem}_viirs_compare.png"
        fig.savefig(out, dpi=150)
        log.info("Saved %s", out)
        plt.close(fig)


if __name__ == "__main__":
    main()
