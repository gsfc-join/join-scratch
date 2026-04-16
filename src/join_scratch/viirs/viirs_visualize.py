#!/usr/bin/env python
"""Visualize VIIRS regridding results for the first set of tiles.

Produces a single PNG in _figures/ with:
  - Top row: original composite NDSI snow cover shown as a scatter plot in
    geographic (lon/lat) space, with the LIS domain boundary overlaid.
    Valid pixels only (flag values masked).  The plot extent covers the tile
    bounding box read from file metadata rather than the full lon2d array, so
    the display is correct regardless of which tiles are loaded.
  - Subsequent rows: regridded data on the LIS grid for each of the four
    satpy methods (nearest, bilinear, ewa, bucket_avg).

NOTE on sample data coverage
-----------------------------
The sample tiles (h00v08, h00v09) cover horizontal tile 0 of the MODIS/VIIRS
sinusoidal grid: roughly 180°W to 170°W, 10°S to 10°N (equatorial Pacific).
This area is entirely outside the Missouri LIS domain (~35°–50°N, ~94°–108°W).
The regridded panels will therefore be all-NaN for this sample data set —
that is the correct result.  With tiles that actually overlap the LIS domain
(e.g. h10v04, h10v05) the regridded panels would contain real values.

NOTE on bucket_avg
-------------------
BucketAvg assigns each *source* pixel to the single *target* cell its centre
falls in, then averages all source pixels in that cell.  This is only useful
when the source is *finer* than the target (many source pixels per target
cell).  For VIIRS 375 m → LIS 1 km the source is finer, so bucket_avg is
appropriate in principle.  However for CEDA/AMSR2 0.1° (≈11 km) → LIS 1 km,
the source is much coarser and bucket_avg will produce a mostly empty result.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from join_scratch.amsr2.amsr2_regrid import build_lis_area_definition
from join_scratch.storage import (
    StorageConfig,
    add_storage_args,
    storage_config_from_namespace,
)
from join_scratch.viirs.viirs_regrid import (
    SATPY_CACHE,
    VIIRS_GLOB,
    build_viirs_swath_definition,
    load_viirs_tiles,
    regrid_bilinear,
    regrid_bucket_avg,
    regrid_ewa,
    regrid_nearest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = ROOT / "_figures"

METHODS = ["nearest", "bilinear", "ewa", "bucket_avg"]
METHOD_LABELS = {
    "nearest": "satpy nearest",
    "bilinear": "satpy bilinear",
    "ewa": "satpy EWA",
    "bucket_avg": "satpy bucket avg",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lis_boundary_lonlat(lis_area) -> tuple[np.ndarray, np.ndarray]:
    """Return (lons, lats) arrays tracing the LIS domain boundary (closed loop)."""
    lons, lats = lis_area.get_lonlats()
    b_lons = np.concatenate(
        [lons[0, :], lons[:, -1], lons[-1, ::-1], lons[::-1, 0], [lons[0, 0]]]
    )
    b_lats = np.concatenate(
        [lats[0, :], lats[:, -1], lats[-1, ::-1], lats[::-1, 0], [lats[0, 0]]]
    )
    return b_lons, b_lats


def _tile_lonlat_extent(
    files: list[str], storage: StorageConfig
) -> tuple[float, float, float, float]:
    """Read bounding coordinates from file metadata (faster than computing from projection).

    Returns (lon_min, lon_max, lat_min, lat_max).
    """
    import h5py

    lons_w, lons_e, lats_s, lats_n = [], [], [], []
    for p in files:
        with storage.open(p) as fobj:
            with h5py.File(fobj, "r") as f:
                attrs = dict(f.attrs)
                lons_w.append(float(attrs["WestBoundingCoord"]))
                lons_e.append(float(attrs["EastBoundingCoord"]))
                lats_s.append(float(attrs["SouthBoundingCoord"]))
                lats_n.append(float(attrs["NorthBoundingCoord"]))
    return min(lons_w), max(lons_e), min(lats_s), max(lats_n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize VIIRS regridding results.")
    add_storage_args(parser)
    args = parser.parse_args()
    storage = storage_config_from_namespace(args)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    viirs_files = storage.glob(VIIRS_GLOB)
    if not viirs_files:
        raise FileNotFoundError(f"No VIIRS files found with glob {VIIRS_GLOB!r}")

    log.info("Loading VIIRS tile(s) …")
    tile = load_viirs_tiles(viirs_files, storage)
    lis_area = build_lis_area_definition(storage)
    source_def = build_viirs_swath_definition(tile)
    lis_lons, lis_lats = lis_area.get_lonlats()

    # --- Tile geographic extent from metadata ---
    lon_min, lon_max, lat_min, lat_max = _tile_lonlat_extent(viirs_files, storage)
    log.info(
        "Tile bounding box: lon [%.2f, %.2f]  lat [%.2f, %.2f]",
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    )

    # --- Compute all regridded outputs ---
    log.info("Regridding with nearest …")
    nearest_out = regrid_nearest(tile, source_def, lis_area, SATPY_CACHE)

    log.info("Regridding with bilinear …")
    bilinear_out = regrid_bilinear(tile, source_def, lis_area, SATPY_CACHE)

    log.info("Regridding with EWA …")
    ewa_out = regrid_ewa(tile, source_def, lis_area)

    log.info("Regridding with bucket_avg …")
    bucket_out = regrid_bucket_avg(tile, source_def, lis_area)

    regridded = {
        "nearest": nearest_out,
        "bilinear": bilinear_out,
        "ewa": ewa_out,
        "bucket_avg": bucket_out,
    }

    # --- Build boundary for overlay ---
    b_lons, b_lats = _lis_boundary_lonlat(lis_area)

    # --- Colour scale from valid raw pixels ---
    all_valid = tile["data"][np.isfinite(tile["data"])]
    if all_valid.size > 0:
        vmin, vmax = np.nanpercentile(all_valid, [2, 98])
    else:
        vmin, vmax = 0, 100
        log.warning(
            "No valid (non-flag) NDSI pixels in the loaded tiles. "
            "The sample tiles (h00v08/v09) cover equatorial Pacific lon [%.1f, %.1f] "
            "lat [%.1f, %.1f] — entirely outside the Missouri LIS domain. "
            "Plots will be blank.",
            lon_min,
            lon_max,
            lat_min,
            lat_max,
        )
    pkw = {"vmin": vmin, "vmax": vmax, "cmap": "Blues"}

    # --- Layout: 1 row for raw + 4 rows for methods ---
    n_rows = 1 + len(METHODS)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(10, 5 * n_rows),
        constrained_layout=True,
    )

    # Row 0: valid pixels as scatter in geographic space
    ax = axes[0]
    valid_mask = np.isfinite(tile["data"])
    if valid_mask.any():
        sc = ax.scatter(
            tile["lon2d"][valid_mask],
            tile["lat2d"][valid_mask],
            c=tile["data"][valid_mask],
            s=1,
            **pkw,
        )
        fig.colorbar(sc, ax=ax, label="NDSI Snow Cover (0–100)", shrink=0.85)
    else:
        ax.text(
            0.5,
            0.5,
            "No valid pixels\n(tiles outside LIS domain)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
    ax.plot(b_lons, b_lats, color="red", linewidth=1.2, label="LIS domain")
    ax.set_xlim(lon_min - 1, lon_max + 1)
    ax.set_ylim(lat_min - 1, lat_max + 1)
    ax.set_title(
        f"Original VIIRS CGF NDSI Snow Cover (composite)\n"
        f"Tile extent: lon [{lon_min:.1f}, {lon_max:.1f}]  "
        f"lat [{lat_min:.1f}, {lat_max:.1f}]",
        fontsize=11,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=8)

    # Rows 1+: regridded outputs on LIS grid
    for row_idx, method in enumerate(METHODS, start=1):
        ax = axes[row_idx]
        arr = regridded[method]
        finite_count = int(np.sum(np.isfinite(arr)))
        pcm = ax.pcolormesh(lis_lons, lis_lats, arr, **pkw)
        note = "" if finite_count > 0 else " — no data (tiles outside LIS domain)"
        ax.set_title(f"{METHOD_LABELS[method]}{note}", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(pcm, ax=ax, label="NDSI Snow Cover (0–100)", shrink=0.85)
        if finite_count == 0:
            ax.text(
                0.5,
                0.5,
                "No data\n(tiles outside LIS domain)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )

    # Derive a date string from the first filename for the output name
    first_stem = Path(viirs_files[0]).stem
    parts = first_stem.split(".")
    date_part = parts[1] if len(parts) > 1 else first_stem
    fname = f"{date_part}_viirs_regrid_comparison.png"
    out_path = FIGURES_DIR / fname
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Figure saved to %s", out_path)


if __name__ == "__main__":
    main()
