#!/usr/bin/env python
"""Visualize AMSR2 regridding results for the first AMSR2 file.

Produces a single PNG in _figures/ with:
  - Top row: original AMSR2 data (global, lat/lon) with LIS domain boundary
    overlaid as a line, one panel per inner dimension slice
  - Subsequent rows: regridded data on the LIS grid for each method
    (xesmf_bilinear, nearest, bilinear, ewa, bucket_avg)

The inner dimension of size 2 (two retrieval channels) is shown side-by-side
in each row.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from join_scratch.amsr2.amsr2_regrid import (
    AMSR2_GLOB,
    WEIGHTS_PATH,
    build_amsr2_swath_definition,
    build_lis_area_definition,
    load_amsr2,
    load_lis_grid,
    load_regridder,
    regrid_bilinear,
    regrid_bucket_avg,
    regrid_ewa,
    regrid_nearest,
)
from join_scratch.storage import (
    StorageConfig,
    add_storage_args,
    storage_config_from_namespace,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = ROOT / "_figures"

N_INNER = 2
INNER_LABELS = ["Channel 0", "Channel 1"]

METHODS = [
    "xesmf_bilinear",
    "nearest",
    "bilinear",
    "ewa",
    "bucket_avg",
]
METHOD_LABELS = {
    "xesmf_bilinear": "xESMF bilinear",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize AMSR2 regridding results.")
    add_storage_args(parser)
    ns = parser.parse_args()
    storage = storage_config_from_namespace(ns)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    amsr2_files = storage.glob(AMSR2_GLOB)
    if not amsr2_files:
        raise FileNotFoundError(f"No AMSR2 files found in {storage.storage_location}")

    log.info("Loading data …")
    lis_grid = load_lis_grid(storage)
    amsr2_ds = load_amsr2(amsr2_files[0], storage)
    lis_area = build_lis_area_definition(storage)
    amsr2_swath = build_amsr2_swath_definition(amsr2_ds)
    lis_lons, lis_lats = lis_area.get_lonlats()

    # --- Compute all regridded outputs ---
    log.info("Regridding with xESMF …")
    source_grid = amsr2_ds[["lat", "lon"]]
    xesmf_regridder = load_regridder(source_grid, lis_grid, WEIGHTS_PATH)
    xesmf_out = xesmf_regridder(amsr2_ds)
    geo_data_xesmf = xesmf_out["Geophysical Data"].values
    if geo_data_xesmf.shape[0] == N_INNER:
        geo_data_xesmf = np.moveaxis(geo_data_xesmf, 0, -1)

    log.info("Regridding with nearest …")
    nearest_out = regrid_nearest(amsr2_ds, amsr2_swath, lis_area)

    log.info("Regridding with bilinear …")
    bilinear_out = regrid_bilinear(amsr2_ds, amsr2_swath, lis_area)

    log.info("Regridding with EWA …")
    ewa_out = regrid_ewa(amsr2_ds, amsr2_swath, lis_area)

    log.info("Regridding with bucket_avg …")
    bucket_out = regrid_bucket_avg(amsr2_ds, amsr2_swath, lis_area)

    regridded = {
        "xesmf_bilinear": geo_data_xesmf,
        "nearest": nearest_out,
        "bilinear": bilinear_out,
        "ewa": ewa_out,
        "bucket_avg": bucket_out,
    }

    # --- Build boundary for overlay ---
    b_lons, b_lats = _lis_boundary_lonlat(lis_area)

    # --- Raw AMSR2 data subsetted to LIS region + padding ---
    raw_data = amsr2_ds["Geophysical Data"].values  # (1800, 3600, 2)
    raw_lons = amsr2_ds["lon"].values
    raw_lats = amsr2_ds["lat"].values
    lat_pad, lon_pad = 3.0, 5.0
    lat_min = float(lis_lats.min()) - lat_pad
    lat_max = float(lis_lats.max()) + lat_pad
    lon_min = float(lis_lons.min()) - lon_pad
    lon_max = float(lis_lons.max()) + lon_pad
    lat_mask = (raw_lats >= lat_min) & (raw_lats <= lat_max)
    lon_mask = (raw_lons >= lon_min) & (raw_lons <= lon_max)
    raw_sub = raw_data[np.ix_(lat_mask, lon_mask)]
    raw_lons_sub = raw_lons[lon_mask]
    raw_lats_sub = raw_lats[lat_mask]

    # Shared colour scale across all panels
    all_valid = raw_sub[np.isfinite(raw_sub)]
    vmin, vmax = np.nanpercentile(all_valid, [2, 98]) if all_valid.size > 0 else (0, 1)
    pkw = {"vmin": vmin, "vmax": vmax, "cmap": "Blues"}

    # --- Layout ---
    n_rows = 1 + len(METHODS)
    fig, axes = plt.subplots(
        n_rows,
        N_INNER,
        figsize=(7 * N_INNER, 5 * n_rows),
        constrained_layout=True,
    )

    # Row 0: original AMSR2
    for ch in range(N_INNER):
        ax = axes[0, ch]
        pcm = ax.pcolormesh(raw_lons_sub, raw_lats_sub, raw_sub[:, :, ch], **pkw)
        ax.plot(b_lons, b_lats, color="red", linewidth=1.2, label="LIS domain")
        ax.set_title(f"Original AMSR2 — {INNER_LABELS[ch]}", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(fontsize=8)
        fig.colorbar(pcm, ax=ax, label="Snow depth (cm × 0.1)", shrink=0.85)

    # Rows 1+: regridded outputs
    for row_idx, method in enumerate(METHODS, start=1):
        arr = regridded[method]  # (1750, 2100, 2)
        for ch in range(N_INNER):
            ax = axes[row_idx, ch]
            pcm = ax.pcolormesh(lis_lons, lis_lats, arr[:, :, ch], **pkw)
            ax.set_title(f"{METHOD_LABELS[method]} — {INNER_LABELS[ch]}", fontsize=11)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(pcm, ax=ax, label="Snow depth (cm × 0.1)", shrink=0.85)

    first_stem = amsr2_files[0].rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    fname = first_stem + "_regrid_comparison.png"
    out_path = FIGURES_DIR / fname
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Figure saved to %s", out_path)


if __name__ == "__main__":
    main()
