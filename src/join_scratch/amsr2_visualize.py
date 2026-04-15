#!/usr/bin/env python
"""Visualize AMSR2 regridding results for the first AMSR2 file.

Produces a single PNG in _figures/ with:
  - Top row: original AMSR2 data (global, lat/lon) with LIS domain boundary
    overlaid as a line, one panel per inner dimension slice
  - Subsequent rows: regridded data on the LIS grid for each method
    (xesmf_bilinear, kd_nearest, kd_gauss, pr_bilinear)

The inner dimension of size 2 (two retrieval channels) is shown side-by-side
in each row.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from join_scratch.amsr2_regrid import (
    AMSR2_GLOB,
    DATA_RAW,
    LIS_PATH,
    WEIGHTS_PATH,
    build_amsr2_swath_definition,
    build_lis_area_definition,
    get_regridder,
    load_amsr2,
    load_lis_grid,
    regrid_bilinear_pyresample,
    regrid_kd_gauss,
    regrid_kd_nearest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "_figures"

# Number of inner dimension slices (retrieval channels)
N_INNER = 2
INNER_LABELS = ["Channel 0", "Channel 1"]

METHODS = [
    "xesmf_bilinear",
    "kd_nearest",
    "kd_gauss",
    "pr_bilinear",
]
METHOD_LABELS = {
    "xesmf_bilinear": "xESMF bilinear",
    "kd_nearest": "pyresample kd_tree nearest",
    "kd_gauss": "pyresample kd_tree Gauss",
    "pr_bilinear": "pyresample bilinear",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lis_boundary_lonlat(lis_area) -> tuple[np.ndarray, np.ndarray]:
    """Return (lons, lats) arrays tracing the LIS domain boundary (closed loop)."""
    lons, lats = lis_area.get_lonlats()
    ny, nx = lons.shape
    # top edge (left to right), right edge (top to bottom),
    # bottom edge (right to left), left edge (bottom to top)
    b_lons = np.concatenate(
        [
            lons[0, :],
            lons[:, -1],
            lons[-1, ::-1],
            lons[::-1, 0],
            [lons[0, 0]],  # close the loop
        ]
    )
    b_lats = np.concatenate(
        [
            lats[0, :],
            lats[:, -1],
            lats[-1, ::-1],
            lats[::-1, 0],
            [lats[0, 0]],
        ]
    )
    return b_lons, b_lats


def _pcolormesh_kwargs(data: np.ndarray) -> dict:
    """Compute robust vmin/vmax from the 2nd–98th percentile of valid data."""
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return {}
    vmin, vmax = np.nanpercentile(valid, [2, 98])
    return {"vmin": vmin, "vmax": vmax, "cmap": "Blues"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    amsr2_files = sorted(DATA_RAW.glob(AMSR2_GLOB))
    if not amsr2_files:
        raise FileNotFoundError(f"No AMSR2 files found under {DATA_RAW}")

    log.info("Loading data …")
    lis_grid = load_lis_grid(LIS_PATH)
    amsr2_ds = load_amsr2(amsr2_files[0])
    lis_area = build_lis_area_definition(LIS_PATH)
    amsr2_swath = build_amsr2_swath_definition(amsr2_ds)
    lis_lons, lis_lats = lis_area.get_lonlats()

    # --- Compute all regridded outputs ---
    log.info("Regridding with xESMF …")
    xesmf_regridder = get_regridder(amsr2_ds[["lat", "lon"]], lis_grid, WEIGHTS_PATH)
    xesmf_out = xesmf_regridder(amsr2_ds)
    # shape: (phony_dim_2=2, north_south=1750, east_west=2100)
    # xesmf result has dims matching the LIS grid; extract as (NY, NX, 2)
    geo_data_xesmf = xesmf_out["Geophysical Data"].values
    # xesmf output may be (2, 1750, 2100) – check and transpose if needed
    if geo_data_xesmf.shape[0] == N_INNER:
        geo_data_xesmf = np.moveaxis(geo_data_xesmf, 0, -1)

    log.info("Regridding with kd_nearest …")
    kd_nearest_out = regrid_kd_nearest(amsr2_ds, amsr2_swath, lis_area)

    log.info("Regridding with kd_gauss …")
    kd_gauss_out = regrid_kd_gauss(amsr2_ds, amsr2_swath, lis_area)

    log.info("Regridding with pr_bilinear …")
    pr_bilinear_out = regrid_bilinear_pyresample(amsr2_ds, amsr2_swath, lis_area)

    regridded = {
        "xesmf_bilinear": geo_data_xesmf,
        "kd_nearest": kd_nearest_out,
        "kd_gauss": kd_gauss_out,
        "pr_bilinear": pr_bilinear_out,
    }

    # --- Build boundary for overlay ---
    log.info("Building LIS boundary …")
    b_lons, b_lats = _lis_boundary_lonlat(lis_area)

    # --- Raw AMSR2 data ---
    raw_data = amsr2_ds["Geophysical Data"].values  # (1800, 3600, 2)
    raw_lons = amsr2_ds["lon"].values
    raw_lats = amsr2_ds["lat"].values

    # --- Subset raw data to a region slightly larger than LIS for clarity ---
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

    # --- Layout: rows = [original] + methods, cols = inner channels ---
    n_rows = 1 + len(METHODS)
    n_cols = N_INNER
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        constrained_layout=True,
    )

    # Shared colour scale across all panels (use the original data range)
    all_valid = raw_sub[np.isfinite(raw_sub)]
    vmin, vmax = np.nanpercentile(all_valid, [2, 98]) if all_valid.size > 0 else (0, 1)
    pkw = {"vmin": vmin, "vmax": vmax, "cmap": "Blues"}

    # Row 0: original AMSR2 + LIS boundary overlay
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

    fname = amsr2_files[0].stem + "_regrid_comparison.png"
    out_path = FIGURES_DIR / fname
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Figure saved to %s", out_path)


if __name__ == "__main__":
    main()
