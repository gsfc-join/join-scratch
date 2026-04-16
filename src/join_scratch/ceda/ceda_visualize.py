#!/usr/bin/env python
"""Visualize CEDA ESA CCI SWE regridding results for the first daily file.

Produces a single PNG in _figures/ with:
  - Top row: original CEDA SWE and SWE-std subsetted to the LIS region,
    displayed in geographic (lon/lat) space with the LIS domain boundary
    overlaid as a red line
  - Subsequent rows: regridded outputs on the LIS grid for each method
    (xesmf_bilinear, nearest, bilinear, ewa, bucket_avg)

Two columns: left = swe, right = swe_std.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from join_scratch.amsr2.amsr2_regrid import (
    build_lis_area_definition,
    load_lis_grid,
    load_regridder,
)
from join_scratch.ceda.ceda_regrid import (
    CEDA_GLOB,
    CEDA_VARS,
    DATA_RAW,
    LIS_PATH,
    SATPY_CACHE,
    WEIGHTS_PATH,
    build_ceda_swath_definition,
    load_ceda,
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

METHODS = ["xesmf_bilinear", "nearest", "bilinear", "ewa", "bucket_avg"]
METHOD_LABELS = {
    "xesmf_bilinear": "xESMF bilinear",
    "nearest": "satpy nearest",
    "bilinear": "satpy bilinear",
    "ewa": "satpy EWA",
    "bucket_avg": "satpy bucket avg",
}
VAR_LABELS = {"swe": "SWE (mm)", "swe_std": "SWE std (mm)"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lis_boundary_lonlat(lis_area) -> tuple[np.ndarray, np.ndarray]:
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
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ceda_files = sorted(DATA_RAW.glob(CEDA_GLOB))
    if not ceda_files:
        raise FileNotFoundError(f"No CEDA files found under {DATA_RAW}")

    log.info("Loading data …")
    lis_grid = load_lis_grid(LIS_PATH)
    ceda_ds = load_ceda(ceda_files[0])
    lis_area = build_lis_area_definition(LIS_PATH)
    ceda_swath = build_ceda_swath_definition(ceda_ds)
    lis_lons, lis_lats = lis_area.get_lonlats()

    # --- Compute all regridded outputs ---
    log.info("Regridding with xESMF …")
    xesmf_regridder = load_regridder(ceda_ds[["lat", "lon"]], lis_grid, WEIGHTS_PATH)
    xesmf_out_ds = xesmf_regridder(ceda_ds)
    # shape (NY, NX, n_vars) channel-last to match satpy outputs
    xesmf_arr = np.stack([xesmf_out_ds[v].values for v in CEDA_VARS], axis=-1)

    log.info("Regridding with nearest …")
    nearest_arr = regrid_nearest(ceda_ds, ceda_swath, lis_area, SATPY_CACHE)

    log.info("Regridding with bilinear …")
    bilinear_arr = regrid_bilinear(ceda_ds, ceda_swath, lis_area, SATPY_CACHE)

    log.info("Regridding with EWA …")
    ewa_arr = regrid_ewa(ceda_ds, ceda_swath, lis_area)

    log.info("Regridding with bucket_avg …")
    bucket_arr = regrid_bucket_avg(ceda_ds, ceda_swath, lis_area)

    regridded = {
        "xesmf_bilinear": xesmf_arr,
        "nearest": nearest_arr,
        "bilinear": bilinear_arr,
        "ewa": ewa_arr,
        "bucket_avg": bucket_arr,
    }

    # --- LIS domain boundary ---
    b_lons, b_lats = _lis_boundary_lonlat(lis_area)

    # --- Subset raw CEDA to LIS region + padding ---
    raw_lons = ceda_ds["lon"].values
    raw_lats = ceda_ds["lat"].values
    lat_pad, lon_pad = 3.0, 5.0
    lat_min = float(lis_lats.min()) - lat_pad
    lat_max = float(lis_lats.max()) + lat_pad
    lon_min = float(lis_lons.min()) - lon_pad
    lon_max = float(lis_lons.max()) + lon_pad
    lat_mask = (raw_lats >= lat_min) & (raw_lats <= lat_max)
    lon_mask = (raw_lons >= lon_min) & (raw_lons <= lon_max)
    sub_lons = raw_lons[lon_mask]
    sub_lats = raw_lats[lat_mask]

    n_vars = len(CEDA_VARS)
    n_rows = 1 + len(METHODS)

    fig, axes = plt.subplots(
        n_rows,
        n_vars,
        figsize=(7 * n_vars, 5 * n_rows),
        constrained_layout=True,
    )

    # Per-variable colour scale from raw subsetted data
    pkws = []
    for vi, var in enumerate(CEDA_VARS):
        raw_sub = ceda_ds[var].values[np.ix_(lat_mask, lon_mask)]
        valid = raw_sub[np.isfinite(raw_sub)]
        vmin, vmax = np.nanpercentile(valid, [2, 98]) if valid.size > 0 else (0, 1)
        pkws.append({"vmin": vmin, "vmax": vmax, "cmap": "Blues"})

    # Row 0: original CEDA subsetted
    for vi, var in enumerate(CEDA_VARS):
        ax = axes[0, vi]
        raw_sub = ceda_ds[var].values[np.ix_(lat_mask, lon_mask)]
        pcm = ax.pcolormesh(sub_lons, sub_lats, raw_sub, **pkws[vi])
        ax.plot(b_lons, b_lats, color="red", linewidth=1.2, label="LIS domain")
        ax.set_title(f"Original CEDA — {VAR_LABELS[var]}", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(fontsize=8)
        fig.colorbar(pcm, ax=ax, label=VAR_LABELS[var], shrink=0.85)

    # Rows 1+: regridded
    for row_idx, method in enumerate(METHODS, start=1):
        arr = regridded[method]  # (NY, NX, n_vars)
        for vi, var in enumerate(CEDA_VARS):
            ax = axes[row_idx, vi]
            pcm = ax.pcolormesh(lis_lons, lis_lats, arr[:, :, vi], **pkws[vi])
            ax.set_title(f"{METHOD_LABELS[method]} — {VAR_LABELS[var]}", fontsize=11)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(pcm, ax=ax, label=VAR_LABELS[var], shrink=0.85)

    date_part = ceda_files[0].name.split("-")[0]  # e.g. "20190101"
    fname = f"{date_part}_ceda_regrid_comparison.png"
    out_path = FIGURES_DIR / fname
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Figure saved to %s", out_path)


if __name__ == "__main__":
    main()
