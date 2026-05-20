#!/usr/bin/env python
"""Visualize regridded AMSR2 snow depth output NetCDF files.

Source data is read from the same IceChunk virtual-Zarr store used by
amsr2_regrid.py, ensuring the source and regridded panels always show the
same date(s).

For each regridded variable (snow_depth_ascending, snow_depth_descending,
quality_flag_ascending, quality_flag_descending), produces:
  1. Side-by-side map: original AMSR2 source (cropped + LIS boundary) vs regridded.
  2. For each zoom polygon in --polygons-path GeoJSON:
       a. Zoomed map figure (same left/right style).
       b. Kernel density plot: source vs resampled.
       c. CDF plot with Kolmogorov-Smirnov test statistics.

All figures are saved to --output-dir (local or s3://).
"""

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path

import cartopy.crs as ccrs
import icechunk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.stats import gaussian_kde

os.environ.setdefault("RUST_LOG", "error")

sys.path.insert(0, str(Path(__file__).parent))

from s3_utils import _is_s3, make_fs, make_store
from visualize_utils import add_map_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

PC = ccrs.PlateCarree()
PROJ = ccrs.LambertConformal(central_longitude=-96.0, standard_parallels=(39.0, 46.0))

S3_BUCKET = "airborne-smce-prod-user-bucket"
S3_JOIN_PREFIX = "JOIN"
DEFAULT_LIS_PATH = f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/lis_input_NMP_1000m_missouri.nc"
DEFAULT_STORE_URI = (
    f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/icechunk-stores/GCOM-W1-AMSR2-L3-SND"
)
DEFAULT_OUTPUT_DIR = "_figures/amsr2"

# IceChunk store constants (must match amsr2_regrid.py)
NLAT, NLON = 1800, 3600
SCALE_FACTOR = np.float32(0.1)
FILL_MISSING = np.int16(-32768)
FILL_NOOBS = np.int16(-32767)
_LATS = np.linspace(89.95, -89.95, NLAT).astype("float32")   # north-first
_LONS = np.linspace(0.05, 359.95, NLON).astype("float32")    # 0°–360°, matches store layout

ORBIT_NAMES = ["Ascending", "Descending"]
BAND_NAMES  = ["snow_depth", "quality_flag"]


# ── IceChunk helpers ──────────────────────────────────────────────────────────

def _s3_storage(bucket: str, prefix: str) -> icechunk.Storage:
    return icechunk.s3_storage(bucket=bucket, prefix=prefix, region="us-west-2")


def _open_icechunk(store_uri: str) -> xr.Dataset:
    """Open the AMSR2 IceChunk store as an xarray Dataset (read-only)."""
    without_scheme = store_uri[len("s3://"):]
    bucket, _, prefix = without_scheme.partition("/")
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "https://gportal.jaxa.jp/", icechunk.http_store()
        )
    )
    repo = icechunk.Repository.open(
        _s3_storage(bucket, prefix),
        config=config,
        authorize_virtual_chunk_access={"https://gportal.jaxa.jp/": None},
    )
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, consolidated=False, zarr_format=3,
                        mask_and_scale=False)


def _load_source_slice(ic_ds: xr.Dataset, date_str: str) -> dict[str, np.ndarray]:
    """Return scaled float32 (lat, lon) arrays for each band × orbit combination.

    Keys are like ``"snow_depth_ascending"``.  Out-of-range fill values are
    masked to NaN.  The returned data is on the native 0–360 longitude grid,
    matching the layout used by amsr2_regrid.py.
    """
    target = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
    time_raw = ic_ds["time"].values  # datetime64[ns]
    matches = np.where(time_raw == np.datetime64(target))[0]
    if len(matches) == 0:
        raise ValueError(
            f"Date {date_str} not found in IceChunk store. "
            f"Available range: {time_raw[0]}…{time_raw[-1]}."
        )
    slot = int(matches[0])
    log.info("Reading IceChunk store: date %s → time slot %d", date_str, slot)

    result: dict[str, np.ndarray] = {}
    for o_idx, orbit in enumerate(ORBIT_NAMES):
        for b_idx, band in enumerate(BAND_NAMES):
            raw = (
                ic_ds["geophysical_data"]
                .isel(time=slot)
                .sel(orbit=orbit, band=band)
                .values
            )  # int16, shape (NLAT, NLON)
            data = raw.astype("float32") * SCALE_FACTOR
            data[raw == FILL_MISSING] = np.nan
            data[raw == FILL_NOOBS]   = np.nan
            key = f"{band}_{orbit.lower()}"
            result[key] = data
    return result


# ── S3 / file helpers ─────────────────────────────────────────────────────────

def _open_nc(path: str, fs=None) -> xr.Dataset:
    """Open a NetCDF file from local path or S3."""
    if _is_s3(path):
        if fs is None:
            fs = make_fs()
        with fs.open(path, "rb") as f:
            data = f.read()
        return xr.open_dataset(io.BytesIO(data), engine="h5netcdf")
    return xr.open_dataset(path, engine="h5netcdf")


def _save_figure(fig: plt.Figure, path: str, fs=None, dpi: int = 150) -> None:
    """Save a matplotlib figure to a local path or S3 URI."""
    if _is_s3(path):
        if fs is None:
            fs = make_fs()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        with fs.open(path, "wb") as f:
            f.write(buf.read())
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    log.info("Saved %s", path)


def _output_path(output_dir: str, filename: str) -> str:
    if _is_s3(output_dir):
        return output_dir.rstrip("/") + "/" + filename
    return str(Path(output_dir) / filename)


# ── map helpers ───────────────────────────────────────────────────────────────

def _draw_lis_boundary(ax, lis_lons_2d: np.ndarray, lis_lats_2d: np.ndarray) -> None:
    b_lons = np.concatenate(
        [lis_lons_2d[0, :], lis_lons_2d[:, -1], lis_lons_2d[-1, ::-1], lis_lons_2d[::-1, 0]]
    )
    b_lats = np.concatenate(
        [lis_lats_2d[0, :], lis_lats_2d[:, -1], lis_lats_2d[-1, ::-1], lis_lats_2d[::-1, 0]]
    )
    ax.plot(b_lons, b_lats, transform=PC, color="red", linewidth=1.0,
            linestyle="-", label="LIS domain")


def _make_map_panel(ax, data, lons, lats, vmin, vmax, title, extent, cmap="Blues",
                    lis_lons_2d=None, lis_lats_2d=None):
    """Plot one gridded pcolormesh map panel."""
    ax.set_extent(extent, crs=PC)
    if lons.ndim == 1 and lats.ndim == 1:
        lo2, la2 = np.meshgrid(lons, lats)
    else:
        lo2, la2 = lons, lats
    im = ax.pcolormesh(lo2, la2, data, transform=PC,
                       vmin=vmin, vmax=vmax, cmap=cmap, shading="auto", rasterized=True)
    add_map_features(ax)
    if lis_lons_2d is not None:
        _draw_lis_boundary(ax, lis_lons_2d, lis_lats_2d)
    ax.set_title(title, fontsize=8)
    return im


def _draw_polygon_box(ax, lon_min, lon_max, lat_min, lat_max, color="orange", lw=1.5):
    xs = [lon_min, lon_max, lon_max, lon_min, lon_min]
    ys = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(xs, ys, transform=PC, color=color, linewidth=lw, linestyle="--")


# ── crop helpers ──────────────────────────────────────────────────────────────

def _to_0_360(lon: float) -> float:
    """Convert a –180…180 longitude value to the 0…360 range."""
    return lon % 360.0


def _crop_2d_grid(data, lons, lats, lon_min, lon_max, lat_min, lat_max):
    """Crop/mask a 2-D grid to a lon/lat bounding box."""
    if lons.ndim == 1 and lats.ndim == 1:
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        return (
            data[np.ix_(lat_mask, lon_mask)],
            lons[lon_mask],
            lats[lat_mask],
        )
    else:
        bbox_mask = (
            (lons >= lon_min) & (lons <= lon_max) &
            (lats >= lat_min) & (lats <= lat_max)
        )
        masked = data.astype(float).copy()
        masked[~bbox_mask] = np.nan
        return masked, lons, lats


# ── distribution plots ────────────────────────────────────────────────────────

def _kde_cdf_figure(src_vals_flat, rg_vals_flat, title: str, var_label: str):
    """Return (kde_fig, cdf_fig) for source vs regridded distributions."""
    src = src_vals_flat[np.isfinite(src_vals_flat)]
    rg = rg_vals_flat[np.isfinite(rg_vals_flat)]
    if len(src) < 5 or len(rg) < 5:
        return None, None

    lo = float(np.percentile(np.concatenate([src, rg]), 1))
    hi = float(np.percentile(np.concatenate([src, rg]), 99))
    if lo >= hi:
        return None, None
    xs = np.linspace(lo, hi, 500)

    try:
        kde_src = gaussian_kde(src, bw_method="scott")
        kde_rg = gaussian_kde(rg, bw_method="scott")
    except Exception:
        return None, None

    kde_fig, ax_kde = plt.subplots(figsize=(6, 4))
    ax_kde.plot(xs, kde_src(xs), label="Source (original)", color="steelblue")
    ax_kde.plot(xs, kde_rg(xs), label="Regridded (LIS 1km)", color="darkorange")
    ax_kde.set_xlabel(var_label)
    ax_kde.set_ylabel("Density")
    ax_kde.set_title(f"KDE — {title}", fontsize=9)
    ax_kde.legend(fontsize=8)
    kde_fig.tight_layout()

    ks_stat, ks_p = stats.ks_2samp(src, rg)
    src_sorted = np.sort(src)
    rg_sorted = np.sort(rg)
    src_cdf = np.arange(1, len(src_sorted) + 1) / len(src_sorted)
    rg_cdf = np.arange(1, len(rg_sorted) + 1) / len(rg_sorted)

    cdf_fig, ax_cdf = plt.subplots(figsize=(6, 4))
    ax_cdf.plot(src_sorted, src_cdf, label="Source (original)", color="steelblue")
    ax_cdf.plot(rg_sorted, rg_cdf, label="Regridded (LIS 1km)", color="darkorange")
    ax_cdf.set_xlabel(var_label)
    ax_cdf.set_ylabel("Cumulative probability")
    ax_cdf.set_title(f"CDF — {title}", fontsize=9)
    textstr = f"KS statistic = {ks_stat:.4f}\np-value = {ks_p:.2e}"
    ax_cdf.text(
        0.97, 0.05, textstr, transform=ax_cdf.transAxes,
        fontsize=8, verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    ax_cdf.legend(fontsize=8)
    cdf_fig.tight_layout()

    return kde_fig, cdf_fig


# ── main ──────────────────────────────────────────────────────────────────────

# Variable metadata: (long title, color map, units)
VAR_META = {
    "snow_depth_ascending":    ("AMSR2 Snow Depth – Ascending",    "Blues",   "Snow depth (cm)"),
    "snow_depth_descending":   ("AMSR2 Snow Depth – Descending",   "Blues",   "Snow depth (cm)"),
    "quality_flag_ascending":  ("AMSR2 Quality Flag – Ascending",  "RdYlGn", "Quality flag"),
    "quality_flag_descending": ("AMSR2 Quality Flag – Descending", "RdYlGn", "Quality flag"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Visualize regridded AMSR2 snow depth output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lis-path", default=DEFAULT_LIS_PATH,
                        help="LIS input NetCDF (local or s3://).")
    parser.add_argument("--regridded-path", required=True,
                        help="Regridded AMSR2 NetCDF produced by amsr2_regrid.py (local or s3://).")
    parser.add_argument("--store-uri", default=DEFAULT_STORE_URI,
                        help="S3 URI of the AMSR2 IceChunk store (source data).")
    parser.add_argument("--polygons-path", default="zoom_polygons.geojson",
                        help="GeoJSON with zoom polygon features.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for output PNGs (local or s3://).")
    parser.add_argument("--time-index", type=int, default=0,
                        help="Time index to visualize from the regridded file.")
    ns = parser.parse_args()

    fs = make_fs() if any(_is_s3(p) for p in [
        ns.lis_path, ns.regridded_path, ns.store_uri, ns.output_dir,
    ]) else None

    if not _is_s3(ns.output_dir):
        Path(ns.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load LIS grid ──
    log.info("Loading LIS grid from %s", ns.lis_path)
    lis_raw = _open_nc(ns.lis_path, fs=fs)
    lis = lis_raw.isel(north_south=slice(None, None, -1))
    lis_lats_2d = lis["lat"].values
    lis_lons_2d = lis["lon"].values

    pad = 2.0
    extent = [
        float(lis_lons_2d.min()) - pad, float(lis_lons_2d.max()) + pad,
        float(lis_lats_2d.min()) - pad, float(lis_lats_2d.max()) + pad,
    ]
    lon_min_e, lon_max_e, lat_min_e, lat_max_e = extent

    # ── Load regridded data ──
    log.info("Loading regridded data from %s", ns.regridded_path)
    ds_rg = _open_nc(ns.regridded_path, fs=fs)
    rg_lats = ds_rg["lat"].values   # 2-D (south_north, west_east)
    rg_lons = ds_rg["lon"].values

    # Derive the date string for the selected time step
    time_val = pd.Timestamp(ds_rg["time"].values[ns.time_index])
    date_str = time_val.strftime("%Y%m%d")
    log.info("Visualizing time index %d → date %s", ns.time_index, date_str)

    # ── Load source data from IceChunk store ──
    log.info("Opening IceChunk store at %s", ns.store_uri)
    ic_ds = _open_icechunk(ns.store_uri)
    src_slices = _load_source_slice(ic_ds, date_str)
    # src_slices keys: "snow_depth_ascending", "snow_depth_descending",
    #                  "quality_flag_ascending", "quality_flag_descending"
    # Each value is a float32 (NLAT, NLON) array with lons in 0–360 range,
    # matching the IceChunk store layout (same as used by amsr2_regrid.py).

    # ── Load polygons ──
    poly_path = Path(ns.polygons_path)
    with open(poly_path) as f:
        geojson = json.load(f)
    polygons = []
    for feat in geojson["features"]:
        name = feat["properties"].get("name", "polygon")
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        polygons.append({
            "name": name,
            "lon_min": min(xs), "lon_max": max(xs),
            "lat_min": min(ys), "lat_max": max(ys),
        })
    log.info("Loaded %d zoom polygons", len(polygons))

    # ── Determine which variables to process ──
    avail_vars = [v for v in VAR_META if v in ds_rg]
    if not avail_vars:
        log.error("No recognised AMSR2 variables found in %s. Expected one of: %s",
                  ns.regridded_path, list(VAR_META.keys()))
        sys.exit(1)
    log.info("Processing variables: %s", avail_vars)

    for var_name in avail_vars:
        title, cmap, var_label = VAR_META[var_name]
        log.info("Visualizing %s", var_name)

        # Regridded slice: shape (south_north, west_east)
        rg_data_full = ds_rg[var_name].isel(time=ns.time_index).values.astype(float)

        rg_valid = rg_data_full[np.isfinite(rg_data_full)]
        if len(rg_valid) == 0:
            log.warning("No valid regridded data for %s at time index %d, skipping",
                        var_name, ns.time_index)
            continue

        # Source slice for this variable (already scaled, filled, lon-shifted)
        src_data_full = src_slices[var_name]   # (NLAT, NLON)

        # Shared color scale from the union of source + regridded valid data
        src_valid = src_data_full[np.isfinite(src_data_full)]
        all_valid = np.concatenate([src_valid, rg_valid])
        vmin = float(np.percentile(all_valid, 1))
        vmax = float(np.percentile(all_valid, 99))

        # Crop source to domain extent
        # Crop source to domain extent (source lons are 0–360)
        src_crop, src_lo_c, src_la_c = _crop_2d_grid(
            src_data_full, _LONS, _LATS,
            _to_0_360(lon_min_e), _to_0_360(lon_max_e), lat_min_e, lat_max_e
        )
        rg_crop, rg_lo_c, rg_la_c = _crop_2d_grid(
            rg_data_full, rg_lons, rg_lats, lon_min_e, lon_max_e, lat_min_e, lat_max_e
        )

        # ── Overview figure ──
        fig, (ax_src, ax_rg) = plt.subplots(
            1, 2, subplot_kw={"projection": PROJ}, figsize=(15, 7)
        )
        fig.suptitle(f"{title}  [{date_str}]", fontsize=10)

        im_src = _make_map_panel(
            ax_src, src_crop, src_lo_c, src_la_c, vmin, vmax,
            "Source AMSR2 (~0.1°)", extent, cmap=cmap,
            lis_lons_2d=lis_lons_2d, lis_lats_2d=lis_lats_2d,
        )
        for poly in polygons:
            _draw_polygon_box(ax_src, poly["lon_min"], poly["lon_max"],
                              poly["lat_min"], poly["lat_max"])
        plt.colorbar(im_src, ax=ax_src, orientation="horizontal",
                     pad=0.02, label=var_label)

        im_rg = _make_map_panel(
            ax_rg, rg_crop, rg_lo_c, rg_la_c, vmin, vmax,
            "Regridded (LIS 1km)", extent, cmap=cmap,
        )
        for poly in polygons:
            _draw_polygon_box(ax_rg, poly["lon_min"], poly["lon_max"],
                              poly["lat_min"], poly["lat_max"])
        plt.colorbar(im_rg, ax=ax_rg, orientation="horizontal",
                     pad=0.02, label=var_label)

        fig.tight_layout()
        _save_figure(fig, _output_path(ns.output_dir, f"{var_name}_overview.png"), fs=fs)
        plt.close(fig)

        # ── Per-polygon figures ──
        for poly in polygons:
            pname = poly["name"]
            plo_min, plo_max = poly["lon_min"], poly["lon_max"]
            pla_min, pla_max = poly["lat_min"], poly["lat_max"]
            poly_extent = [plo_min - 0.1, plo_max + 0.1, pla_min - 0.1, pla_max + 0.1]
            poly_label = f"{var_name} / {pname}"

            # Source polygon values (source lons are 0–360)
            src_poly_crop, _, _ = _crop_2d_grid(
                src_data_full, _LONS, _LATS,
                _to_0_360(plo_min), _to_0_360(plo_max), pla_min, pla_max
            )
            src_poly_flat = src_poly_crop.ravel()

            rg_poly_crop, _, _ = _crop_2d_grid(
                rg_data_full, rg_lons, rg_lats, plo_min, plo_max, pla_min, pla_max
            )
            rg_poly_flat = rg_poly_crop.ravel()

            # ── Zoomed map ──
            fig_z, (ax_zs, ax_zr) = plt.subplots(
                1, 2, subplot_kw={"projection": PROJ}, figsize=(12, 6)
            )
            fig_z.suptitle(f"{title}  [{date_str}]  — zoom: {pname}", fontsize=9)

            sc, slo, sla = _crop_2d_grid(
                src_data_full, _LONS, _LATS,
                _to_0_360(poly_extent[0]), _to_0_360(poly_extent[1]),
                poly_extent[2], poly_extent[3]
            )
            im_zs = _make_map_panel(
                ax_zs, sc, slo, sla, vmin, vmax,
                "Source AMSR2 (~0.1°)", poly_extent, cmap=cmap,
            )
            plt.colorbar(im_zs, ax=ax_zs, orientation="horizontal",
                         pad=0.02, label=var_label)

            rz, rzlo, rzla = _crop_2d_grid(
                rg_data_full, rg_lons, rg_lats,
                poly_extent[0], poly_extent[1], poly_extent[2], poly_extent[3]
            )
            im_rz = _make_map_panel(
                ax_zr, rz, rzlo, rzla, vmin, vmax,
                "Regridded (LIS 1km)", poly_extent, cmap=cmap,
            )
            plt.colorbar(im_rz, ax=ax_zr, orientation="horizontal",
                         pad=0.02, label=var_label)

            fig_z.tight_layout()
            _save_figure(fig_z, _output_path(ns.output_dir, f"{var_name}_{pname}_zoom.png"), fs=fs)
            plt.close(fig_z)

            # ── KDE + CDF ──
            kde_fig, cdf_fig = _kde_cdf_figure(
                src_poly_flat, rg_poly_flat, poly_label, var_label
            )
            if kde_fig is not None:
                _save_figure(kde_fig,
                             _output_path(ns.output_dir, f"{var_name}_{pname}_kde.png"), fs=fs)
                plt.close(kde_fig)
            if cdf_fig is not None:
                _save_figure(cdf_fig,
                             _output_path(ns.output_dir, f"{var_name}_{pname}_cdf.png"), fs=fs)
                plt.close(cdf_fig)

    log.info("Done. All figures saved to %s", ns.output_dir)


if __name__ == "__main__":
    main()
