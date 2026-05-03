#!/usr/bin/env python
"""Visualize regridded SWE datasets using the combined swe_combined.nc output.

For each regridded variable, produces:
  1. Side-by-side map: original source (cropped + LIS boundary) vs regridded.
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
import re
import sys
import tempfile
import os
from collections import defaultdict
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent))

from join_scratch.datasets import (
    Amsr2FileHandler,
    CedaFileHandler,
    Icesat2FileHandler,
    ViirsFileHandler,
)
from s3_utils import _is_s3, make_fs, make_store, list_s3, handler_from_s3
from visualize_utils import add_map_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

PC = ccrs.PlateCarree()
PROJ = ccrs.LambertConformal(central_longitude=-96.0, standard_parallels=(39.0, 46.0))


# ── S3 / file helpers ─────────────────────────────────────────────────────────

def _list_files(dir_path: str, suffix: str, fs=None) -> list[str]:
    if _is_s3(dir_path):
        without_scheme = dir_path[len("s3://"):]
        bucket, _, prefix = without_scheme.partition("/")
        store = make_store(bucket, prefix=prefix)
        keys = list_s3(store)
        prefix_slash = prefix.rstrip("/") + "/" if prefix else ""
        return sorted(f"s3://{bucket}/{prefix_slash}{k}" for k in keys if k.endswith(suffix))
    else:
        return [str(p) for p in sorted(Path(dir_path).glob(f"**/*{suffix}"))]


def _path_name(p: str) -> str:
    return p.rstrip("/").split("/")[-1]


def _open_nc(path: str, fs=None) -> xr.Dataset:
    """Open a NetCDF file from local path or S3 into an xarray Dataset."""
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
    """Draw the LIS domain boundary as a polygon outline."""
    b_lons = np.concatenate(
        [lis_lons_2d[0, :], lis_lons_2d[:, -1], lis_lons_2d[-1, ::-1], lis_lons_2d[::-1, 0]]
    )
    b_lats = np.concatenate(
        [lis_lats_2d[0, :], lis_lats_2d[:, -1], lis_lats_2d[-1, ::-1], lis_lats_2d[::-1, 0]]
    )
    ax.plot(b_lons, b_lats, transform=PC, color="red", linewidth=1.0,
            linestyle="-", label="LIS domain")


def _make_map_panel(ax, data, lons, lats, vmin, vmax, title, extent,
                    lis_lons_2d=None, lis_lats_2d=None,
                    cmap="Blues", scatter=False,
                    scatter_lons=None, scatter_lats=None):
    """Plot one map panel; handles gridded pcolormesh or scatter points."""
    ax.set_extent(extent, crs=PC)
    if scatter:
        im = ax.scatter(scatter_lons, scatter_lats, c=data,
                        s=1, transform=PC, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    else:
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
    """Draw a rectangle for a zoom polygon on a map axis."""
    xs = [lon_min, lon_max, lon_max, lon_min, lon_min]
    ys = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(xs, ys, transform=PC, color=color, linewidth=lw, linestyle="--")


# ── source data loaders ───────────────────────────────────────────────────────

def _load_amsr2_source(amsr2_dir: str, fs=None):
    """Load first AMSR2 file. Returns (lons_1d, lats_1d, data_mean, data_unc)."""
    files = _list_files(amsr2_dir, ".h5", fs=fs)
    if not files:
        return None
    path = files[0]
    log.info("Loading AMSR2 source: %s", _path_name(path))
    handler = handler_from_s3(Amsr2FileHandler, path, fs=fs) if _is_s3(path) else Amsr2FileHandler.from_path(path)
    da = handler.get_dataset().rename({"y": "lat", "x": "lon"}).sortby("lat")
    lons = da["lon"].values
    lats = da["lat"].values
    mean_vals = da.isel(inner=0).values.astype(float)
    unc_vals = da.isel(inner=1).values.astype(float)
    mean_vals[mean_vals <= 0] = np.nan
    unc_vals[unc_vals <= 0] = np.nan
    return lons, lats, mean_vals, unc_vals


def _load_ceda_source(ceda_dir: str, fs=None):
    """Load first CEDA file. Returns (lons_1d, lats_1d, swe, swe_std)."""
    files = _list_files(ceda_dir, ".nc", fs=fs)
    if not files:
        return None
    # pick the deepest directory file (skip top-level index files)
    files_deep = [f for f in files if f.count("/") > (files[0].count("/") + 1)] or files
    path = files_deep[0]
    log.info("Loading CEDA source: %s", _path_name(path))
    handler = handler_from_s3(CedaFileHandler, path, fs=fs) if _is_s3(path) else CedaFileHandler.from_path(path)
    ds = handler.get_dataset()
    lats = ds["lat"].values if "lat" in ds else ds["y"].values
    lons = ds["lon"].values if "lon" in ds else ds["x"].values
    if lats.ndim == 2:
        lats = lats[:, 0]
    if lons.ndim == 2:
        lons = lons[0, :]
    lats = np.sort(np.unique(lats))
    lons = np.sort(np.unique(lons))
    ds2 = ds.swap_dims({"y": "lat", "x": "lon"})
    swe = ds2["swe"].values.astype(float)
    swe_std = ds2["swe_std"].values.astype(float)
    swe[swe < 0] = np.nan
    swe_std[swe_std < 0] = np.nan
    return lons, lats, swe, swe_std


def _load_viirs_source(viirs_dir: str, lis_lons, lis_lats, fs=None):
    """Load the first date group of VIIRS tiles. Returns list of (data, lons, lats)."""
    files = _list_files(viirs_dir, ".h5", fs=fs)
    if not files:
        return None
    date_groups: dict[str, list[str]] = defaultdict(list)
    for p in files:
        fname = _path_name(p)
        parts = fname.rsplit(".", 1)[0].split(".")
        date_key = parts[1] if len(parts) > 1 else fname
        date_groups[date_key].append(p)
    date_key, paths = next(iter(sorted(date_groups.items())))

    # Filter to tiles near the LIS domain (same logic as full-workflow)
    import math
    def _tile_bbox(h, v):
        lat_max = 90.0 - v * 10.0
        lat_min = lat_max - 10.0
        def hw(lat_deg):
            r = math.radians(abs(lat_deg))
            c = math.cos(r) if r < math.pi / 2 else 1e-9
            return 10.0 / c
        w = max(hw(lat_min), hw(lat_max))
        lon_c = -180.0 + (h + 0.5) * (360.0 / 36.0)
        return lon_c - w, lat_min, lon_c + w, lat_max

    lon_min_lis, lon_max_lis = float(lis_lons.min()) - 2, float(lis_lons.max()) + 2
    lat_min_lis, lat_max_lis = float(lis_lats.min()) - 2, float(lis_lats.max()) + 2

    tiles = []
    for p in paths:
        fname = _path_name(p)
        m = re.search(r"\.h(\d{2})v(\d{2})\.", fname)
        if m:
            h, v = int(m.group(1)), int(m.group(2))
            tlo_min, tla_min, tlo_max, tla_max = _tile_bbox(h, v)
            if (tlo_max < lon_min_lis or tlo_min > lon_max_lis or
                    tla_max < lat_min_lis or tla_min > lat_max_lis):
                continue
        handler = handler_from_s3(ViirsFileHandler, p, fs=fs) if _is_s3(p) else ViirsFileHandler.from_path(p)
        da = handler.get_dataset()
        swath = da.attrs["area"]
        tile_data = da.values.astype(float)
        tile_lons = swath.lons.values
        tile_lats = swath.lats.values
        # mask fill values
        tile_data[(tile_data < 0) | (tile_data > 200)] = np.nan
        tiles.append((tile_data, tile_lons, tile_lats))
    log.info("Loaded %d VIIRS tiles for date %s", len(tiles), date_key)
    return tiles


def _load_icesat2_source(parquet_path: str, fs=None):
    """Load ICESat-2 ATL06 point cloud. Returns (lons, lats, vals)."""
    handler = Icesat2FileHandler.from_path(parquet_path)
    da = handler.get_dataset()
    src_area = da.attrs["area"]
    lons = src_area.lons.values
    lats = src_area.lats.values
    vals = da.values.astype(float)
    return lons, lats, vals


# ── crop helpers ──────────────────────────────────────────────────────────────

def _crop_2d_grid(data, lons, lats, lon_min, lon_max, lat_min, lat_max):
    """Mask a 2-D grid to a lon/lat bounding box.

    Works for both 1-D (regular) and 2-D (projected) coordinate arrays.
    Returns the same shaped arrays with out-of-bbox values set to NaN.
    For 1-D inputs, slices the arrays; for 2-D inputs, returns full arrays masked.
    """
    if lons.ndim == 1 and lats.ndim == 1:
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        return (
            data[np.ix_(lat_mask, lon_mask)],
            lons[lon_mask],
            lats[lat_mask],
        )
    else:
        # 2-D projected grid: mask out-of-bbox as NaN, return full arrays
        bbox_mask = (
            (lons >= lon_min) & (lons <= lon_max) &
            (lats >= lat_min) & (lats <= lat_max)
        )
        masked = data.astype(float).copy()
        masked[~bbox_mask] = np.nan
        return masked, lons, lats


def _crop_scatter(vals, lons, lats, lon_min, lon_max, lat_min, lat_max):
    mask = (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)
    return vals[mask], lons[mask], lats[mask]


# ── distribution plots ────────────────────────────────────────────────────────

def _kde_cdf_figure(src_vals_flat, rg_vals_flat, title: str, var_label: str):
    """Return (kde_fig, cdf_fig) for source vs regridded distributions."""
    src = src_vals_flat[np.isfinite(src_vals_flat)]
    rg = rg_vals_flat[np.isfinite(rg_vals_flat)]
    if len(src) < 5 or len(rg) < 5:
        return None, None

    # shared range
    lo = float(np.percentile(np.concatenate([src, rg]), 1))
    hi = float(np.percentile(np.concatenate([src, rg]), 99))
    if lo >= hi:
        return None, None
    xs = np.linspace(lo, hi, 500)

    # KDE
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

    # CDF + KS test
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


# ── per-dataset visualization ─────────────────────────────────────────────────

class DatasetViz:
    """Container for one dataset's source + regridded data and metadata."""

    def __init__(self, name, var_label, cmap="Blues"):
        self.name = name          # short identifier (for filenames)
        self.var_label = var_label
        self.cmap = cmap

    def source_overview_im(self, ax_src, extent, lis_lons_2d, lis_lats_2d):
        raise NotImplementedError

    def regrid_data_at_extent(self, lon_min, lon_max, lat_min, lat_max):
        """Return flat array of regridded values within the given bbox."""
        raise NotImplementedError

    def source_data_at_extent(self, lon_min, lon_max, lat_min, lat_max):
        """Return flat array of source values within the given bbox."""
        raise NotImplementedError


def _make_overview_figure(viz_list, lis_lons, lis_lats, lis_lons_2d, lis_lats_2d,
                           extent, polygons, output_dir, fs):
    """For each (source_key, regrid_var) pair, make the overview map PNG."""
    pass  # implemented inline below


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize regridded SWE data from the combined NetCDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lis-path", required=True,
                        help="LIS input NetCDF (local or s3://).")
    parser.add_argument("--regridded-path", required=True,
                        help="Combined swe_combined.nc (local or s3://).")
    parser.add_argument("--amsr2-dir", default=None)
    parser.add_argument("--ceda-dir", default=None)
    parser.add_argument("--viirs-dir", default=None)
    parser.add_argument("--icesat2-parquet", default=None)
    parser.add_argument("--polygons-path", default="zoom_polygons.geojson",
                        help="GeoJSON with zoom polygon features.")
    parser.add_argument("--output-dir", default="_figures/swe",
                        help="Directory for output PNGs (local or s3://).")
    ns = parser.parse_args()

    fs = make_fs() if any(_is_s3(p) for p in [
        ns.lis_path, ns.regridded_path,
        ns.amsr2_dir or "", ns.ceda_dir or "",
        ns.viirs_dir or "", ns.icesat2_parquet or "",
        ns.output_dir,
    ]) else None

    if not _is_s3(ns.output_dir):
        Path(ns.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load LIS grid ──
    log.info("Loading LIS grid from %s", ns.lis_path)
    lis_raw = _open_nc(ns.lis_path, fs=fs)
    lis = lis_raw.isel(north_south=slice(None, None, -1))
    lis_lats_2d = lis["lat"].values   # 2-D (north_south, east_west)
    lis_lons_2d = lis["lon"].values   # 2-D
    lis_lats = lis_lats_2d            # keep alias for extent/filter helpers
    lis_lons = lis_lons_2d

    # LIS domain extent + buffer for source panels
    pad = 2.0
    extent = [
        float(lis_lons.min()) - pad, float(lis_lons.max()) + pad,
        float(lis_lats.min()) - pad, float(lis_lats.max()) + pad,
    ]
    lon_min_e, lon_max_e, lat_min_e, lat_max_e = extent

    # ── Load combined regridded data ──
    log.info("Loading regridded data from %s", ns.regridded_path)
    ds_rg = _open_nc(ns.regridded_path, fs=fs)
    rg_lats = ds_rg["lat"].values
    rg_lons = ds_rg["lon"].values

    # ── Load polygons ──
    poly_path = Path(ns.polygons_path)
    with open(poly_path) as f:
        geojson = json.load(f)
    polygons = []
    for feat in geojson["features"]:
        name = feat["properties"].get("name", "polygon")
        coords = feat["geometry"]["coordinates"][0]  # exterior ring
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        poly_lon_min, poly_lon_max = min(xs), max(xs)
        poly_lat_min, poly_lat_max = min(ys), max(ys)
        polygons.append({
            "name": name,
            "coords": coords,
            "lon_min": poly_lon_min, "lon_max": poly_lon_max,
            "lat_min": poly_lat_min, "lat_max": poly_lat_max,
        })
    log.info("Loaded %d zoom polygons", len(polygons))

    # ── Define per-dataset configs ──
    datasets = []

    # AMSR2 mean snow depth
    if ns.amsr2_dir and "amsr2_snow_depth_mean" in ds_rg:
        src = _load_amsr2_source(ns.amsr2_dir, fs=fs)
        if src:
            lons_s, lats_s, mean_s, unc_s = src
            datasets.append({
                "name": "amsr2_snow_depth_mean",
                "title": "AMSR2 Snow Depth (mean)",
                "var_label": "Snow depth (mm)",
                "cmap": "Blues",
                "rg_data": ds_rg["amsr2_snow_depth_mean"].values,
                "src_lons": lons_s, "src_lats": lats_s,
                "src_data": mean_s,
                "src_scatter": False,
                "src_title": "Source AMSR2 (~0.1°)",
            })
            datasets.append({
                "name": "amsr2_snow_depth_uncertainty",
                "title": "AMSR2 Snow Depth (uncertainty)",
                "var_label": "Snow depth uncertainty (mm)",
                "cmap": "Oranges",
                "rg_data": ds_rg["amsr2_snow_depth_uncertainty"].values,
                "src_lons": lons_s, "src_lats": lats_s,
                "src_data": unc_s,
                "src_scatter": False,
                "src_title": "Source AMSR2 (~0.1°)",
            })

    # CEDA SWE
    if ns.ceda_dir and "ceda_swe" in ds_rg:
        src = _load_ceda_source(ns.ceda_dir, fs=fs)
        if src:
            lons_s, lats_s, swe_s, swe_std_s = src
            datasets.append({
                "name": "ceda_swe",
                "title": "CEDA ESA CCI SWE",
                "var_label": "SWE (mm)",
                "cmap": "Blues",
                "rg_data": ds_rg["ceda_swe"].values,
                "src_lons": lons_s, "src_lats": lats_s,
                "src_data": swe_s,
                "src_scatter": False,
                "src_title": "Source CEDA (~0.25°)",
            })
            datasets.append({
                "name": "ceda_swe_std",
                "title": "CEDA ESA CCI SWE Std",
                "var_label": "SWE std (mm)",
                "cmap": "Oranges",
                "rg_data": ds_rg["ceda_swe_std"].values,
                "src_lons": lons_s, "src_lats": lats_s,
                "src_data": swe_std_s,
                "src_scatter": False,
                "src_title": "Source CEDA (~0.25°)",
            })

    # VIIRS snow cover
    if ns.viirs_dir and "viirs_cgf_ndsi_snow_cover" in ds_rg:
        tiles = _load_viirs_source(ns.viirs_dir, lis_lons, lis_lats, fs=fs)
        if tiles:
            datasets.append({
                "name": "viirs_cgf_ndsi_snow_cover",
                "title": "VIIRS CGF NDSI Snow Cover",
                "var_label": "NDSI snow cover (%)",
                "cmap": "Blues",
                "rg_data": ds_rg["viirs_cgf_ndsi_snow_cover"].values,
                "src_tiles": tiles,
                "src_scatter": False,
                "src_title": "Source VIIRS (sinusoidal 500m tiles)",
                "is_viirs": True,
            })

    # ICESat-2
    if ns.icesat2_parquet and "icesat2_h_li" in ds_rg:
        lons_i, lats_i, vals_i = _load_icesat2_source(ns.icesat2_parquet, fs=fs)
        datasets.append({
            "name": "icesat2_h_li",
            "title": "ICESat-2 ATL06 Land-ice Height",
            "var_label": "h_li (m)",
            "cmap": "viridis",
            "rg_data": ds_rg["icesat2_h_li"].values,
            "src_lons": lons_i, "src_lats": lats_i,
            "src_data": vals_i,
            "src_scatter": True,
            "src_title": "Source ATL06 points",
        })

    log.info("Processing %d dataset(s)", len(datasets))

    for ds_info in datasets:
        name = ds_info["name"]
        rg_data = ds_info["rg_data"]
        cmap = ds_info["cmap"]
        var_label = ds_info["var_label"]
        is_viirs = ds_info.get("is_viirs", False)
        log.info("Visualizing %s", name)

        # Crop regridded to domain extent
        rg_crop, rg_lons_c, rg_lats_c = _crop_2d_grid(
            rg_data, rg_lons, rg_lats, lon_min_e, lon_max_e, lat_min_e, lat_max_e
        )

        # Common color scale from regridded valid data
        rg_valid = rg_data[np.isfinite(rg_data)]
        if len(rg_valid) == 0:
            log.warning("No valid regridded data for %s, skipping", name)
            continue
        vmin = float(np.percentile(rg_valid, 1))
        vmax = float(np.percentile(rg_valid, 99))

        # ── Overview figure ──
        fig, (ax_src, ax_rg) = plt.subplots(
            1, 2, subplot_kw={"projection": PROJ}, figsize=(15, 7)
        )
        fig.suptitle(ds_info["title"], fontsize=10)

        if is_viirs:
            # Plot VIIRS tiles individually
            ax_src.set_extent(extent, crs=PC)
            im = None
            for tile_data, tile_lons, tile_lats in ds_info["src_tiles"]:
                d = tile_data.copy()
                d[(tile_lons < lon_min_e) | (tile_lons > lon_max_e) |
                  (tile_lats < lat_min_e) | (tile_lats > lat_max_e)] = np.nan
                im = ax_src.pcolormesh(
                    tile_lons, tile_lats, d, transform=PC,
                    vmin=vmin, vmax=vmax, cmap=cmap, shading="auto", rasterized=True
                )
            add_map_features(ax_src)
            _draw_lis_boundary(ax_src, lis_lons_2d, lis_lats_2d)
            for poly in polygons:
                _draw_polygon_box(ax_src, poly["lon_min"], poly["lon_max"],
                                  poly["lat_min"], poly["lat_max"])
            ax_src.set_title(ds_info["src_title"], fontsize=8)
            if im:
                plt.colorbar(im, ax=ax_src, orientation="horizontal",
                             pad=0.02, label=var_label)
        else:
            src_lons = ds_info["src_lons"]
            src_lats = ds_info["src_lats"]
            src_data = ds_info["src_data"]
            scatter = ds_info["src_scatter"]
            if scatter:
                src_c, src_lo_c, src_la_c = _crop_scatter(
                    src_data, src_lons, src_lats, lon_min_e, lon_max_e, lat_min_e, lat_max_e
                )
                im = _make_map_panel(
                    ax_src, src_c, None, None, vmin, vmax, ds_info["src_title"], extent,
                    lis_lons_2d=lis_lons_2d, lis_lats_2d=lis_lats_2d,
                    cmap=cmap, scatter=True, scatter_lons=src_lo_c, scatter_lats=src_la_c,
                )
            else:
                src_crop, src_lo_c, src_la_c = _crop_2d_grid(
                    src_data, src_lons, src_lats, lon_min_e, lon_max_e, lat_min_e, lat_max_e
                )
                im = _make_map_panel(
                    ax_src, src_crop, src_lo_c, src_la_c, vmin, vmax,
                    ds_info["src_title"], extent,
                    lis_lons_2d=lis_lons_2d, lis_lats_2d=lis_lats_2d, cmap=cmap,
                )
            # Draw polygon boxes on source panel
            for poly in polygons:
                _draw_polygon_box(ax_src, poly["lon_min"], poly["lon_max"],
                                  poly["lat_min"], poly["lat_max"])
            if im:
                plt.colorbar(im, ax=ax_src, orientation="horizontal",
                             pad=0.02, label=var_label)

        # Regridded panel
        im_rg = _make_map_panel(
            ax_rg, rg_crop, rg_lons_c, rg_lats_c, vmin, vmax,
            "Regridded (LIS 1km)", extent, cmap=cmap,
        )
        for poly in polygons:
            _draw_polygon_box(ax_rg, poly["lon_min"], poly["lon_max"],
                              poly["lat_min"], poly["lat_max"])
        if im_rg:
            plt.colorbar(im_rg, ax=ax_rg, orientation="horizontal",
                         pad=0.02, label=var_label)

        fig.tight_layout()
        _save_figure(fig, _output_path(ns.output_dir, f"{name}_overview.png"), fs=fs)
        plt.close(fig)

        # ── Per-polygon figures (skip for ICESat-2: data too sparse) ──
        if name == "icesat2_h_li":
            continue
        for poly in polygons:
            pname = poly["name"]
            plo_min, plo_max = poly["lon_min"], poly["lon_max"]
            pla_min, pla_max = poly["lat_min"], poly["lat_max"]
            poly_extent = [plo_min - 0.1, plo_max + 0.1, pla_min - 0.1, pla_max + 0.1]
            poly_label = f"{name} / {pname}"

            # Regridded values in polygon
            rg_poly_crop, _, _ = _crop_2d_grid(
                rg_data, rg_lons, rg_lats, plo_min, plo_max, pla_min, pla_max
            )
            rg_poly_flat = rg_poly_crop.ravel()

            # Source values in polygon
            if is_viirs:
                src_poly_flat = []
                for tile_data, tile_lons, tile_lats in ds_info["src_tiles"]:
                    mask = (
                        (tile_lons >= plo_min) & (tile_lons <= plo_max) &
                        (tile_lats >= pla_min) & (tile_lats <= pla_max)
                    )
                    src_poly_flat.append(tile_data[mask])
                src_poly_flat = np.concatenate(src_poly_flat) if src_poly_flat else np.array([])
            else:
                scatter = ds_info["src_scatter"]
                if scatter:
                    src_poly_flat, _, _ = _crop_scatter(
                        ds_info["src_data"], ds_info["src_lons"], ds_info["src_lats"],
                        plo_min, plo_max, pla_min, pla_max
                    )
                else:
                    crop, _, _ = _crop_2d_grid(
                        ds_info["src_data"], ds_info["src_lons"], ds_info["src_lats"],
                        plo_min, plo_max, pla_min, pla_max
                    )
                    src_poly_flat = crop.ravel()

            # ── Zoomed map ──
            fig_z, (ax_zs, ax_zr) = plt.subplots(
                1, 2, subplot_kw={"projection": PROJ}, figsize=(12, 6)
            )
            fig_z.suptitle(f"{ds_info['title']} — zoom: {pname}", fontsize=9)

            if is_viirs:
                ax_zs.set_extent(poly_extent, crs=PC)
                im_z = None
                for tile_data, tile_lons, tile_lats in ds_info["src_tiles"]:
                    # skip tiles with no overlap with this polygon
                    if (tile_lons.max() < poly_extent[0] or tile_lons.min() > poly_extent[1] or
                            tile_lats.max() < poly_extent[2] or tile_lats.min() > poly_extent[3]):
                        continue
                    im_z = ax_zs.pcolormesh(
                        tile_lons, tile_lats, tile_data, transform=PC,
                        vmin=vmin, vmax=vmax, cmap=cmap, shading="auto", rasterized=True
                    )
                add_map_features(ax_zs)
                ax_zs.set_title(ds_info["src_title"], fontsize=8)
                if im_z is not None:
                    plt.colorbar(im_z, ax=ax_zs, orientation="horizontal",
                                 pad=0.02, label=var_label)
            else:
                if scatter:
                    sc, slo, sla = _crop_scatter(
                        ds_info["src_data"], ds_info["src_lons"], ds_info["src_lats"],
                        poly_extent[0], poly_extent[1], poly_extent[2], poly_extent[3]
                    )
                    im = _make_map_panel(
                        ax_zs, sc, None, None, vmin, vmax, ds_info["src_title"], poly_extent,
                        cmap=cmap, scatter=True, scatter_lons=slo, scatter_lats=sla,
                    )
                else:
                    sc, slo, sla = _crop_2d_grid(
                        ds_info["src_data"], ds_info["src_lons"], ds_info["src_lats"],
                        poly_extent[0], poly_extent[1], poly_extent[2], poly_extent[3]
                    )
                    im = _make_map_panel(
                        ax_zs, sc, slo, sla, vmin, vmax, ds_info["src_title"], poly_extent,
                        cmap=cmap,
                    )
                if im:
                    plt.colorbar(im, ax=ax_zs, orientation="horizontal",
                                 pad=0.02, label=var_label)

            # Regridded zoom
            rz, rzlo, rzla = _crop_2d_grid(
                rg_data, rg_lons, rg_lats,
                poly_extent[0], poly_extent[1], poly_extent[2], poly_extent[3]
            )
            im_rz = _make_map_panel(
                ax_zr, rz, rzlo, rzla, vmin, vmax,
                "Regridded (LIS 1km)", poly_extent, cmap=cmap,
            )
            if im_rz:
                plt.colorbar(im_rz, ax=ax_zr, orientation="horizontal",
                             pad=0.02, label=var_label)

            fig_z.tight_layout()
            _save_figure(fig_z, _output_path(ns.output_dir, f"{name}_{pname}_zoom.png"), fs=fs)
            plt.close(fig_z)

            # ── KDE + CDF ──
            kde_fig, cdf_fig = _kde_cdf_figure(
                src_poly_flat, rg_poly_flat, poly_label, var_label
            )
            if kde_fig is not None:
                _save_figure(kde_fig, _output_path(ns.output_dir, f"{name}_{pname}_kde.png"), fs=fs)
                plt.close(kde_fig)
            if cdf_fig is not None:
                _save_figure(cdf_fig, _output_path(ns.output_dir, f"{name}_{pname}_cdf.png"), fs=fs)
                plt.close(cdf_fig)

    log.info("Done. All figures saved to %s", ns.output_dir)


if __name__ == "__main__":
    main()
