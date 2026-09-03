#!/usr/bin/env python3
"""
Generate geographic maps for SWE/Snow output variables.

This is a clean, from-scratch plotting implementation for 2D (curvilinear)
lat/lon grids, written for Cartopy 0.25 / Matplotlib 3.11.
"""

import argparse
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def find_lat_lon_names(ds: xr.Dataset):
    """Return (lat_name, lon_name) by inspecting standard_name / common names."""
    lat_name = lon_name = None
    for name in ds.variables:
        std = ds[name].attrs.get("standard_name", "").lower()
        low = name.lower()
        if "latitude" in std or low in ("lat", "latitude"):
            lat_name = name
        if "longitude" in std or low in ("lon", "longitude"):
            lon_name = name
    return lat_name, lon_name


def _nice_ticks(vmin: float, vmax: float, n: int = 6):
    """Return ~n evenly spaced, rounded tick locations spanning [vmin, vmax]."""
    locator = mticker.MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10])
    ticks = locator.tick_values(vmin, vmax)
    # keep only ticks inside the domain (with a tiny tolerance)
    span = vmax - vmin
    tol = 0.01 * span if span > 0 else 0.0
    return [t for t in ticks if (vmin - tol) <= t <= (vmax + tol)]


def plot_variable_2d(
    ds: xr.Dataset,
    var: str,
    lat_name: str,
    lon_name: str,
    out_path: Path,
    stride: int = 5,
    use_scatter: bool = True,
    add_features: bool = False,
    title_suffix: str = "",
):
    """
    Plot a single 2D variable on a PlateCarree map.

    Parameters
    ----------
    ds : xr.Dataset
    var : str
        Name of the data variable to plot.
    lat_name, lon_name : str
        Names of the 2D coordinate variables.
    out_path : Path
        Where to write the PNG.
    stride : int
        Decimation stride. A full 1750x2100 masked Quadmesh pushed through
        Cartopy's transform can render blank / very slowly; decimating avoids
        that. Use stride=1 for full resolution once you trust the output.
    use_scatter : bool
        If True (default), plot finite points with scatter -- the most robust
        renderer, immune to the blank-Quadmesh problem. If False, use
        pcolormesh for filled cells.
    add_features : bool
        If True, draw Natural Earth coastlines/borders/states. These require
        the Natural Earth data files (downloaded on first use). Leave False in
        offline environments -- otherwise these features silently draw nothing
        AND you still get a valid data plot, which is exactly the "map missing
        but colorbar fine" symptom. When False we draw an offline lat/lon
        reference grid instead so the axes are still georeferenced visually.
    
    """
   
    # --- pull raw arrays -------------------------------------------------
    lon = np.asarray(ds[lon_name].values, dtype="float64")
    lat = np.asarray(ds[lat_name].values, dtype="float64")
    data = np.asarray(ds[var].values, dtype="float64")

    # --- convert 0-360 longitude to -180..180 if needed ------------------
    if np.nanmax(lon) > 180.0:
        lon = np.where(lon > 180.0, lon - 360.0, lon)

    # --- decimate for a tractable render ---------------------------------
    if stride > 1:
        lon = lon[::stride, ::stride]
        lat = lat[::stride, ::stride]
        data = data[::stride, ::stride]

    # --- basic diagnostics -----------------------------------------------
    finite = np.isfinite(data)
    n_finite = int(finite.sum())
    n_nan = int(np.isnan(data).sum())
    if n_finite == 0:
        log.warning(f"[{var}] no finite values after decimation; skipping.")
        return
    n_zero = int((finite & (data == 0)).sum())
    vmin = float(np.nanmin(data[finite]))
    vmax = float(np.nanmax(data[finite]))
    if vmin == vmax:
        vmax = vmin + 1.0
    log.info(
        f"[{var}] plotting: finite={n_finite}, NaN={n_nan}, zeros={n_zero}, "
        f"vmin={vmin:.4g}, vmax={vmax:.4g}, grid={data.shape}"
    )

    # --- domain extent from finite coords --------------------------------
    lon_f = lon[np.isfinite(lon)]
    lat_f = lat[np.isfinite(lat)]
    x_min, x_max = float(lon_f.min()), float(lon_f.max())
    y_min, y_max = float(lat_f.min()), float(lat_f.max())

    # --- colormap / labels -----------------------------------------------
    cmap = "Blues" if ("swe" in var or "snow" in var) else "viridis"
    unit = ds[var].attrs.get("units", ds[var].attrs.get("unit", ""))
    long_name = ds[var].attrs.get("long_name", var)

    # --- figure ----------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.PlateCarree())

    if use_scatter:
        # Plot only finite points as colored markers positioned by lat/lon.
        # No quad-mesh geometry => cannot hit the blank-Quadmesh failure.
        pts = ax.scatter(
            lon[finite],
            lat[finite],
            c=data[finite],
            s=2,
            marker="s",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            linewidths=0,
            zorder=2,
        )
        mappable = pts
    else:
        masked = np.ma.masked_invalid(data)
        mesh = ax.pcolormesh(
            lon,
            lat,
            masked,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            shading="nearest",
            vmin=vmin,
            vmax=vmax,
            zorder=2,
        )
        mappable = mesh

    # --- geographic reference --------------------------------------------
    if add_features:
        # Requires Natural Earth data (downloaded on first use). In an offline
        # environment these silently draw nothing.
        try:
            ax.coastlines(resolution="110m", linewidth=0.8, color="black",
                          zorder=3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":",
                           zorder=3)
            ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=":",
                           zorder=3)
        except Exception as exc:  # pragma: no cover
            log.warning(f"Could not add Natural Earth features: {exc}")

    # --- offline lat/lon reference grid ----------------------------------
    # This does NOT require any downloads: it's pure Matplotlib tick drawing on
    # the GeoAxes. It guarantees you always get a georeferenced frame with
    # labeled latitude/longitude, even with no network access.
    xticks = _nice_ticks(x_min, x_max, n=7)
    yticks = _nice_ticks(y_min, y_max, n=6)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.grid(True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--",
            zorder=4)
    ax.tick_params(labelsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = plt.colorbar(mappable, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label(unit)

    ax.set_title(f"LIS Domain: {long_name}{title_suffix}", fontsize=14, pad=15)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved plot to {out_path}")


def main(
    nc_path: str = "./_data/swe_combined_20190101.nc",
    plot_dir: str = "./plot",
    stride: int = 5,
    use_scatter: bool = True,
    add_features: bool = False,
):
    nc_file = Path(nc_path)
    if not nc_file.exists():
        log.error(f"Input file not found: {nc_file}")
        return

    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    log.info(f"Opening dataset: {nc_file}")
    ds = xr.open_dataset(nc_file)
    print(ds)

    lat_name, lon_name = find_lat_lon_names(ds)
    if lat_name is None or lon_name is None:
        log.error("Could not identify latitude/longitude variables.")
        return
    log.info(f"Using coordinates: lat='{lat_name}', lon='{lon_name}'")

    # Date used both in the title suffix and appended to output filenames.
    date_str = ds.attrs.get("date", "unknown-date")
    title_suffix = f" ({date_str})"

    variables_to_plot = [
        "amsr2_snow_depth_mean",
        "ceda_swe",
        "viirs_cgf_ndsi_snow_cover",
        "icesat2_snow_depth",
        "icesat2_h_li",
        "icesat2_3dep_dem_10m",
    ]

    for var in variables_to_plot:
        print(f"working on {var}...")
        if var not in ds.variables:
            log.warning(f"Variable '{var}' not found. Skipping.")
            continue
        out_path = Path(plot_dir) / f"{var}_{date_str}.png"
        plot_variable_2d(
            ds,
            var,
            lat_name,
            lon_name,
            out_path,
            stride=stride,
            use_scatter=use_scatter,
            add_features=add_features,
            title_suffix=title_suffix,
        )


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate geographic maps for SWE/Snow output variables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "nc_path",
        nargs="?",
        default="./_data/swe_combined_20190101.nc",
        help="Path to the input NetCDF file.",
    )
    parser.add_argument(
        "-o",
        "--plot-dir",
        default="./plot",
        help="Directory to write the output PNG plots.",
    )
    parser.add_argument(
        "-s",
        "--stride",
        type=int,
        default=5,
        help="Decimation stride (use 1 for full resolution).",
    )
    parser.add_argument(
        "--pcolormesh",
        action="store_true",
        help="Use pcolormesh instead of the default scatter renderer.",
    )
    parser.add_argument(
        "--add-features",
        action="store_true",
        help="Draw Natural Earth coastlines/borders/states (needs network).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        nc_path=args.nc_path,
        plot_dir=args.plot_dir,
        stride=args.stride,
        use_scatter=not args.pcolormesh,
        add_features=args.add_features,
    )