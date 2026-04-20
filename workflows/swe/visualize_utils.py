"""Shared visualization utilities for SWE workflow scripts."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from pyresample.geometry import AreaDefinition


def _lis_boundary_lonlat(lis_area: AreaDefinition) -> tuple[np.ndarray, np.ndarray]:
    """Return (lons, lats) arrays tracing the LIS domain boundary (closed loop)."""
    lons, lats = lis_area.get_lonlats()
    b_lons = np.concatenate(
        [lons[0, :], lons[:, -1], lons[-1, ::-1], lons[::-1, 0], [lons[0, 0]]]
    )
    b_lats = np.concatenate(
        [lats[0, :], lats[:, -1], lats[-1, ::-1], lats[::-1, 0], [lats[0, 0]]]
    )
    return b_lons, b_lats


def _lis_extent(lis_grid: xr.Dataset, pad: float = 0.5) -> list[float]:
    """Return [lon_min, lon_max, lat_min, lat_max] for the LIS domain with padding."""
    lons = lis_grid["lon"].values
    lats = lis_grid["lat"].values
    return [
        float(lons.min()) - pad,
        float(lons.max()) + pad,
        float(lats.min()) - pad,
        float(lats.max()) + pad,
    ]


def add_map_features(ax) -> None:
    """Add standard cartographic features (coastlines, borders, lakes, states) to *ax*."""
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.7,
        linestyle="-",
        edgecolor="black",
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "lakes", "50m", edgecolor="black", facecolor="none"
        ),
        linewidth=0.7,
        linestyle="-",
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.7,
        linestyle="--",
        edgecolor="black",
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural",
            "admin_1_states_provinces_lines",
            "50m",
            edgecolor="gray",
            facecolor="none",
        ),
        linewidth=0.5,
        linestyle=":",
    )


def plot_panel(ax, data, lons, lats, vmin, vmax, title, extent, pc=None):
    """Plot a single map panel with pcolormesh and cartographic features.

    Parameters
    ----------
    ax:
        Cartopy GeoAxes.
    data:
        2-D array to plot.
    lons, lats:
        1-D or 2-D lon/lat arrays matching *data*.
    vmin, vmax:
        Color scale limits.
    title:
        Panel title.
    extent:
        [lon_min, lon_max, lat_min, lat_max] for ax.set_extent.
    pc:
        PlateCarree CRS instance (created if not provided).
    """
    if pc is None:
        pc = ccrs.PlateCarree()
    ax.set_extent(extent, crs=pc)
    im = ax.pcolormesh(
        lons,
        lats,
        data,
        transform=pc,
        vmin=vmin,
        vmax=vmax,
        cmap="Blues",
        shading="auto",
    )
    add_map_features(ax)
    ax.set_title(title, fontsize=8)
    return im
