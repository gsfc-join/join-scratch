"""SwathDefinition (1-D sparse) → AreaDefinition regridding via pixel binning."""

import logging

import numpy as np
import pyproj
import xarray as xr
from pyresample.geometry import AreaDefinition, SwathDefinition

log = logging.getLogger(__name__)


def regrid_mean(
    data: xr.DataArray,
    source_area: SwathDefinition,
    target_area: AreaDefinition,
) -> xr.DataArray:
    """Project 1-D point-cloud data onto target_area and compute per-pixel mean.

    Parameters
    ----------
    data:
        1-D DataArray with dim=["y"] containing values to grid.
    source_area:
        SwathDefinition with 1-D lons/lats (dim=["y"]).
    target_area:
        AreaDefinition describing the target grid.

    Returns
    -------
    2-D float32 DataArray with dims=["y", "x"].  Pixels with no observations
    are NaN.
    """
    nx = target_area.width
    ny = target_area.height
    x_min, y_min, x_max, y_max = target_area.area_extent
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    lons = np.asarray(source_area.lons).ravel()
    lats = np.asarray(source_area.lats).ravel()
    values = np.asarray(data).ravel().astype(np.float64)

    if len(values) == 0:
        log.warning("source_area has no points; returning all-NaN grid")
        return xr.DataArray(
            np.full((ny, nx), np.nan, dtype=np.float32), dims=["y", "x"]
        )

    # Project lon/lat to target CRS
    target_crs = target_area.crs
    transformer = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    xs, ys = transformer.transform(lons, lats)

    # Map projected coordinates to pixel col/row
    col = np.floor((xs - x_min) / dx).astype(int)
    row = np.floor((y_max - ys) / dy).astype(int)  # rows from top

    # Keep only points that fall inside the grid and have finite values
    valid = (col >= 0) & (col < nx) & (row >= 0) & (row < ny) & np.isfinite(values)
    col = col[valid]
    row = row[valid]
    values = values[valid]
    log.info("%d of %d points fall within the target grid", valid.sum(), len(valid))

    # Accumulate sum and count per pixel
    flat_idx = row * nx + col
    val_sum = np.zeros(ny * nx, dtype=np.float64)
    count = np.zeros(ny * nx, dtype=np.int32)
    np.add.at(val_sum, flat_idx, values)
    np.add.at(count, flat_idx, 1)

    with np.errstate(invalid="ignore"):
        result = np.where(count > 0, val_sum / count, np.nan).reshape(ny, nx)

    n_filled = int((count > 0).sum())
    log.info(
        "Gridded %d points into %d/%d pixels (%.2f%% fill)",
        valid.sum(),
        n_filled,
        ny * nx,
        100.0 * n_filled / (ny * nx),
    )
    return xr.DataArray(result.astype(np.float32), dims=["y", "x"])
