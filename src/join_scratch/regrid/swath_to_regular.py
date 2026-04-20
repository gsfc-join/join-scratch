"""SwathDefinition (2-D) → AreaDefinition regridding."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from pyresample.ewa.dask_ewa import DaskEWAResampler
from pyresample.geometry import AreaDefinition, SwathDefinition
from satpy.resample.bucket import BucketAvg

from join_scratch.regrid._resample_helpers import _bilinear_resample, _kdtree_nearest

log = logging.getLogger(__name__)


def regrid_nearest(
    data: xr.DataArray,
    source_area: SwathDefinition,
    target_area: AreaDefinition,
    radius_of_influence: float = 15_000.0,
    cache_dir: Path | None = None,
) -> xr.DataArray:
    """Regrid using satpy KDTreeResampler (nearest-neighbour).

    Returns a 2-D float32 DataArray with dims=["y", "x"].
    """
    return _kdtree_nearest(
        data, source_area, target_area, radius_of_influence, cache_dir
    )


def regrid_bilinear(
    data: xr.DataArray,
    source_area: SwathDefinition,
    target_area: AreaDefinition,
    radius_of_influence: float = 15_000.0,
    cache_dir: Path | None = None,
) -> xr.DataArray:
    """Regrid using satpy BilinearResampler.

    Returns a 2-D float32 DataArray with dims=["y", "x"].
    """
    return _bilinear_resample(
        data, source_area, target_area, radius_of_influence, cache_dir
    )


def regrid_ewa(
    data: xr.DataArray,
    source_area: SwathDefinition,
    target_area: AreaDefinition,
) -> xr.DataArray:
    """Regrid using DaskEWAResampler (Elliptical Weighted Averaging).

    rows_per_scan=0 disables scan-line grouping.
    Returns a 2-D float32 DataArray with dims=["y", "x"].
    """
    resampler = DaskEWAResampler(source_area, target_area)
    result = np.asarray(
        resampler.resample(data, rows_per_scan=0, fill_value=np.nan),
        dtype=np.float32,
    )
    return xr.DataArray(result, dims=["y", "x"])


def regrid_bucket_avg(
    data: xr.DataArray,
    source_area: SwathDefinition,
    target_area: AreaDefinition,
) -> xr.DataArray:
    """Regrid using satpy BucketAvg (average of all source pixels per target cell).

    Returns a 2-D float32 DataArray with dims=["y", "x"].
    """
    resampler = BucketAvg(source_area, target_area)
    result = np.asarray(resampler.resample(data, fill_value=np.nan), dtype=np.float32)
    return xr.DataArray(result, dims=["y", "x"])
