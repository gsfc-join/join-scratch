"""Private shared resampler helpers used by swath_to_regular and regular_to_regular."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from satpy.resample.kdtree import BilinearResampler, KDTreeResampler

log = logging.getLogger(__name__)


def _kdtree_nearest(
    data: xr.DataArray,
    source_area,
    target_area,
    radius_of_influence: float = 15_000.0,
    cache_dir: Path | None = None,
) -> xr.DataArray:
    """KDTree nearest-neighbour resample. Returns float32 DataArray dims=[y, x]."""
    resampler = KDTreeResampler(source_area, target_area)
    precompute_kwargs: dict = {"radius_of_influence": radius_of_influence}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        precompute_kwargs["cache_dir"] = str(cache_dir)
    resampler.precompute(**precompute_kwargs)
    result = np.asarray(resampler.compute(data, fill_value=np.nan), dtype=np.float32)
    return xr.DataArray(result, dims=["y", "x"])


def _bilinear_resample(
    data: xr.DataArray,
    source_area,
    target_area,
    radius_of_influence: float = 15_000.0,
    cache_dir: Path | None = None,
) -> xr.DataArray:
    """Bilinear resample. Returns float32 DataArray dims=[y, x]."""
    resampler = BilinearResampler(source_area, target_area)
    resample_kwargs: dict = {
        "radius_of_influence": radius_of_influence,
        "fill_value": np.nan,
    }
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        resample_kwargs["cache_dir"] = str(cache_dir)
    result = np.asarray(resampler.resample(data, **resample_kwargs), dtype=np.float32)
    return xr.DataArray(result, dims=["y", "x"])
