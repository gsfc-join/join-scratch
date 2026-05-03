"""AreaDefinition → AreaDefinition regridding (regular grid to regular grid)."""

import logging
from pathlib import Path

import xarray as xr
from pyresample.geometry import AreaDefinition

from join_scratch.regrid._resample_helpers import _bilinear_resample, _kdtree_nearest

log = logging.getLogger(__name__)


def _xesmf():
    """Lazy import of xesmf to avoid breaking the module when xesmf is absent."""
    try:
        import xesmf

        return xesmf
    except ImportError as exc:
        raise ImportError(
            "xesmf is required for compute_weights/load_regridder. "
            "Install it with: pixi add xesmf"
        ) from exc


def compute_weights(
    source_grid: xr.Dataset,
    target_grid: xr.Dataset,
    weights_path: Path,
    method: str = "bilinear",
    overwrite: bool = False,
) -> Path:
    """Compute xESMF regridding weights and save them to *weights_path*.

    If *weights_path* already exists and *overwrite* is False, skip recomputation
    and return the existing path.  Set *overwrite=True* to always recompute.
    Returns *weights_path*.
    """
    if weights_path.exists() and not overwrite:
        log.info("Reusing existing xESMF weights from %s", weights_path)
        return weights_path
    log.info("Computing xESMF %s weights …", method)
    regridder = _xesmf().Regridder(
        source_grid,
        target_grid,
        method=method,
        periodic=True,
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    regridder.to_netcdf(str(weights_path))
    log.info("xESMF weights saved to %s", weights_path)
    return weights_path


def load_regridder(
    source_grid: xr.Dataset,
    target_grid: xr.Dataset,
    weights_path: Path,
    method: str = "bilinear",
):
    """Load a pre-computed xESMF regridder from *weights_path*.

    Raises FileNotFoundError if the weights file does not exist — call
    compute_weights() first.
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}. Run compute_weights() first."
        )
    log.info("Loading xESMF weights from %s", weights_path)
    return _xesmf().Regridder(
        source_grid,
        target_grid,
        method=method,
        periodic=True,
        weights=str(weights_path),
    )


def regrid_nearest(
    data: xr.DataArray,
    source_area: AreaDefinition,
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
    source_area: AreaDefinition,
    target_area: AreaDefinition,
    radius_of_influence: float = 15_000.0,
    cache_dir: Path | None = None,
    backend: str = "pyresample",
) -> xr.DataArray:
    """Regrid using bilinear interpolation.

    backend="pyresample": uses satpy BilinearResampler.
    backend="xesmf": raises NotImplementedError (use load_regridder() directly).

    Returns a 2-D float32 DataArray with dims=["y", "x"].
    """
    if backend == "xesmf":
        raise NotImplementedError(
            "xESMF bilinear via regrid() dispatch; use load_regridder() directly"
        )
    return _bilinear_resample(
        data, source_area, target_area, radius_of_influence, cache_dir
    )


def regrid_conservative(
    data: xr.DataArray,
    source_area: AreaDefinition,
    target_area: AreaDefinition,
    **kwargs,
) -> xr.DataArray:
    """Conservative regridding (xESMF only).

    Raises NotImplementedError — use compute_weights(method='conservative') +
    load_regridder() directly.
    """
    raise NotImplementedError(
        "conservative regridding via xESMF; use compute_weights(method='conservative') + load_regridder() directly"
    )
