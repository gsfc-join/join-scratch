"""Regrid subpackage — dispatch by grid-type pair.

Method/backend matrix:
  AreaDef → AreaDef:
    nearest:      backend=pyresample (default)
    bilinear:     backend=pyresample (default) or xesmf
    conservative: backend=xesmf only
  SwathDef (2-D) → AreaDef:
    nearest:      backend=pyresample (default)
    bilinear:     backend=pyresample (default)
    ewa:          backend=pyresample (default)
    bucket_avg:   backend=pyresample (default)
  SwathDef (1-D) → AreaDef:
    mean:         backend=pixel_binning (default)
"""

import xarray as xr
from pyresample.geometry import AreaDefinition, SwathDefinition

from join_scratch.regrid import regular_to_regular, sparse_to_regular, swath_to_regular


def regrid(
    data: xr.DataArray,
    source_area,
    target_area: AreaDefinition,
    method: str,
    backend: str | None = None,
    **kwargs,
) -> xr.DataArray:
    """Regrid data from source_area to target_area.

    Dispatch is by (type(source_area), type(target_area)) pair.
    Within each pair, ``method`` selects the algorithm and ``backend``
    optionally selects among multiple implementations.

    Parameters
    ----------
    data : xr.DataArray
    source_area : AreaDefinition or SwathDefinition
    target_area : AreaDefinition
    method : str
    backend : str, optional
    **kwargs : passed to the underlying resampler

    Returns
    -------
    xr.DataArray with dims=["y", "x"]
    """
    if isinstance(source_area, AreaDefinition) and isinstance(
        target_area, AreaDefinition
    ):
        if method == "nearest":
            return regular_to_regular.regrid_nearest(
                data, source_area, target_area, **kwargs
            )
        elif method == "bilinear":
            return regular_to_regular.regrid_bilinear(
                data,
                source_area,
                target_area,
                backend=backend or "pyresample",
                **kwargs,
            )
        elif method == "conservative":
            return regular_to_regular.regrid_conservative(
                data, source_area, target_area, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown method '{method}' for AreaDef→AreaDef. "
                "Valid: nearest, bilinear, conservative"
            )

    elif isinstance(source_area, SwathDefinition) and isinstance(
        target_area, AreaDefinition
    ):
        if source_area.lons.ndim == 2:
            # 2-D swath
            if method == "nearest":
                return swath_to_regular.regrid_nearest(
                    data, source_area, target_area, **kwargs
                )
            elif method == "bilinear":
                return swath_to_regular.regrid_bilinear(
                    data, source_area, target_area, **kwargs
                )
            elif method == "ewa":
                return swath_to_regular.regrid_ewa(
                    data, source_area, target_area, **kwargs
                )
            elif method == "bucket_avg":
                return swath_to_regular.regrid_bucket_avg(
                    data, source_area, target_area, **kwargs
                )
            else:
                raise ValueError(
                    f"Unknown method '{method}' for SwathDef(2-D)→AreaDef. "
                    "Valid: nearest, bilinear, ewa, bucket_avg"
                )
        else:
            # 1-D sparse / point cloud
            if method == "mean":
                return sparse_to_regular.regrid_mean(
                    data, source_area, target_area, **kwargs
                )
            else:
                raise ValueError(
                    f"Unknown method '{method}' for SwathDef(1-D)→AreaDef. Valid: mean"
                )

    else:
        raise NotImplementedError(
            f"Unsupported grid-type pair: {type(source_area).__name__} → "
            f"{type(target_area).__name__}"
        )
