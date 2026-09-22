"""VIIRS CGF Snow Cover file handler for join_scratch datasets."""

import logging
import re

import h5py
import numpy as np
import pyproj
import xarray as xr
from pyproj.crs import GeographicCRS, ProjectedCRS
from pyproj.crs.coordinate_operation import SinusoidalConversion
from pyproj.crs.datum import CustomDatum, CustomEllipsoid
from pyresample.geometry import SwathDefinition

from join_scratch.datasets.base import JoinFileHandler

log = logging.getLogger(__name__)

# MODIS Sinusoidal sphere radius (metres)
_SIN_R = 6371007.181

# MODIS Sinusoidal projection CRS (built programmatically to avoid PROJ-string round-trips)
_SIN_CRS = ProjectedCRS(
    name="MODIS Sinusoidal",
    conversion=SinusoidalConversion(),
    geodetic_crs=GeographicCRS(
        datum=CustomDatum(
            ellipsoid=CustomEllipsoid(
                name="MODIS Sphere",
                semi_major_axis=_SIN_R,
                inverse_flattening=0,
            )
        )
    ),
)

# HDF5 paths within the HDFEOS tile file
_HDFEOS_DATA_PATH = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields/CGF_NDSI_Snow_Cover"
_HDFEOS_XDIM_PATH = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/XDim"
_HDFEOS_YDIM_PATH = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/YDim"

# Regex for extracting h/v tile indices from filenames
_HV_RE = re.compile(r"\.h(\d{2})v(\d{2})\.")


class ViirsFileHandler(JoinFileHandler):
    """Handler for one VIIRS VJ110A1F CGF Snow Cover HDF5 tile.

    The tile uses the MODIS Sinusoidal (SIN) projection; pixel centres are
    derived from XDim/YDim projection coordinates via pyproj.
    """

    RADIUS_OF_INFLUENCE = 600.0

    def __init__(self, filename, filename_info=None, filetype_info=None):
        super().__init__(filename, filename_info, filetype_info)
        # Extract h/v tile indices from the filename string
        fname_str = str(filename)
        m = _HV_RE.search(fname_str)
        if m is not None:
            self.h = int(m.group(1))
            self.v = int(m.group(2))
        else:
            self.h = None
            self.v = None

    def get_area_def(self, dataset_id=None):
        """VIIRS tiles use SwathDefinition; AreaDefinition is not applicable."""
        return None

    def get_dataset(
        self, dataset_id="CGF_NDSI_Snow_Cover", ds_info=None
    ) -> xr.DataArray:
        """Load the VIIRS HDF5 tile and return a 2-D DataArray with SwathDefinition.

        Flag values (> 100) are masked to NaN.  The attrs["area"] is set to a
        SwathDefinition built from per-pixel lon/lat computed via pyproj.

        Parameters
        ----------
        dataset_id:
            Must be ``"CGF_NDSI_Snow_Cover"`` — the only data variable in
            VJ110A1F HDF5 files.  Any other value raises ``ValueError``.

        Returns
        -------
        xr.DataArray with dims=["y", "x"] and attrs["area"] set.
        """
        _VIIRS_VAR = "CGF_NDSI_Snow_Cover"
        if dataset_id != _VIIRS_VAR:
            raise ValueError(
                f"VIIRS files contain only '{_VIIRS_VAR}'; got dataset_id={dataset_id!r}"
            )
        log.info("Loading VIIRS tile %s", self.filename)

        fname = self.filename
        if hasattr(fname, "open"):
            with fname.open() as fobj:
                with h5py.File(fobj, "r") as f:
                    raw = f[_HDFEOS_DATA_PATH][:]
                    x_coords = f[_HDFEOS_XDIM_PATH][:]
                    y_coords = f[_HDFEOS_YDIM_PATH][:]
        else:
            with h5py.File(fname, "r") as f:
                raw = f[_HDFEOS_DATA_PATH][:]
                x_coords = f[_HDFEOS_XDIM_PATH][:]
                y_coords = f[_HDFEOS_YDIM_PATH][:]

        data = raw.astype(np.float32)
        data[data > 100] = np.nan

        x2d, y2d = np.meshgrid(x_coords, y_coords)
        transformer = pyproj.Transformer.from_crs(_SIN_CRS, "EPSG:4326", always_xy=True)
        lon2d, lat2d = transformer.transform(x2d, y2d)

        lons_da = xr.DataArray(lon2d, dims=["y", "x"])
        lats_da = xr.DataArray(lat2d, dims=["y", "x"])
        area = SwathDefinition(lons=lons_da, lats=lats_da)

        return xr.DataArray(
            data,
            dims=["y", "x"],
            attrs={"dataset_id": dataset_id, "sensor": "viirs", "area": area},
        )
