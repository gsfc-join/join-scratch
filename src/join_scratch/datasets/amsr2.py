"""AMSR2 file handler for join_scratch datasets."""

import logging

import numpy as np
import xarray as xr
from pyresample.geometry import AreaDefinition

from join_scratch.datasets.base import JoinFileHandler

log = logging.getLogger(__name__)


class Amsr2FileHandler(JoinFileHandler):
    """Handler for AMSR2 L3 monthly HDF5 files.

    The AMSR2 L3 product uses a fixed global 0.1° equirectangular grid
    (1800 × 3600) with no explicit coordinate variables in the HDF5 file.
    """

    AMSR2_N_LAT = 1800
    AMSR2_N_LON = 3600
    RADIUS_OF_INFLUENCE = 15_000.0

    _lat = np.linspace(89.95, -89.95, 1800)
    _lon = np.linspace(-179.95, 179.95, 3600)

    def get_area_def(self, dataset_id=None) -> AreaDefinition:
        """Return the AMSR2 0.1° equirectangular global AreaDefinition."""
        _ = dataset_id
        return AreaDefinition(
            "amsr2_equirect",
            "AMSR2 0.1° equirectangular global",
            "amsr2_equirect",
            {"proj": "longlat", "datum": "WGS84"},
            3600,
            1800,
            (-180.0, -90.0, 180.0, 90.0),
        )

    def _open_ds(self):
        """Open the HDF5 file as an xarray Dataset."""
        if hasattr(self.filename, "open"):
            with self.filename.open() as f:
                ds = xr.open_dataset(f, engine="h5netcdf", phony_dims="sort")
                ds.load()
            return ds
        return xr.open_dataset(self.filename, engine="h5netcdf", phony_dims="sort")

    def get_dataset(self, dataset_id="Geophysical Data", ds_info=None) -> xr.DataArray:
        """Load the AMSR2 HDF5 file and return a 3-D DataArray (y, x, inner).

        Fill values (< 0) are masked to NaN and values are cast to float32.

        Parameters
        ----------
        dataset_id:
            Must be ``"Geophysical Data"`` — the only data variable in AMSR2
            L3 HDF5 files.  Any other value raises ``ValueError``.
        """
        _ = ds_info  # Silence the type checker
        _AMSR2_VAR = "Geophysical Data"
        if dataset_id != _AMSR2_VAR:
            raise ValueError(
                f"AMSR2 files contain only '{_AMSR2_VAR}'; got dataset_id={dataset_id!r}"
            )
        log.info("Loading AMSR2 file %s", self.filename)
        ds = self._open_ds()
        ds = ds.rename_dims({"phony_dim_0": "y", "phony_dim_1": "x"})
        ds = ds.assign_coords(y=self._lat, x=self._lon)

        data = ds["Geophysical Data"]
        data = data.where(data >= 0).astype(np.float32)

        return xr.DataArray(
            data.values,
            dims=["y", "x", "inner"],
            coords={"y": self._lat, "x": self._lon},
            attrs={"dataset_id": dataset_id, "sensor": "amsr2"},
        )
