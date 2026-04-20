"""CEDA ESA CCI SWE file handler for join_scratch datasets."""

import logging

import numpy as np
import xarray as xr
from pyresample.geometry import AreaDefinition

from join_scratch.datasets.base import JoinFileHandler

log = logging.getLogger(__name__)

CEDA_VARS = ["swe", "swe_std"]


class CedaFileHandler(JoinFileHandler):
    """Handler for CEDA ESA CCI Snow SWE NetCDF files.

    The CEDA L3C daily product uses a global 0.1° regular lat/lon grid
    (1800 × 3600) — structurally identical to AMSR2.
    """

    CEDA_VARS = CEDA_VARS
    RADIUS_OF_INFLUENCE = 15_000.0

    def get_area_def(self, dataset_id=None) -> AreaDefinition:
        """Return the CEDA 0.1° equirectangular global AreaDefinition."""
        return AreaDefinition(
            "ceda_equirect",
            "CEDA 0.1° equirectangular global",
            "ceda_equirect",
            {"proj": "longlat", "datum": "WGS84"},
            3600,
            1800,
            (-180.0, -90.0, 180.0, 90.0),
        )

    def _open_ds(self):
        """Open the NetCDF file as an xarray Dataset."""
        if hasattr(self.filename, "open"):
            with self.filename.open() as f:
                ds = xr.open_dataset(f)
                ds.load()
            return ds
        return xr.open_dataset(self.filename)

    def get_dataset(self, dataset_id=None, ds_info=None):
        """Load the CEDA NetCDF file.

        Parameters
        ----------
        dataset_id:
            If None or "all", return an xr.Dataset with both swe and swe_std.
            If "swe" or "swe_std", return that variable as an xr.DataArray.

        Returns
        -------
        xr.Dataset or xr.DataArray
        """
        log.info("Loading CEDA file %s", self.filename)
        ds = self._open_ds()

        # Squeeze size-1 time dimension if present
        if "time" in ds.dims:
            ds = ds.squeeze("time", drop=True)

        # Keep only required variables
        ds = ds[CEDA_VARS + ["lat", "lon"]]

        # Mask flag values (< 0) to NaN
        for var in CEDA_VARS:
            ds[var] = ds[var].where(ds[var] >= 0).astype(np.float32)

        # Rename lat/lon dims to y/x
        ds = ds.rename_dims({"lat": "y", "lon": "x"})
        ds.load()

        if dataset_id is None or dataset_id == "all":
            return ds
        return ds[dataset_id]
