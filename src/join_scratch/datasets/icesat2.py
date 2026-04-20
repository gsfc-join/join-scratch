"""ICESat-2 file handler for join_scratch datasets (reads cached parquet)."""

import logging

import numpy as np
import xarray as xr
from pyresample.geometry import SwathDefinition

from join_scratch.datasets.base import JoinFileHandler

log = logging.getLogger(__name__)


class Icesat2FileHandler(JoinFileHandler):
    """Handler for ICESat-2 cached parquet files (GeoDataFrame with h_li column).

    No SlideRule imports are used; this handler reads a pre-cached parquet file.
    """

    def get_area_def(self, dataset_id=None):
        """Not applicable for ICESat-2 sparse point data; returns None."""
        return None

    def get_dataset(self, dataset_id="h_li", ds_info=None) -> xr.DataArray:
        """Load the parquet file and return a 1-D DataArray with SwathDefinition.

        Parameters
        ----------
        dataset_id:
            Column name to extract from the GeoDataFrame (default "h_li").

        Returns
        -------
        xr.DataArray with dim ["y"] and attrs["area"] set to a SwathDefinition.
        """
        import geopandas

        log.info("Loading ICESat-2 parquet %s", self.filename)
        gdf = geopandas.read_parquet(str(self.filename))

        lons = gdf.geometry.x.values.astype(np.float64)
        lats = gdf.geometry.y.values.astype(np.float64)
        values = gdf[dataset_id].values.astype(np.float32)

        lons_da = xr.DataArray(lons, dims=["y"])
        lats_da = xr.DataArray(lats, dims=["y"])
        area = SwathDefinition(lons=lons_da, lats=lats_da)

        return xr.DataArray(
            values,
            dims=["y"],
            attrs={"dataset_id": dataset_id, "sensor": "icesat2", "area": area},
        )
