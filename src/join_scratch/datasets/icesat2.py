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
        import geopandas
        import pandas as pd # <-- Add this

        log.info("Loading ICESat-2 parquet %s", self.filename)
        gdf = geopandas.read_parquet(str(self.filename))

        # --- NEW: Clean and Sort the Time Index ---
        # 1. Ensure it is explicitly a DatetimeIndex
        if not isinstance(gdf.index, pd.DatetimeIndex):
            gdf.index = pd.to_datetime(gdf.index)
            
        # 2. Strip timezone (make it naive) so xarray string slicing works
        if gdf.index.tz is not None:
            gdf.index = gdf.index.tz_localize(None)

        # 3. CRITICAL: Sort by time! Xarray slice() crashes if time jumps around
        gdf = gdf.sort_index()
        # ------------------------------------------

        lons = gdf.geometry.x.values.astype(np.float64)
        lats = gdf.geometry.y.values.astype(np.float64)
        values = gdf[dataset_id].values.astype(np.float32)

        lons_da = xr.DataArray(lons, dims=["y"])
        lats_da = xr.DataArray(lats, dims=["y"])
        area = SwathDefinition(lons=lons_da, lats=lats_da)
        
        coords = {
            "time": ("time", gdf.index.values),
            "lon": ("time", lons),
            "lat": ("time", lats)
        }

        da = xr.DataArray(
            values,
            dims=["time"],
            coords=coords, 
            attrs={"dataset_id": dataset_id, "sensor": "icesat2"} # Removed area
        )
        return da