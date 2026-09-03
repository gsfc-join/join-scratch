import sys
import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from join_scratch.datasets import Amsr2FileHandler, CedaFileHandler, Icesat2FileHandler, ViirsFileHandler
from join_scratch.regrid import regrid
from join_scratch.regrid.regular_to_regular import compute_weights, load_regridder
from lis_grid import build_lis_area_definition, load_lis_grid
from s3_utils import _is_s3, make_fs, make_store, list_s3, handler_from_s3

import sliderule
from sliderule import icesat2, raster

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

import pandas as pd
from pyresample.geometry import SwathDefinition 

import os
import boto3
import time

def _regrid_amsr2(
    input_dir: str,
    lis_grid: xr.Dataset,
    method: str,
    weights_dir: str,    
    current_date: pd.Timestamp,
    overwrite_weights: bool = False,
    fs=None,
) -> dict[str, xr.DataArray]:
    """Regrid the first matching AMSR2 file found and return {var_name: DataArray}."""
    
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from pathlib import Path
    import xarray as xr
    import numpy as np

    # Format the date to match how it appears in your filenames
    # Example format: "20190101"
    date_str = current_date.strftime("%Y%m%d") 

    all_files = _list_files(input_dir, ".h5", fs=fs)
    # Filter for files that belong to this specific date
    files = [f for f in all_files if "01D" in f and "EQMD" in f and date_str in f]
    
    if not files:
        log.warning("No daily descending AMSR2 files found — skipping")
        return {}
        
    path = files[0]
    log.info("AMSR2: using %s", _path_name(path))

    if _is_s3(path):
        handler = handler_from_s3(Amsr2FileHandler, path, fs=fs)
    else:
        handler = Amsr2FileHandler.from_path(path)

    # 1. Load dataset, rename dimensions
    da = handler.get_dataset().rename({"y": "lat", "x": "lon"})
    
    # 2. Fix the 180-degree phase shift in the source data matrix
    log.info("Applying a physical 180-degree roll to the AMSR2 data matrix...")
    half_width = len(handler._lon) // 2
    da = da.roll(lon=half_width, roll_coords=False)
    
    # Strictly attach the physical coordinates
    da = da.assign_coords(lat=handler._lat, lon=handler._lon)

    # Reverse Latitudes if descending
    if da.lat.values[0] > da.lat.values[-1]:
        log.info("Reversing latitude matrix to be strictly ascending...")
        da = da.isel(lat=slice(None, None, -1))

    # =========================================================================
    # Ensure LIS grid has lat/lon officially set as coordinates
    # =========================================================================
    if "lat" not in lis_grid.coords or "lon" not in lis_grid.coords:
        log.info("LIS grid is missing lat/lon in coords. Promoting from data variables...")
        lat_var = next((n for n in ["lat", "latitude", "XLAT"] if n in lis_grid.variables), None)
        lon_var = next((n for n in ["lon", "longitude", "XLONG"] if n in lis_grid.variables), None)
        
        if lat_var and lon_var:
            rename_dict = {lat_var: "lat", lon_var: "lon"} if lat_var != "lat" or lon_var != "lon" else {}
            if rename_dict: lis_grid = lis_grid.rename(rename_dict)
            lis_grid = lis_grid.set_coords(["lat", "lon"])
        else:
            raise ValueError(f"Could not find lat/lon in LIS grid!")

    # Calculate LIS extents for zoomed-in plotting
    min_lon, max_lon = lis_grid.lon.min().item(), lis_grid.lon.max().item()
    min_lat, max_lat = lis_grid.lat.min().item(), lis_grid.lat.max().item()
    lis_extents = [min_lon - 2, max_lon + 2, min_lat - 2, max_lat + 2]

    # --- DEBUG PLOTTING HELPER ---
    def _debug_plot(data_array, title, filename, extents=None, vmax=None):
        log.info(f"Generating debug plot: {filename}")
        Path("./plot").mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=':')
        
        try:
            coords = list(data_array.coords.keys()) + list(data_array.dims)
            x_var = 'lon' if 'lon' in coords else ('longitude' if 'longitude' in coords else ('east_west' if 'east_west' in coords else None))
            y_var = 'lat' if 'lat' in coords else ('latitude' if 'latitude' in coords else ('north_south' if 'north_south' in coords else None))
            
            plot_kwargs = {'ax': ax, 'transform': ccrs.PlateCarree(), 'cmap': "Blues", 'cbar_kwargs': {'shrink': 0.6}}
            if x_var and y_var:
                plot_kwargs.update({'x': x_var, 'y': y_var})
            if vmax is not None:
                plot_kwargs['vmax'] = vmax
                
            data_array.plot.pcolormesh(**plot_kwargs)
        except Exception as e:
            log.warning(f"Could not plot {filename}: {e}")
            
        ax.set_title(title)
        
        if extents:
            ax.set_extent(extents, crs=ccrs.PlateCarree())
        else:
            ax.set_global()  
            
        fig.savefig(f"./plot/{filename}", dpi=150, bbox_inches="tight")
        plt.close(fig)
    # -----------------------------

    # Extract native data array, but don't plot it just yet
    da_mean_native = da.isel(inner=0).drop_vars("inner", errors="ignore")

    # 4. Create source grid EXACTLY from our native coordinates
    source_grid = xr.Dataset(coords={"lat": da.lat, "lon": da.lon})

    # 5. Compute weights and initialize regridder (Static File)
    weights_local_dir = _ensure_local_dir(weights_dir)
    weights_path = weights_local_dir / f"amsr2-lis-weights-{method}.nc"
    
    # This will naturally skip computing if the file exists and overwrite_weights=False
    compute_weights(source_grid, lis_grid, weights_path, method=method, overwrite=overwrite_weights)
    regridder = load_regridder(source_grid, lis_grid, weights_path, method=method)

    # 6. Extract the data and apply the regridder
    ds_mean = da.isel(inner=0).drop_vars("inner", errors="ignore").to_dataset(name="Geophysical Data")
    ds_unc  = da.isel(inner=1).drop_vars("inner", errors="ignore").to_dataset(name="Geophysical Data")

    log.info("AMSR2: regridding mean (inner=0) …")
    rg_mean = regridder(ds_mean)["Geophysical Data"]
    
    # Re-attach LIS coordinates to the output array for accurate map plotting
    rg_mean = rg_mean.assign_coords(lat=lis_grid.lat, lon=lis_grid.lon)

    # Find the max value from the regridded data to match colorbars
    plot_vmax = float(np.nanmax(rg_mean.values))
    log.info(f"Setting plot colorbar maximum to {plot_vmax:.2f}")

    # PLOT 1 (Delayed): Native Data using the regridded max value
    _debug_plot(da_mean_native, "AMSR2 Native Grid (Central U.S.)", "amsr2_01_native.png", extents=lis_extents, vmax=plot_vmax)

    # PLOT 3: Final Regridded map
    _debug_plot(rg_mean, "AMSR2 Regridded (Central U.S.)", "amsr2_03_regridded.png", extents=lis_extents, vmax=plot_vmax)

    log.info("AMSR2: regridding uncertainty (inner=1) …")
    rg_unc  = regridder(ds_unc)["Geophysical Data"]

    def _da(arr, long_name, units):
        return xr.DataArray(
            arr.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={"long_name": long_name, "units": units, "source": _path_name(path)},
        )

    return {
        "amsr2_snow_depth_mean": _da(
            rg_mean,
            "AMSR2 snow depth (daily descending-pass mean)",
            "mm",
        ),
        "amsr2_snow_depth_uncertainty": _da(
            rg_unc,
            "AMSR2 snow depth uncertainty (daily descending-pass value)",
            "mm",
        ),
    }
    