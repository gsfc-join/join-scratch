#!/usr/bin/env python3
"""
Generate geographic maps for SWE/Snow output variables from swe_combined.nc.
Handles 0-360 longitude conversion to -180-180 automatically.
Iterates over the 'time' dimension if present, creating a plot for each time step.
"""

import logging
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def generate_maps(nc_path: str = "./_data/swe_combined.nc", plot_dir: str = "./plot"):
    """Reads a NetCDF file and plots requested variables on Lat/Lon maps."""
    nc_file = Path(nc_path)
    if not nc_file.exists():
        log.error(f"Input file not found: {nc_file}")
        return

    # Create the output directory if it doesn't exist
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    
    log.info(f"Opening dataset: {nc_file}")
    ds = xr.open_dataset(nc_file)

    # ---------------------------------------------------------
    # FIX LONGITUDE: Convert 0-360 to -180-180
    # ---------------------------------------------------------
    lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    if lon_name is not None:
        if ds[lon_name].max() > 180:
            log.info(f"Converting {lon_name} from [0, 360] to [-180, 180]...")
            ds.coords[lon_name] = (ds.coords[lon_name] + 180) % 360 - 180
            
            if ds[lon_name].ndim == 1:
                ds = ds.sortby(lon_name)
    # ---------------------------------------------------------

    variables_to_plot = [
        "amsr2_snow_depth_mean",
        "ceda_swe",
        "viirs_cgf_ndsi_snow_cover",
        "icesat2_snow_depth" 
    ]
    
    # Identify the coordinate names for plotting
    x_coord = "lon" if "lon" in ds.coords else "longitude"
    y_coord = "lat" if "lat" in ds.coords else "latitude"

    # Define bounding box for the map extent once
    min_lon, max_lon = float(ds[x_coord].min()), float(ds[x_coord].max())
    min_lat, max_lat = float(ds[y_coord].min()), float(ds[y_coord].max())

    # Loop through each variable
    for var in variables_to_plot:
        if var not in ds.variables:
            log.warning(f"Variable '{var}' not found in the dataset. Skipping.")
            continue
            
        log.info(f"Processing variable: {var}")
        
        data_var = ds[var]
        has_time = "time" in data_var.dims
        times_to_plot = ds.time.values if has_time else [None]

        # Calculate a global vmin/vmax across all time steps for a consistent colorbar
        try:
            v_min = float(data_var.quantile(0.01))
            v_max = float(data_var.quantile(0.99))
        except Exception:
            # Fallback if quantile fails (e.g. all NaNs)
            v_min, v_max = 0, 1

        cmap = "Blues" if "swe" in var or "snow" in var else "viridis"
        unit_label = data_var.attrs.get('units', data_var.attrs.get('unit', ''))
        long_name = data_var.attrs.get("long_name", var)

        # Loop through each time step
        for t in times_to_plot:
            if has_time:
                # Convert numpy datetime64 to a readable string (YYYY-MM-DD)
                date_str = str(t).split('T')[0]
                data_slice = data_var.sel(time=t)
            else:
                date_str = "static"
                data_slice = data_var

            # Skip if the entire array is NaN for this day
            if data_slice.isnull().all().item():
                log.info(f"  -> No data exists for {var} on {date_str}. Skipping.")
                continue

            log.info(f"  -> Generating map for {var} on {date_str}...")

            fig = plt.figure(figsize=(12, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            # Map styling
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":", zorder=3)
            ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=":", zorder=3)
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=4)
            gl.top_labels = False
            gl.right_labels = False

            try:
                # Extract raw numpy arrays
                lon_vals = ds[x_coord].values
                lat_vals = ds[y_coord].values
                data_vals = data_slice.values

                # Plot using pure Matplotlib pcolormesh
                mesh = ax.pcolormesh(
                    lon_vals,
                    lat_vals,
                    data_vals,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    zorder=2,
                    vmin=v_min,
                    vmax=v_max
                )
                
                # Add colorbar
                cbar = plt.colorbar(mesh, ax=ax, shrink=0.7, pad=0.05)
                cbar.set_label(unit_label)
                
            except Exception as e:
                log.error(f"  -> Failed to plot {var} on {date_str}: {e}")
                plt.close(fig)
                continue
            
            # Zoom exactly to domain bounds
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
            
            title_str = f"LIS Domain: {long_name}\n{date_str}" if has_time else f"LIS Domain: {long_name}"
            ax.set_title(title_str, fontsize=14, pad=15)
            
            filename = f"{var}_{date_str}.png" if has_time else f"{var}.png"
            output_path = Path(plot_dir) / filename
            
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
    ds.close()
    log.info("Finished generating all plots.")

if __name__ == "__main__":
    generate_maps()
    