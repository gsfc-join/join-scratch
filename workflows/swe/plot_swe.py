#!/usr/bin/env python3
"""
Generate geographic maps for SWE/Snow output variables from swe_combined.nc.
Handles 0-360 longitude conversion to -180-180 automatically.
"""

import logging
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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

    print(ds)

    # ---------------------------------------------------------
    # FIX LONGITUDE: Convert 0-360 to -180-180
    # ---------------------------------------------------------
    lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    if lon_name is not None:
        if ds[lon_name].max() > 180:
            log.info(f"Converting {lon_name} from [0, 360] to [-180, 180]...")
            # Shift the coordinates
            ds.coords[lon_name] = (ds.coords[lon_name] + 180) % 360 - 180
            
            # If longitude is a 1D coordinate array, we must sort the dataset 
            # by longitude so pcolormesh doesn't tear across the map.
            if ds[lon_name].ndim == 1:
                ds = ds.sortby(lon_name)
    # ---------------------------------------------------------

    variables_to_plot = [
        "amsr2_snow_depth_mean",
        "ceda_swe",
        "viirs_cgf_ndsi_snow_cover",
        "icesat2_snow_depth" 
    ]
    
    # Identify the coordinate names for xarray plotting
    coord_kwargs = {}
    if "lon" in ds.coords and "lat" in ds.coords:
        coord_kwargs = {"x": "lon", "y": "lat"}
    elif "longitude" in ds.coords and "latitude" in ds.coords:
        coord_kwargs = {"x": "longitude", "y": "latitude"}

    for var in variables_to_plot:
        if var in ds.variables:
            log.info(f"Generating map for {var}...")
            
            fig = plt.figure(figsize=(12, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
            ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=":")
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            
            cmap = "Blues" if "swe" in var or "snow" in var else "viridis"
            unit_label = ds[var].attrs.get('units', ds[var].attrs.get('unit', ''))
            
            # Plot the variable onto the Cartopy projection
            ds[var].plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                cbar_kwargs={'shrink': 0.7, 'pad': 0.05, 'label': unit_label},
                **coord_kwargs
            )           
            
            long_name = ds[var].attrs.get("long_name", var)
            ax.set_title(f"LIS Domain: {long_name}", fontsize=14, pad=15)
            
            output_path = Path(plot_dir) / f"{var}.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            log.info(f"Saved plot to {output_path}")
        else:
            log.warning(f"Variable '{var}' not found in the dataset. Skipping.")

    ds.close()
    log.info("Finished generating all plots.")

if __name__ == "__main__":
    generate_maps(nc_path="./_data/swe_combined.nc", plot_dir="./plot")