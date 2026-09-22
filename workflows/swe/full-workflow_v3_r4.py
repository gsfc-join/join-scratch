#!/usr/bin/env python
"""Full SWE workflow: regrid all input datasets to the LIS grid and combine.

Regrids each of the following datasets onto the LIS 1 km Lambert Conformal grid
and writes all outputs into a single NetCDF file with informatively named variables:

  Dataset              Variable(s)           Default method    Output variable(s)
  ─────────────────────────────────────────────────────────────────────────────────
  AMSR2 snow depth     Geophysical Data[0]   bilinear          amsr2_snow_depth_mean
                       Geophysical Data[1]   bilinear          amsr2_snow_depth_uncertainty
  CEDA ESA CCI SWE     swe                   bilinear          ceda_swe
                       swe_std               bilinear          ceda_swe_std
  VIIRS CGF snow cover CGF_NDSI_Snow_Cover   nearest           viirs_cgf_ndsi_snow_cover
  ICESat-2 ATL06       h_li                  mean              icesat2_h_li

For AMSR2 ``inner`` dimension (phony_dim_2): index 0 = ascending-pass mean value,
index 1 = descending-pass uncertainty, per dataset documentation.

Paths may be local file-system paths or ``s3://`` URIs — detection is automatic.

Usage example (local)
─────────────────────
    python full-workflow.py \\
        --lis-path /data/lis_input_NMP_1000m_missouri.nc \\
        --amsr2-dir /data/amsr2 \\
        --ceda-dir  /data/ceda \\
        --viirs-dir /data/viirs \\
        --output-path /data/swe_combined.nc

Usage example (S3)
──────────────────
    python full-workflow.py \\
        --lis-path s3://my-bucket/lis_input_NMP_1000m_missouri.nc \\
        --amsr2-dir s3://my-bucket/amsr2 \\
        --ceda-dir  s3://my-bucket/ceda \\
        --viirs-dir s3://my-bucket/viirs \\
        --output-path /data/swe_combined.nc

"""

import sys
import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr
import h5py

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
from pyresample.kd_tree import resample_nearest

import pyproj

import os
import boto3
import time

import icechunk

import earthaccess
import numpy as np
import logging

log = logging.getLogger(__name__)

# Constants for VIIRS HDF-EOS5 Structure
_HDFEOS_DATA_PATH = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields/CGF_NDSI_Snow_Cover"
_HDFEOS_XDIM_PATH = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/XDim"
_HDFEOS_YDIM_PATH = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/YDim"

# Standard MODIS/VIIRS Sinusoidal projection definition
_SIN_CRS = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
    "+a=6371007.181 +b=6371007.181 +units=m +no_defs"
)

# Silence icechunk rust warnings. Must be set before importing icechunk.
os.environ.setdefault("RUST_LOG", "error")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

AMSR2_GLOB = "**/*.h5"
CEDA_GLOB = "**/*.nc"
VIIRS_GLOB = "**/*.h5"

# AMSR2 Icechunk constants
GPORTAL_URL = "https://gportal.jaxa.jp/"
CEDA_URL = "https://dap.ceda.ac.uk/"

import boto3
import os

def refresh_aws_credentials():
    """Fetches fresh credentials from the boto3 session and injects them into the environment."""
    log.info("Refreshing AWS credentials...")
    session = boto3.Session()
    creds = session.get_credentials()
    
    if creds:
        # get_frozen_credentials() forces a refresh if the current ones are expired
        frozen_creds = creds.get_frozen_credentials()
        os.environ["AWS_ACCESS_KEY_ID"] = frozen_creds.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = frozen_creds.secret_key
        if frozen_creds.token:
            os.environ["AWS_SESSION_TOKEN"] = frozen_creds.token
            
        os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
        log.info("Successfully refreshed and injected AWS credentials.")
    else:
        log.warning("Warning: boto3 could not find AWS credentials to refresh.")

# Call it once at initialization
refresh_aws_credentials()


# ── path helpers ──────────────────────────────────────────────────────────────

def _list_files(dir_path: str, suffix: str, fs=None) -> list[str]:
    """List files under *dir_path* matching *suffix*, local or S3."""
    if _is_s3(dir_path):
        # Parse bucket and prefix from s3://bucket/prefix
        without_scheme = dir_path[len("s3://"):]
        bucket, _, prefix = without_scheme.partition("/")
        store = make_store(bucket, prefix=prefix)
        keys = list_s3(store)
        prefix_slash = prefix.rstrip("/") + "/" if prefix else ""
        urls = [f"s3://{bucket}/{prefix_slash}{k}" for k in keys if k.endswith(suffix)]
        return sorted(urls)
    else:
        return [str(p) for p in sorted(Path(dir_path).glob(f"**/*{suffix}"))]


def _path_name(path_str: str) -> str:
    """Return a short display name for a path (last component)."""
    return path_str.rstrip("/").split("/")[-1]


def _ensure_local_dir(path: str) -> Path:
    """Ensure a local directory exists and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# ── helpers ───────────────────────────────────────────────────────────────────
def _regrid_amsr2(
    input_dir: str,
    lis_grid: xr.Dataset,
    method: str,
    weights_dir: str,    
    current_date: pd.Timestamp,
    overwrite_weights: bool = False,
    fs=None,
) -> dict[str, xr.DataArray]:
    """Regrid AMSR2 data from Icechunk store and return {var_name: DataArray}."""
     
    # The Icechunk store is accessed via Repository.open with S3 storage
    log.info(f"AMSR2: Opening Icechunk repository from S3...")
    
    try:
        # Open the Icechunk repository using the proper API
        repo = icechunk.Repository.open(
            icechunk.s3_storage(
                bucket="airborne-smce-prod-user-bucket",
                prefix="JOIN/icechunk-stores/GCOM-W1-AMSR2-L3-SND",
                from_env=True  # Force dynamic credential refresh
            ),
            authorize_virtual_chunk_access={GPORTAL_URL: icechunk.credentials.HttpAccess},
        )
        log.info("AMSR2: Successfully opened Icechunk repository")
        
        # Open a read-only session
        session = repo.readonly_session("main")
        
        # Open the dataset with xarray
        ds_full = xr.open_zarr(session.store)
        log.info(f"AMSR2 Icechunk store opened successfully. Available variables: {list(ds_full.data_vars)}")
        
    except Exception as e:
        log.error(f"Failed to open AMSR2 Icechunk store: {e}")
        return {}
    
    # Select data for the current date using boolean indexing (time may not be sorted)
    date_str = current_date.strftime("%Y-%m-%d")
    try:
        times = pd.DatetimeIndex(pd.to_datetime(ds_full["time"].values))
        mask = (times >= pd.Timestamp(date_str)) & (times <= pd.Timestamp(date_str))
        matching_indices = np.flatnonzero(mask)
        
        if len(matching_indices) == 0:
            log.warning(f"AMSR2: No data found for {date_str}")
            return {}
            
        ds_day = ds_full.isel(time=matching_indices[0])
        log.info(f"AMSR2: Selected data for {date_str}")
    except Exception as e:
        log.warning(f"AMSR2: Error selecting data for {date_str}: {e}")
        return {}
    
    # Extract the geophysical_data variable
    # Structure: (time, orbit, lat, lon, band)
    # We want orbit="Descending" and band="snow_depth"
    try:
        # Select descending orbit and snow_depth band
        da = ds_day["geophysical_data"].sel(orbit="Descending", band="snow_depth")
        
        # Apply scale factor (data is stored as int16, needs to be scaled to cm)
        scale_factor = ds_day["geophysical_data"].attrs.get("scale_factor", 0.1)
        da = da.astype(np.float32) * scale_factor
        
        # Convert from cm to mm for consistency with other datasets
        da = da * 10.0
        
        log.info(f"AMSR2: Extracted snow depth data with shape {da.shape}")
    except Exception as e:
        log.error(f"AMSR2: Error extracting geophysical_data: {e}")
        return {}
    
    # Ensure lat/lon coordinates are attached
    if 'lat' in ds_day.coords and 'lon' in ds_day.coords:
        da = da.assign_coords(lat=ds_day['lat'], lon=ds_day['lon'])
    
    # Convert from 0-360 to -180-180 if necessary
    if 'lon' in da.coords and da.lon.max() > 180:
        log.info("Converting AMSR2 longitudes from 0-360 to -180 to 180 and sorting matrix...")
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        da = da.sortby("lon")
    
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
            
            # Explicitly force 'x' and 'y' to point to the attached coordinate matrices
            if "lon" in data_array.coords and "lat" in data_array.coords:
                plot_kwargs.update({'x': 'lon', 'y': 'lat'})
            elif x_var and y_var:
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

    # Create source grid from native coordinates
    source_grid = xr.Dataset(coords={"lat": da.lat, "lon": da.lon})

    # Compute weights and initialize regridder
    weights_local_dir = _ensure_local_dir(weights_dir)
    weights_path = weights_local_dir / f"amsr2-lis-weights-{method}.nc"
    
    compute_weights(source_grid, lis_grid, weights_path, method=method, overwrite=overwrite_weights)
    regridder = load_regridder(source_grid, lis_grid, weights_path, method=method)

    # Prepare datasets for regridding
    ds_mean = da.to_dataset(name="snow_depth")
    
    # Quick diagnostic
    valid_native = np.count_nonzero(~np.isnan(ds_mean["snow_depth"].values))
    log.info(f"AMSR2 native matrix has {valid_native} valid pixels before regridding.")

    # Rechunk to a single large chunk to avoid chunk multiplication warnings
    log.info("AMSR2: rechunking data to single chunk for regridding...")
    ds_mean = ds_mean.chunk({"lat": -1, "lon": -1})

    log.info("AMSR2: regridding snow depth...")
    rg_mean = regridder(ds_mean)["snow_depth"]
    
    # Re-attach LIS coordinates
    rg_mean = rg_mean.assign_coords(lat=lis_grid.lat, lon=lis_grid.lon)

    # Find the max value for plotting
    plot_vmax = float(np.nanmax(rg_mean.values))
    log.info(f"Setting plot colorbar maximum to {plot_vmax:.2f}")

    # Generate debug plots
    da_plot_native = da.sortby("lon")
#    _debug_plot(da_plot_native, "AMSR2 Native Grid (Central U.S.)", "amsr2_01_native.png", extents=lis_extents, vmax=plot_vmax)
#    _debug_plot(rg_mean, "AMSR2 Regridded (Central U.S.)", "amsr2_03_regridded.png", extents=lis_extents, vmax=plot_vmax)

    result = {
        "amsr2_snow_depth_mean": xr.DataArray(
            rg_mean.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "AMSR2 snow depth (daily descending-pass mean)",
                "units": "mm",
                "source": "s3://airborne-smce-prod-user-bucket/JOIN/icechunk-stores/GCOM-W1-AMSR2-L3-SND"
            },
        )
    }

    return result
    

def _regrid_ceda(
    ceda_dir,
    lis_grid,
    method,
    weights_dir,
    current_date,
    overwrite_weights=False,
    fs=None
) -> dict:
    """
    Regrid CEDA SWE data using the Icechunk virtual Zarr V3 store.
    Includes regional plotting zoomed into the provided LIS grid extents.
    """
    date_str = current_date.strftime("%Y-%m-%d")
    log.info(f"Starting CEDA processing for {date_str}...")

    # Calculate LIS extents for zoomed-in plotting
    min_lon, max_lon = lis_grid.lon.min().item(), lis_grid.lon.max().item()
    min_lat, max_lat = lis_grid.lat.min().item(), lis_grid.lat.max().item()
    lis_extents = [min_lon - 2, max_lon + 2, min_lat - 2, max_lat + 2]

    # ==========================================
    # 1. Connect to Icechunk Store
    # ==========================================
    bucket = "airborne-smce-prod-user-bucket"
    prefix = "JOIN/icechunk-stores/CEDA_Store_v3/"
    base_ceda_url = "https://dap.ceda.ac.uk/neodc/esacci/snow/data/swe/MERGED/v4.0/"

    try:
        storage = icechunk.s3_storage(bucket=bucket, prefix=prefix)
        config = icechunk.RepositoryConfig.default()
        container = icechunk.VirtualChunkContainer(
            name="ceda_archive",       
            url_prefix=base_ceda_url,  
            store=icechunk.http_store()
        )
        config.set_virtual_chunk_container(container)

        repo = icechunk.Repository.open(
            storage=icechunk.s3_storage(
                bucket=bucket, 
                prefix=prefix, 
                from_env=True  # Force dynamic credential refresh
            ),
            config=config,
            authorize_virtual_chunk_access={base_ceda_url: icechunk.credentials.HttpAccess}
        )
        
        # repo = icechunk.Repository.open(
        #     storage=storage,
        #     config=config,
        #     authorize_virtual_chunk_access={base_ceda_url: icechunk.credentials.HttpAccess}
        # )
        session = repo.readonly_session("main")
        
        # Open virtual dataset 
        ds_full = xr.open_zarr(session.store, zarr_format=3)
    except Exception as e:
        log.error(f"Failed to open CEDA Icechunk store: {e}")
        return {}

    # ==========================================
    # 2. Select Data for Current Date
    # ==========================================
    try:
        # .squeeze() removes the time dimension leaving 2D (lat, lon) arrays
        swe_day = ds_full['swe'].sel(time=date_str).squeeze()
        
        # If the dataset has swe_std, grab it too. Otherwise fallback to empty/NaN.
        if 'swe_std' in ds_full:
            swe_std_day = ds_full['swe_std'].sel(time=date_str).squeeze()
        else:
            swe_std_day = xr.full_like(swe_day, np.nan)
            
        ds_ceda = xr.Dataset({"swe": swe_day, "swe_std": swe_std_day})
    except KeyError:
        log.warning(f"CEDA: No data found for {date_str}")
        return {}
    except Exception as e:
        log.error(f"CEDA: Error selecting data for {date_str}: {e}")
        return {}

    valid_native_swe = np.count_nonzero(~np.isnan(ds_ceda["swe"].values))
    log.info(f"CEDA native matrix has {valid_native_swe} valid pixels before regridding.")
    if valid_native_swe == 0:
        log.warning(f"CEDA: Found file but no valid SWE data for {date_str}.")
        return {}

    # ==========================================
    # 3. Generate Diagnostic Plot (Zoomed to LIS)
    # ==========================================
    log.info("Generating regional CEDA plot (fetching byte-ranges via HTTPS)...")
    Path("./plot").mkdir(parents=True, exist_ok=True)
    
    # We use PlateCarree since we are zooming into a specific rectangular boundary
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=':')
    ax.gridlines(draw_labels=True, alpha=0.3, color='gray', linestyle='--')

    # Zoom into the LIS domain extents
    ax.set_extent(lis_extents, crs=ccrs.PlateCarree())

    try:
        ds_ceda['swe'].plot(
            ax=ax, 
            transform=ccrs.PlateCarree(),
            cmap='Blues',           
            vmin=0,                 
            cbar_kwargs={
                'label': 'Snow Water Equivalent (mm)', 
                'shrink': 0.7, 
                'orientation': 'vertical'
            }
        )
        ax.set_title(f"ESA CCI Snow Water Equivalent (SWE) - {date_str}\n(LIS Domain)", fontsize=14, pad=10)
        plt.tight_layout()
        
        output_filename = f"./plot/swe_lis_domain_{date_str.replace('-', '')}.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        log.info(f"Plot saved to {output_filename}")
    except Exception as e:
        log.warning(f"Could not generate plot for {date_str}: {e}")
    finally:
        plt.close(fig)

    # ==========================================
    # 4. Regridding to LIS Grid
    # ==========================================
    # Define native source coordinates
    source_grid = xr.Dataset(coords={"lat": ds_ceda.lat, "lon": ds_ceda.lon})

    weights_local_dir = _ensure_local_dir(weights_dir)
    weights_path = weights_local_dir / f"ceda-lis-weights-{method}.nc"

    # Compute weights and load xESMF regridder
    compute_weights(
        source_grid,
        lis_grid,
        weights_path,
        method=method,
        overwrite=overwrite_weights,
    )
    regridder = load_regridder(
        source_grid,
        lis_grid,
        weights_path,
        method=method,
    )

    log.info("CEDA: rechunking data to single chunks for regridding...")
    ds_ceda = ds_ceda.chunk({"lat": -1, "lon": -1})

    log.info("CEDA: regridding swe and swe_std...")
    rg = regridder(ds_ceda)

    return {
        "ceda_swe": xr.DataArray(
            rg["swe"].values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "CEDA ESA CCI snow water equivalent",
                "units": "mm",
                "source": f"s3://{bucket}/{prefix}",
            },
        ),
        "ceda_swe_std": xr.DataArray(
            rg["swe_std"].values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "CEDA ESA CCI SWE standard deviation",
                "units": "mm",
                "source": f"s3://{bucket}/{prefix}",
            },
        ),
    }


def _viirs_tile_bbox(h: int, v: int) -> tuple[float, float, float, float]:
    """
    The VIIRS sinusoidal tile grid divides the globe into 36 × 18 tiles.
    Each tile spans exactly 10° of latitude and 10° equivalent in the
    sinusoidal projection. Tile (h=0, v=0) is the top-left (north-west) tile.
    """
    import math
    
    lat_max = 90.0 - v * 10.0
    lat_min = lat_max - 10.0
    
    def _lon_bound(h_index, lat_deg):
        lat_r = math.radians(abs(lat_deg))
        cos_lat = math.cos(lat_r) if lat_r < math.pi / 2 else 1e-9
        # The unprojected x-coordinate (in degrees) is -180 + h_index * 10
        x_deg = -180.0 + h_index * 10.0
        return x_deg / cos_lat

    # Find the longitude at the four corners of the tile
    lon_left_bottom = _lon_bound(h, lat_min)
    lon_right_bottom = _lon_bound(h + 1, lat_min)
    
    lon_left_top = _lon_bound(h, lat_max)
    lon_right_top = _lon_bound(h + 1, lat_max)
    
    # The bounding box uses the maximum extents
    lon_min = min(lon_left_bottom, lon_left_top)
    lon_max = max(lon_right_bottom, lon_right_top)
    
    return lon_min, lat_min, lon_max, lat_max

def _viirs_tile_overlaps_area(h: int, v: int, area) -> bool:
    """Return True if the VIIRS tile (h, v) may overlap *area*'s lat/lon bbox."""
    lon_min, lat_min, lon_max, lat_max = _viirs_tile_bbox(h, v)
    # Get the area's geographic bounding box from its 4 corner lon/lats
    try:
        import math as _math
        corners = area.outer_boundary_corners  # list of Coordinate objects (lon/lat in radians)
        # Coordinate objects have .lon and .lat attributes in radians
        corner_lons = [_math.degrees(c.lon) for c in corners]
        corner_lats = [_math.degrees(c.lat) for c in corners]
        a_lon_min, a_lon_max = min(corner_lons), max(corner_lons)
        a_lat_min, a_lat_max = min(corner_lats), max(corner_lats)
    except Exception:
        return True  # Can't determine — include tile to be safe
    return not (lon_max < a_lon_min or lon_min > a_lon_max or
                lat_max < a_lat_min or lat_min > a_lat_max)




def _regrid_viirs(
    lis_grid, 
    method: str, 
    current_date, 
    max_tiles: int = None,
    **kwargs
) -> dict:
    
    date_str = current_date.strftime("%Y-%m-%d")
    log.info(f"Starting VIIRS processing for {date_str} via Earthdata...")

    # 1. Calculate bounding box for Earthdata search
    try:
        if hasattr(lis_grid, 'get_lonlats'):
            lons, lats = lis_grid.get_lonlats()
            min_lon, max_lon = float(lons.min()), float(lons.max())
            min_lat, max_lat = float(lats.min()), float(lats.max())
        else:
            min_lon = float(lis_grid.lon.min().item())
            max_lon = float(lis_grid.lon.max().item())
            min_lat = float(lis_grid.lat.min().item())
            max_lat = float(lis_grid.lat.max().item())
    except Exception as e:
        log.warning(f"Could not extract bounding box from lis_grid: {e}. Falling back to default region.")
        min_lon, min_lat, max_lon, max_lat = -116.1, 35.32, -89.46, 50.59
        
    bounding_box = (min_lon, min_lat, max_lon, max_lat)

    # 2. Search for the granules directly via Earthdata
    short_name = "VJ110A1F"
    log.info(f"Searching Earthdata for {short_name} granules on {date_str} within bounding box {bounding_box}...")
    
    raw_results = earthaccess.search_data(
        short_name=short_name,
        bounding_box=bounding_box,
        temporal=(date_str, date_str)   
    )

    if not raw_results:
        log.warning(f"No VIIRS granules found for {date_str}. Skipping VIIRS processing.")
        return {}

    # 2.5 Filter to tiles that spatially overlap the LIS domain exactly
    import re
    results = []
    skipped = 0
    
    for granule in raw_results:
        # The filename is usually stored in the data links or URIs
        links = granule.data_links()
        if not links:
            continue
            
        fname = links[0].split('/')[-1]
        m = re.search(r"\.h(\d{2})v(\d{2})\.", fname)
        
        if m is not None:
            h, v = int(m.group(1)), int(m.group(2))
            lis_area_obj = getattr(lis_grid, "area", lis_grid)
            
            # Use your existing overlap check function
            if not _viirs_tile_overlaps_area(h, v, lis_area_obj):
                skipped += 1
                continue
                
        results.append(granule)

    if skipped:
        log.info(f"VIIRS: skipped {skipped} tile(s) outside precise LIS domain AreaDefinition")

    if not results:
        log.warning("VIIRS: no tiles overlap the LIS domain — skipping")
        return {}

    num_granules = len(results)
    log.info(f"VIIRS: using date {date_str} ({num_granules} tile(s))")

    if max_tiles is not None and num_granules > max_tiles:
        raise RuntimeError(
            f"VIIRS: {num_granules} tile(s) would be loaded but --max-viirs-tiles={max_tiles}. "
            "Aborting to prevent OOM. Either the spatial filter is not working correctly "
            "or the domain is unusually large. Increase --max-viirs-tiles only if expected."
        )

    # 3. Stream data directly
    try:
        log.info("Opening files directly via Earthdata S3 streaming...")
        file_objects = earthaccess.open(results)
    except Exception as e:
        log.error(f"Failed to open stream for VIIRS granules: {e}")
        return {}

    # Setup the projection transformer once
    transformer = pyproj.Transformer.from_crs(_SIN_CRS, "EPSG:4326", always_xy=True)

    # 4. Initialize an empty accumulator array sized to our target LIS grid
    lis_area_obj = getattr(lis_grid, "area", lis_grid)
    target_shape = lis_area_obj.shape
    combined_snow = np.full(target_shape, np.nan, dtype=np.float32)

    # 5. Process Tile-by-Tile (Strict Memory Bounding)
    log.info("Processing and regridding VIIRS tiles sequentially to conserve memory...")
    
    for i, file_obj in enumerate(file_objects):
        try:
            # A. Stream and extract just this tile
            with h5py.File(file_obj, "r") as f:
                raw = f[_HDFEOS_DATA_PATH][:]
                x_coords = f[_HDFEOS_XDIM_PATH][:]
                y_coords = f[_HDFEOS_YDIM_PATH][:]

            data = raw.astype(np.float32)
            data[data > 100] = np.nan

            # Optimization: If the tile is completely empty, skip pyresample entirely
            if np.all(np.isnan(data)):
                continue

            # B. Calculate coords for this tile only
            x2d, y2d = np.meshgrid(x_coords, y_coords)
            lon2d, lat2d = transformer.transform(x2d, y2d)

            source_def = SwathDefinition(
                lons=xr.DataArray(lon2d, dims=["y", "x"]), 
                lats=xr.DataArray(lat2d, dims=["y", "x"])
            )
            
            da_snow = xr.DataArray(data, dims=["y", "x"])
            da_snow.attrs["area"] = source_def

            # C. Regrid just this tile
            log.info(f"  -> Regridding tile {i+1}/{len(file_objects)}...")
            rg_tile = regrid(da_snow, source_def, lis_area_obj, method=method)

            # D. Accumulate into the master LIS array
            # Where the tile has valid data, write it into the combined array
            tile_data = rg_tile.values
            valid_mask = ~np.isnan(tile_data)
            combined_snow[valid_mask] = tile_data[valid_mask]
            
            # E. Memory cleanup (variables overwritten in next loop, file stream closed)
        except Exception as e:
            log.warning(f"Could not load or regrid VIIRS granule {i+1}: {e}")
        finally:
            file_obj.close()

    # 6. Format output mapping
    result = {
        "viirs_cgf_ndsi_snow_cover": xr.DataArray(
            combined_snow,
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "VIIRS Cloud Gap Filled NDSI Snow Cover",
                "units": "percentage",
                "source": "NASA Earthdata (VJ110A1F via earthaccess)",
            },
        )
    }
    
    log.info("VIIRS Earthdata streaming and regridding completed successfully.")
    return result

def _regrid_icesat2(
    df_full,  # Now taking a pandas DataFrame
    lis_area,
    current_date: pd.Timestamp,
    source_path: str,
    fs=None,
) -> dict[str, xr.DataArray]:

    print("starting the _regrid_icesat2")
    
    """Regrid ICESat-2 ATL06 data and compute snow depth (h_mean - mosaic.median)."""
    
    # 1. TEMPORAL FILTERING: Slice data for the current 24-hour period
    date_str = current_date.strftime('%Y-%m-%d')
    start_of_day = f"{date_str} 00:00:00"
    end_of_day = f"{date_str} 23:59:59"
    
    try:
        # Pandas allows string-based date slicing if 'time' is the index
        df_daily = df_full.loc[start_of_day:end_of_day]
    except Exception as e:
        log.warning(f"ICESat-2: Failed to filter by time: {e}")
        return {}
        
    # If the slice is empty, skip this date
    if len(df_daily) == 0:
        log.info(f"ICESat-2: No data points found for {date_str}")
        return {}

    log.info(f"ICESat-2: Found {len(df_daily)} points for {date_str}. Regridding to LIS grid...")

    # Extract required arrays using Pandas to_numeric to safely force float conversion
    try:
        import pandas as pd
        import numpy as np
        
        lons = pd.to_numeric(df_daily["lon"], errors="coerce").to_numpy(dtype=np.float64)
        lats = pd.to_numeric(df_daily["lat"], errors="coerce").to_numpy(dtype=np.float64)
        h_mean = pd.to_numeric(df_daily["h_mean"], errors="coerce").to_numpy(dtype=np.float64)
        
        # Unpack the list structure in mosaic.median
        if "mosaic.median" in df_daily.columns:
            # Custom function to grab the first item if it's a list, otherwise NaN
            def extract_val(x):
                if isinstance(x, (list, np.ndarray)):
                    return x[0] if len(x) > 0 else np.nan
                return x # In case it's already a float
                
            # Apply unpacking, then coerce to float64
            dem_vals_unpacked = df_daily["mosaic.median"].apply(extract_val)
            dem_vals = pd.to_numeric(dem_vals_unpacked, errors="coerce").to_numpy(dtype=np.float64)
        else:
            log.warning("ICESat-2: 'mosaic.median' column not found in Parquet. Cannot compute snow depth.")
            return {}
            
    except Exception as e:
        log.error(f"ICESat-2: Error extracting column values: {e}")
        return {}
    
    # Calculate snow depth
    snow_depth_vals = h_mean - dem_vals
    # snow_depth_vals = np.where(
    #     (h_mean != 0.0) & (dem_vals != 0.0), 
    #     h_mean - dem_vals, 
    #     np.nan
    # ) 

    # Create the source area definition for Pyresample from the flat arrays
    lons_da = xr.DataArray(lons, dims=["obs"])
    lats_da = xr.DataArray(lats, dims=["obs"])
    source_def = SwathDefinition(lons=lons_da, lats=lats_da)

    # Wrap the pure float64 data arrays AND attach the source_def to their metadata
    da_snow = xr.DataArray(snow_depth_vals, dims=["obs"])
    da_snow.attrs["area"] = source_def

    da_h_mean = xr.DataArray(h_mean, dims=["obs"])
    da_h_mean.attrs["area"] = source_def

    da_dem = xr.DataArray(dem_vals, dims=["obs"])
    da_dem.attrs["area"] = source_def

    # Use the existing regrid wrapper
    from join_scratch.regrid import regrid

    print(lis_area)
    
    log.info("ICESat-2: Regridding snow depth...")
    rg_snow_depth = regrid(da_snow, source_def, lis_area, method="mean")
    
    log.info("ICESat-2: Regridding h_mean...")
    rg_h_mean = regrid(da_h_mean, source_def, lis_area, method="mean")

    log.info("ICESat-2: Regridding 3DEP DEM...")
    rg_dem = regrid(da_dem, source_def, lis_area, method="mean")

    log.info("ICESat-2: Regridding snow depth...")
    rg_snow_depth = regrid(da_snow, source_def, lis_area, method="mean")
    
    log.info("ICESat-2: Regridding h_mean...")
    rg_h_mean = regrid(da_h_mean, source_def, lis_area, method="mean")

    # ---------------------------------------------------------
    # NEW DIAGNOSTICS: Inspect the 2D grid after regridding
    # ---------------------------------------------------------
    # grid_hli_valid = np.count_nonzero(~np.isnan(rg_h_mean.values))
    # grid_snow_valid = np.count_nonzero(~np.isnan(rg_snow_depth.values))
    # grid_snow_zeros = np.count_nonzero(rg_snow_depth.values == 0.0)
    
    # log.info(f"ICESat-2 Grid Diagnostics:")
    # log.info(f"  -> Gridded h_mean pixels: {grid_hli_valid}")
    # log.info(f"  -> Gridded snow pixels (total valid): {grid_snow_valid}")
    # log.info(f"  -> Gridded snow pixels (exactly zero): {grid_snow_zeros}")
    # ---------------------------------------------------------

    return {
        "icesat2_snow_depth": xr.DataArray(
            rg_snow_depth.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "ICESat-2 estimated snow depth (h_mean - mosaic.median)",
                "units": "meters",
                "source": f"{source_path}",
            }
        ),
        "icesat2_h_mean": xr.DataArray(
            rg_h_mean.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "ICESat-2 ATL06 land-ice surface height",
                "units": "meters",
                "source": source_path,
            }
        ),
        "icesat2_3dep_dem_10m": xr.DataArray(
            rg_dem.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "ICESat-2 Sliderule 3DEP DEM 10M",
                "units": "meters",
                "source": source_path,
            }
        )
    }
    
def _write_output(ds_out: xr.Dataset, output_path: str, encoding: dict, fs=None) -> None:
    """Write *ds_out* to *output_path*, which may be a local path or S3 URI."""
    if _is_s3(output_path):
        if fs is None:
            fs = make_fs()
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            ds_out.to_netcdf(tmp_path, engine="h5netcdf", encoding=encoding)
            with open(tmp_path, "rb") as f:
                data = f.read()
            with fs.open(output_path, "wb") as fout:
                fout.write(data)
        finally:
            os.unlink(tmp_path)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(output_path, engine="h5netcdf", encoding=encoding)



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regrid all SWE input datasets to the LIS grid and combine into "
            "daily NetCDF files. All paths may be local or s3:// URIs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    
    parser.add_argument("--lis-path", required=True,
                        help="Path to the LIS input NetCDF file (local or s3://).")
    parser.add_argument("--amsr2-dir", default=None,
                        help="Process AMSR2 data from Icechunk store (flag only, no path needed).")
    parser.add_argument("--ceda-dir", default=None,
                        help="Directory containing CEDA ESA CCI SWE NetCDF files (local or s3://).")
    parser.add_argument("--viirs-dir", default=None,
                        help="Directory containing VIIRS CGF snow cover HDF5 files (local or s3://).")
    parser.add_argument("--weights-dir", default="_data/weights",
                        help="Local directory for xESMF weights files.")
    parser.add_argument("--output-path", default="_data/swe_combined.nc",
                        help="Base output path. Dates will be injected automatically.")
    
    # Per-dataset method overrides
    parser.add_argument("--amsr2-method", default="bilinear",
                        choices=["bilinear", "nearest_s2d", "conservative"],
                        help="xESMF regridding method for AMSR2.")
    parser.add_argument("--ceda-method", default="bilinear",
                        choices=["bilinear", "nearest_s2d", "conservative"],
                        help="xESMF regridding method for CEDA.")
    parser.add_argument("--viirs-method", default="nearest",
                        choices=["nearest", "bilinear", "ewa", "bucket_avg"],
                        help="pyresample regridding method for VIIRS.")
    parser.add_argument("--overwrite-weights", action="store_true", default=False,
                        help="Always recompute xESMF weights even if cached files exist.")
    parser.add_argument("--max-viirs-tiles", type=int, default=None, metavar="N",
                        help="Abort before loading if the filtered VIIRS tile count exceeds N.")
    
    ns = parser.parse_args()

    # ---------------------------------------------------------
    # [NEW] Authenticate with NASA Earthdata Login
    # ---------------------------------------------------------
    log.info("Authenticating with NASA Earthdata...")
    try:
        # If credentials aren't in env variables or ~/.netrc, it will prompt here
        earthaccess.login(persist=True)
        log.info("Earthdata authentication successful.")
    except Exception as e:
        log.error(f"Earthdata login failed. Cannot access VIIRS data: {e}")
        return
    # ---------------------------------------------------------
    

    # Build a shared fsspec store if any S3 paths are present
    any_s3 = any(
        _is_s3(str(p))
        for p in [ns.lis_path, ns.amsr2_dir, ns.ceda_dir, ns.viirs_dir]
        if p is not None
    )
    fs = make_fs() if any_s3 else None

    log.info("Loading LIS grid and AreaDefinition...")
    lis_grid = load_lis_grid(ns.lis_path, fs=fs)    
    lis_area = build_lis_area_definition(ns.lis_path, fs=fs, cache_dir=ns.weights_dir, overwrite=ns.overwrite_weights)
  
    # Generate a list of daily dates
    dates = pd.date_range(start=ns.start_date, end=ns.end_date, freq='D')
    
    # -------------------------------------------------------------------------
    # MAIN TIME LOOP
    # -------------------------------------------------------------------------
    for current_date in dates:
        # 1. Refresh global AWS credentials (boto3/os.environ)
        refresh_aws_credentials()
        
        # 2. Re-instantiate the fsspec FileSystem so it picks up the fresh os.environ variables
        if any_s3:
            fs = make_fs()
            
        # Use %Y%m%d to get '20190101' instead of '2019-01-01'
        date_str = current_date.strftime('%Y%m%d') 
        log.info(f"--- Processing date: {date_str} ---")     
    
        data_vars: dict[str, xr.DataArray] = {}

        if ns.amsr2_dir is not None:
            data_vars.update(_regrid_amsr2(ns.amsr2_dir, lis_grid, ns.amsr2_method, ns.weights_dir, current_date, overwrite_weights=ns.overwrite_weights, fs=fs))
    
        if ns.ceda_dir is not None:
            data_vars.update(_regrid_ceda(ns.ceda_dir, lis_grid, ns.ceda_method, ns.weights_dir, current_date, overwrite_weights=ns.overwrite_weights, fs=fs))
    
        if ns.viirs_dir is not None:
            #data_vars.update(_regrid_viirs(ns.viirs_dir, lis_area, ns.viirs_method, current_date, fs=fs, max_tiles=ns.max_viirs_tiles))
            data_vars.update(_regrid_viirs(lis_area, ns.viirs_method, current_date, fs=fs, max_tiles=ns.max_viirs_tiles))
    
        # Load ICESat-2 data for this specific date from S3
        icesat2_s3_path = f"s3://airborne-smce-prod-user-bucket/JOIN/ICESAT-2/ATL06_MOSAIC/{current_date.strftime('%Y/%m/%d')}/atl06_{date_str}.parquet"
        try:
            log.info(f"Loading ICESat-2 data from {icesat2_s3_path}...")
            icesat2_df = pd.read_parquet(icesat2_s3_path)
            
            if len(icesat2_df) > 0:
                log.info(f"ICESat-2 Parquet columns: {list(icesat2_df.columns)}")
                icesat2_vars = _regrid_icesat2(
                    icesat2_df,
                    lis_area,
                    current_date,
                    source_path=icesat2_s3_path
                )
                data_vars.update(icesat2_vars)
            else:
                log.info(f"ICESat-2: No data points found for {date_str}")
        except FileNotFoundError:
            log.info(f"ICESat-2 file not found for {date_str} at {icesat2_s3_path}")
        except Exception as e:
            log.warning(f"Failed to load ICESat-2 data for {date_str}: {e}")
    
        # Check if we actually found any data for today
        if not data_vars:
            log.warning(f"No valid data found for {date_str}. Skipping file generation.")
            continue

        # Build a perfectly flat 2D daily dataset (NO 'time' dimension!)
        ds_day = xr.Dataset(
            data_vars,
            coords={
                "lat": lis_grid["lat"],
                "lon": lis_grid["lon"]
            },
            attrs={
                "description": f"Daily combined SWE and snow-cover observations for {date_str}.",
                "date": date_str,
                "conventions": "CF-1.8",
            }
        )

        # Build the dynamic output filename
        out_path_obj = Path(ns.output_path)
        if out_path_obj.suffix == '.nc':
            daily_out_path = out_path_obj.parent / f"{out_path_obj.stem}_{date_str}{out_path_obj.suffix}"
        else:
            out_path_obj.mkdir(parents=True, exist_ok=True)
            daily_out_path = out_path_obj / f"swe_combined_{date_str}.nc"

        # Reverse the entire dataset along the north_south dimension 
        # (orders data from lowest to highest latitude / South to North)
        ds_day = ds_day.isel(north_south=slice(None, None, -1))
        
        # 1. Fill all NaN values in the dataset with -9999.0
        ds_day = ds_day.fillna(-9999.0)

        # 2. Build the encoding dictionary for standard data variables
        encoding = {
            var: {"dtype": "float32", "_FillValue": -9999.0, "zlib": True}
            for var in ds_day.data_vars
        }
        
        # 3. Explicitly add coordinate encoding to fix the NaNf mismatch!
        # It's important NOT to compress (zlib) 1D/2D coordinates for faster CF-compliant reading
        encoding["lat"] = {"dtype": "float32", "_FillValue": -9999.0}
        encoding["lon"] = {"dtype": "float32", "_FillValue": -9999.0}
        
        log.info(f"Writing daily output to {daily_out_path} …")
        ds_day.to_netcdf(daily_out_path, encoding=encoding)
        
        # Explicitly close and clean up to prevent memory spikes
        ds_day.close()
        del ds_day
        
        log.info(f"Finished {date_str} successfully!")

    log.info("Workflow complete. All valid daily files have been generated.")

if __name__ == "__main__":
    main()