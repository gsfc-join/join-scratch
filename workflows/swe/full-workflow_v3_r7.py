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
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.kd_tree import resample_nearest

import pyproj

from sliderule import icesat2

import os
import boto3
import time

import icechunk
import fsspec

import earthaccess
import numpy as np
import logging

import re
import gc

log = logging.getLogger(__name__)

# Constants for VIIRS HDF-EOS5 Structure
_HDFEOS_DATA_PATH = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields/CGF_NDSI_Snow_Cover"
_HDFEOS_XDIM_PATH = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/XDim"
_HDFEOS_YDIM_PATH = "HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/YDim"

# Standard MODIS/VIIRS global sinusoidal tile grid definition (LP DAAC).
_MODIS_SIN_TILE_SIZE_M = 1111950.519667   # meters per tile edge
_MODIS_SIN_X_MIN = -20015109.354          # left edge of h00 column, meters
_MODIS_SIN_Y_MAX = 10007554.677           # top edge of v00 row, meters
_VIIRS_H_MAX = 35   # tile columns: h00..h35
_VIIRS_V_MAX = 17   # tile rows:    v00..v17

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
#GPORTAL_URL = "https://gportal.jaxa.jp/"
GPORTAL_URL = "https://www.gportal.jaxa.jp/"
CEDA_URL = "https://dap.ceda.ac.uk/"

import boto3
import os

def _check_sliderule_availability(domain: str = "slideruleearth.io", verbose: bool = True) -> bool:
    """
    Pre-flight + actual initialization of the SlideRule client.

    icesat2.init() (wrapping sliderule.init()) performs a version check
    against the server and returns a bool indicating whether that check
    succeeded, so this single call both validates connectivity AND
    performs the one-time client initialization the workflow needs later
    -- no separate init() call should be made elsewhere in the script.

    rethrow=False (the default) means init() will not raise on failure;
    it should just return False, which is what we rely on here. If a
    truly unreachable server instead causes init() to hang rather than
    fail fast, a socket timeout is applied as a safety net.
    """
    log.info(f"Checking SlideRule API server availability at {domain} (and initializing client)...")
    import socket
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)
    try:
        status = icesat2.init(domain, verbose=verbose, rethrow=False)
    except Exception as e:
        log.warning(f"SlideRule API server check raised an exception: {e}")
        status = False
    finally:
        socket.setdefaulttimeout(old_timeout)

    if status:
        log.info("SlideRule API server is reachable; client initialized.")
    else:
        log.warning("SlideRule API server version check failed -- server may be unavailable.")
    return bool(status)

def refresh_aws_credentials():
    """Fetches fresh credentials from the boto3 session and injects them into the environment."""
    log.info("Refreshing AWS credentials...")
    
    # Create a fresh session
    session = boto3.Session()
    creds = session.get_credentials()
    region = session.region_name or "us-west-2" 
    
    if creds:
        # get_frozen_credentials() gets the credentials
        frozen_creds = creds.get_frozen_credentials()
        os.environ["AWS_ACCESS_KEY_ID"] = frozen_creds.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = frozen_creds.secret_key
        if frozen_creds.token:
            os.environ["AWS_SESSION_TOKEN"] = frozen_creds.token
            
        os.environ["AWS_REGION"] = region
        os.environ["AWS_DEFAULT_REGION"] = region
        os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
        
        # NOTE: We do not log the actual token, just that it was successfully refreshed
        log.info(f"Successfully refreshed and injected AWS credentials (Region: {region}).")
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
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import h5py
def _build_amsr2_grid_coords(data_shape: tuple[int, int]):
    """
    Build lat/lon coordinate arrays for the AMSR2 L3 0.1deg global EQR grid.
    (unchanged -- see prior confirmation against the validated Icechunk pipeline)
    """
    nrows, ncols = data_shape
    lat_data = np.linspace(89.95, -89.95, nrows).astype("float32")
    lon_data = np.linspace(0.05, 359.95, ncols).astype("float32")
    return lat_data, lon_data


def _regrid_amsr2(
    input_dir: str,
    lis_grid: xr.Dataset,
    method: str,
    weights_dir: str,
    current_date: pd.Timestamp,
    overwrite_weights: bool = False,
    fs=None,
) -> dict[str, xr.DataArray]:
    """
    Regrid AMSR2 data from G-Portal SFTP and return {var_name: DataArray}.

    File selection, per JAXA GCOM-W AMSR2 L3 filename convention:
        GW1AM2_<YYYYMMDD>_<period>_<pass><grid>_<product>.h5
      - '01D'  -> 0.1deg Daily product (REQUIRED -- rejects weekly/monthly
                  or other-resolution variants that might otherwise match
                  a loose glob).
      - 'EQMD' -> Equirectangular, Descending pass (REQUIRED -- 'EQMA' is
                  Ascending and must NOT be selected for the descending
                  output variable).
    The file's own 'OrbitDirection' global attribute is checked after
    opening as an independent confirmation that the filename-based
    selection actually matches the pass contained in the file, since
    relying on the filename alone is fragile.

    Band layout of 'Geophysical Data' (shape (lat, lon, 2)):
      index 0 -> mean snow depth (descending pass, daily, 0.1deg)
      index 1 -> standard deviation of snow depth (same pass/product)

    Units: JAXA stores this product in cm (confirmed via the dataset's
    own 'UNIT' attribute). Output is converted to mm (cm * 10) before
    being returned.
    """

    log.info("AMSR2: Reading data via direct h5py over SFTP...")

    if fs is None:
        log.error("AMSR2: No SFTP filesystem (fs) provided. Please pass an initialized fsspec SFTP filesystem.")
        return {}

    year_str = current_date.strftime("%Y")
    month_str = current_date.strftime("%m")
    date_str_compact = current_date.strftime("%Y%m%d")

    base_dir = input_dir.rstrip("/")
    remote_dir = f"{base_dir}/{year_str}/{month_str}"

    try:
        # Require '01D' (0.1deg Daily) AND 'EQMD' (Equirectangular,
        # Descending) explicitly in the filename, rather than a loose
        # '*SND*' glob that could silently match the wrong period or pass
        # (e.g. the Ascending 'EQMA' file seen in earlier runs).
        search_pattern = f"{remote_dir}/*{date_str_compact}_01D_EQMD*SND*.h5"
        log.info(f"AMSR2: Searching for files using pattern: {search_pattern}")
        matching_files = fs.glob(search_pattern)

        if not matching_files:
            log.warning(
                f"AMSR2: No 0.1deg Daily Descending (01D_EQMD) HDF5 file found for "
                f"{date_str_compact} in {remote_dir}. Not falling back to an "
                "Ascending (EQMA) file, since that would silently mislabel the pass."
            )
            return {}

        target_file = matching_files[0]
        if len(matching_files) > 1:
            log.warning(
                f"AMSR2: multiple files matched {search_pattern}; using the first: "
                f"{target_file}"
            )
        log.info(f"AMSR2: Found target file: {target_file}")

        with fs.open(target_file, 'rb') as remote_file:
            with h5py.File(remote_file, 'r') as h5f:

                # --- Independent confirmation via global attribute ---
                orbit_attr = h5f.attrs.get("OrbitDirection")
                orbit_direction = None
                if orbit_attr is not None:
                    raw_val = np.ravel(orbit_attr)[0]
                    orbit_direction = raw_val.decode() if isinstance(raw_val, bytes) else str(raw_val)

                if orbit_direction is None:
                    log.warning(
                        "AMSR2: file has no 'OrbitDirection' attribute to confirm pass; "
                        "proceeding based on filename ('EQMD') only."
                    )
                elif orbit_direction.strip().lower() != "descending":
                    log.error(
                        f"AMSR2: filename indicated Descending (EQMD) but the file's "
                        f"'OrbitDirection' attribute says '{orbit_direction}'. "
                        f"Rejecting {target_file} to avoid mislabeling the pass."
                    )
                    return {}
                else:
                    log.info(f"AMSR2: confirmed OrbitDirection='{orbit_direction}' matches EQMD filename tag.")

                snd_key = 'Geophysical Data'
                if snd_key not in h5f:
                    log.warning(f"AMSR2: {snd_key} not found. Keys available: {list(h5f.keys())}")
                    snd_key = list(h5f.keys())[0]

                raw_data = h5f[snd_key][:]

                # --- Band assignment (confirmed): index 0 = mean snow depth,
                # index 1 = standard deviation, both for the Descending pass
                # contained in this EQMD file. ---
                mean_layer = None
                std_layer = None
                if raw_data.ndim == 3:
                    if raw_data.shape[-1] == 2:
                        mean_layer = raw_data[:, :, 0]
                        std_layer = raw_data[:, :, 1]
                    elif raw_data.shape[0] == 2:
                        mean_layer = raw_data[0, :, :]
                        std_layer = raw_data[1, :, :]
                    else:
                        log.warning(
                            f"AMSR2: unexpected band dimension in shape {raw_data.shape}; "
                            "using index 0 as mean, no std available."
                        )
                        mean_layer = raw_data[:, :, 0]
                else:
                    mean_layer = raw_data

                # --- Confirmed grid definition (matches the validated
                # Icechunk-store pipeline for this exact product) ---
                nrows, ncols = mean_layer.shape
                lat_data, lon_data = _build_amsr2_grid_coords(mean_layer.shape)
                log.info(
                    f"AMSR2: grid coordinates set to confirmed 0.1deg global EQR grid "
                    f"(lat {lat_data[0]:.2f}->{lat_data[-1]:.2f}, "
                    f"lon {lon_data[0]:.2f}->{lon_data[-1]:.2f})"
                )

                attrs = h5f[snd_key].attrs
                scale_factor_raw = attrs.get('SCALE FACTOR', attrs.get('SCALE_FACTOR', 0.1))
                fill_value_raw = attrs.get('ERROR_VALUE', -9999)

                # These HDF5 attributes have a (1,) dataspace, so h5py returns a
                # 1-element numpy array rather than a plain scalar. Coerce to
                # Python float/scalar here so later f-string formatting
                # (e.g. ":.3f") and arithmetic behave as expected everywhere.
                scale_factor = float(np.ravel(scale_factor_raw)[0])
                fill_value = float(np.ravel(fill_value_raw)[0])
                
                unit_attr = attrs.get('UNIT', b'cm')
                unit_str = unit_attr[0].decode() if isinstance(unit_attr, np.ndarray) else str(unit_attr)
                if unit_str.strip().lower() != "cm":
                    log.warning(
                        f"AMSR2: dataset UNIT attribute reports '{unit_str}', not 'cm' as "
                        "expected. Output is still being converted assuming the source "
                        "units are cm (and returned as mm) -- verify this file's units "
                        "if that warning persists."
                    )

                def _scale_to_mm(layer):
                    layer_f = layer.astype(np.float32)
                    layer_f[layer == fill_value] = np.nan
                    layer_f[layer_f < 0] = np.nan
                    return layer_f * scale_factor * 10.0  # cm -> mm

                snow_depth_mean = _scale_to_mm(mean_layer)
                snow_depth_std = _scale_to_mm(std_layer) if std_layer is not None else None

                log.info(
                    f"AMSR2: scale_factor={scale_factor} -> output mm = raw_DN * {scale_factor} * 10.0 "
                    f"(= raw_DN * {scale_factor * 10.0:.3f})"
                )
                da_mean = xr.DataArray(
                    snow_depth_mean, coords={"lat": lat_data, "lon": lon_data}, dims=["lat", "lon"]
                )
                da_std = None
                if snow_depth_std is not None:
                    da_std = xr.DataArray(
                        snow_depth_std, coords={"lat": lat_data, "lon": lon_data}, dims=["lat", "lon"]
                    )

        log.info(f"AMSR2: Extracted descending-pass snow depth mean with shape {snow_depth_mean.shape}")
        if da_std is not None:
            log.info(f"AMSR2: Extracted descending-pass snow depth standard deviation with shape {snow_depth_std.shape}")
        else:
            log.warning("AMSR2: no standard-deviation band available in this file.")

    except Exception as e:
        log.error(f"AMSR2: Error reading HDF5 via SFTP: {e}")
        return {}

    def _fix_lon_wrap(da):
        if 'lon' in da.coords and da.lon.max() > 180:
            return da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")
        return da

    da_mean = _fix_lon_wrap(da_mean)
    if da_std is not None:
        da_std = _fix_lon_wrap(da_std)

    log.info("AMSR2: Preparing weights and regridding...")

    source_grid = xr.Dataset(coords={"lat": da_mean.lat, "lon": da_mean.lon})

    weights_local_dir = Path(weights_dir)
    weights_local_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_local_dir / f"amsr2-lis-weights-{method}.nc"

    compute_weights(source_grid, lis_grid, weights_path, method=method, overwrite=overwrite_weights)
    regridder = load_regridder(source_grid, lis_grid, weights_path, method=method)

    # Regrid mean (and std, if available) together in one Dataset so the
    # regridder only needs to be invoked once.
    ds_in = {"snow_depth_mean": da_mean}
    if da_std is not None:
        ds_in["snow_depth_std"] = da_std
    ds_in = xr.Dataset(ds_in)

    valid_native = np.count_nonzero(~np.isnan(ds_in["snow_depth_mean"].values))
    log.info(f"AMSR2 native matrix has {valid_native} valid pixels before regridding.")

    log.info("AMSR2: rechunking data to single chunk for regridding...")
    ds_in = ds_in.chunk({"lat": -1, "lon": -1})

    log.info("AMSR2: regridding snow depth...")
    rg = regridder(ds_in)

    rg_mean = rg["snow_depth_mean"].assign_coords(lat=lis_grid.lat, lon=lis_grid.lon)
    rg_mean.attrs["long_name"] = "AMSR2 snow depth mean (descending pass, daily, 0.1deg)"
    rg_mean.attrs["units"] = "mm"   # changed from "cm"

    result = {
        "amsr2_snow_depth_mean": xr.DataArray(
            rg_mean.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "AMSR2 snow depth mean (descending pass, daily, 0.1deg)",
                "units": "mm",   # changed from "cm"
                "source": f"JAXA G-Portal SFTP: {input_dir}",
            },
        )
    }

    if "snow_depth_std" in rg:
        rg_std = rg["snow_depth_std"].assign_coords(lat=lis_grid.lat, lon=lis_grid.lon)
        result["amsr2_snow_depth_std"] = xr.DataArray(
            rg_std.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "AMSR2 snow depth standard deviation (descending pass, daily, 0.1deg)",
                "units": "mm",   # changed from "cm"
                "source": f"JAXA G-Portal SFTP: {input_dir}",
            },
        )

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
        
        # MASK NEGATIVE FLAG VALUES
        # Flag values (-30, -20, -10, -1) will be replaced with np.nan 
        # so they are ignored during regridding. We keep 0 (no_snow) and above.
        swe_day = swe_day.where(swe_day >= 0)
        
        # If the dataset has swe_std, grab it too. Otherwise fallback to empty/NaN.
        if 'swe_std' in ds_full:
            swe_std_day = ds_full['swe_std'].sel(time=date_str).squeeze()
            
            # Apply the same mask to the uncertainty variable so they match
            swe_std_day = swe_std_day.where(swe_day >= 0)
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

    # # ==========================================
    # # 3. Generate Diagnostic Plot (Zoomed to LIS)
    # # ==========================================
    # log.info("Generating regional CEDA plot (fetching byte-ranges via HTTPS)...")
    # Path("./plot").mkdir(parents=True, exist_ok=True)
    
    # # We use PlateCarree since we are zooming into a specific rectangular boundary
    # fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    # ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=':')
    # ax.gridlines(draw_labels=True, alpha=0.3, color='gray', linestyle='--')

    # # Zoom into the LIS domain extents
    # ax.set_extent(lis_extents, crs=ccrs.PlateCarree())

    # # try:
    #     ds_ceda['swe'].plot(
    #         ax=ax, 
    #         transform=ccrs.PlateCarree(),
    #         cmap='Blues',           
    #         vmin=0,                 
    #         cbar_kwargs={
    #             'label': 'Snow Water Equivalent (mm)', 
    #             'shrink': 0.7, 
    #             'orientation': 'vertical'
    #         }
    #     )
    #     ax.set_title(f"ESA CCI Snow Water Equivalent (SWE) - {date_str}\n(LIS Domain)", fontsize=14, pad=10)
    #     plt.tight_layout()
        
    #     output_filename = f"./plot/swe_lis_domain_{date_str.replace('-', '')}.png"
    #     plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    #     log.info(f"Plot saved to {output_filename}")
    # except Exception as e:
    #     log.warning(f"Could not generate plot for {date_str}: {e}")
    # finally:
    #     plt.close(fig)

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
    10deg x 10deg sinusoidal tile bounding box. Near the poles the
    unprojected-longitude math can produce values far outside [-180,180]
    (division by a very small cos(lat)); CLAMP (not wrap) to valid range
    since the true geographic extent can never exceed it. This matters
    for higher-latitude domains where several of the v<=03 "polar" tiles
    live.
    """
    import math
    lat_max = 90.0 - v * 10.0
    lat_min = lat_max - 10.0

    def _lon_bound(h_index, lat_deg):
        lat_r = math.radians(abs(lat_deg))
        cos_lat = math.cos(lat_r) if lat_r < math.pi / 2 else 1e-9
        x_deg = -180.0 + h_index * 10.0
        return x_deg / cos_lat

    lons = [
        _lon_bound(h, lat_min), _lon_bound(h + 1, lat_min),
        _lon_bound(h, lat_max), _lon_bound(h + 1, lat_max),
    ]
    lon_min = max(-180.0, min(180.0, min(lons)))
    lon_max = max(-180.0, min(180.0, max(lons)))
    return lon_min, lat_min, lon_max, lat_max


def _viirs_tile_overlaps_area(h: int, v: int, area) -> bool:
    lon_min, lat_min, lon_max, lat_max = _viirs_tile_bbox(h, v)
    try:
        import math as _math
        corners = area.outer_boundary_corners
        corner_lons = [_math.degrees(c.lon) for c in corners]
        corner_lats = [_math.degrees(c.lat) for c in corners]
        a_lon_min, a_lon_max = min(corner_lons), max(corner_lons)
        a_lat_min, a_lat_max = min(corner_lats), max(corner_lats)
    except Exception:
        return True
    return not (lon_max < a_lon_min or lon_min > a_lon_max or
                lat_max < a_lat_min or lat_min > a_lat_max)



def _resolve_pad_px(ds: "h5py.Dataset", requested_pad_px: int) -> int:
    """
    Clamp the border padding to the dataset's chunk shape. Padding wider
    than one chunk forces reads of additional chunk rows/cols with no
    extra benefit (the resampler only needs a few pixels of context, not
    an extra full chunk), so clamp to chunk size and warn once.
    """
    chunks = ds.chunks
    if chunks is None:
        return requested_pad_px  # not chunked -- nothing to align to
    chunk_rows, chunk_cols = chunks
    max_safe_pad = min(chunk_rows, chunk_cols)
    if requested_pad_px > max_safe_pad:
        log.warning(
            f"VIIRS: --viirs-pad-px={requested_pad_px} exceeds dataset chunk size "
            f"{chunks}; clamping to {max_safe_pad}px to avoid pulling extra chunk "
            "rows/columns from neighbor tiles with no added benefit."
        )
        return max_safe_pad
    return requested_pad_px
    
def _parse_tile_hv(fname: str) -> tuple[int, int] | None:
    m = re.search(r"\.h(\d{2})v(\d{2})\.", fname)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _viirs_tile_corner_xy(h: int, v: int) -> tuple[float, float]:
    x_min = _MODIS_SIN_X_MIN + h * _MODIS_SIN_TILE_SIZE_M
    y_max = _MODIS_SIN_Y_MAX - v * _MODIS_SIN_TILE_SIZE_M
    return x_min, y_max


def _expand_tile_set_with_halo(tiles: set[tuple[int, int]], radius: int = 1) -> set[tuple[int, int]]:
    """Used only for logging / the --max-viirs-tiles sanity check."""
    expanded = set(tiles)
    for h, v in tiles:
        for dv in range(-radius, radius + 1):
            vv = v + dv
            if vv < 0 or vv > _VIIRS_V_MAX:
                continue
            for dh in range(-radius, radius + 1):
                expanded.add(((h + dh) % (_VIIRS_H_MAX + 1), vv))
    return expanded


def _search_tile_granule(short_name: str, h: int, v: int, date_str: str):
    """Locate the single granule covering tile (h, v). Used for halo tiles
    that fall outside the domain bbox search but are needed as thin-border
    context so seams don't land on the domain edge."""
    lon_min, lat_min, lon_max, lat_max = _viirs_tile_bbox(h, v)
    pad_lon = (lon_max - lon_min) * 0.25
    pad_lat = (lat_max - lat_min) * 0.25
    bbox = (lon_min + pad_lon, lat_min + pad_lat, lon_max - pad_lon, lat_max - pad_lat)
    try:
        candidates = earthaccess.search_data(
            short_name=short_name, bounding_box=bbox, temporal=(date_str, date_str)
        )
    except Exception as e:
        log.debug(f"VIIRS halo tile search failed for h{h:02d}v{v:02d}: {e}")
        return None
    for g in candidates:
        links = g.data_links()
        if links and _parse_tile_hv(links[0].split("/")[-1]) == (h, v):
            return g
    return None


def _axis_slices(delta: int, pad_px: int, tile_len: int):
    """
    For one axis (row or col), given a neighbor offset delta in {-1,0,1},
    return (neighbor_slice, dest_slice):
      neighbor_slice -> the ONLY part of the neighbor's dataset to read
      dest_slice     -> where to place it in our padded array
    This works uniformly for edge neighbors (delta=0 on one axis) and
    corner neighbors (delta != 0 on both axes).
    """
    if delta == -1:
        return slice(tile_len - pad_px, tile_len), slice(0, pad_px)
    elif delta == 1:
        return slice(0, pad_px), slice(pad_px + tile_len, pad_px + tile_len + pad_px)
    else:
        return slice(0, tile_len), slice(pad_px, pad_px + tile_len)


def _padded_tile_geolocation(h, v, tile_rows, tile_cols, pixel_size_x, pixel_size_y, pad_px):
    x_origin, y_origin = _viirs_tile_corner_xy(h, v)
    x0 = x_origin - pad_px * pixel_size_x
    y0 = y_origin + pad_px * pixel_size_y
    x_coords_1d = x0 + (np.arange(tile_cols + 2 * pad_px) + 0.5) * pixel_size_x
    y_coords_1d = y0 - (np.arange(tile_rows + 2 * pad_px) + 0.5) * pixel_size_y
    return x_coords_1d, y_coords_1d


class _Hdf5TileCache:
    def __init__(self, chunk_cache_mb: int = 16):
        self._handles: dict[tuple[int, int], tuple[object, h5py.File]] = {}
        # rdcc_nbytes: enough to hold several 1000x1000 chunks so that a
        # neighbor chunk touched by one domain tile's border stays cached
        # if a different adjacent domain tile's border needs it again,
        # instead of re-downloading it over the network.
        self._rdcc_nbytes = chunk_cache_mb * 1024 * 1024

    def get_h5(self, hv, granule):
        if hv in self._handles:
            return self._handles[hv][1]
        file_obj = earthaccess.open([granule])[0]
        h5f = h5py.File(
            file_obj, "r",
            rdcc_nbytes=self._rdcc_nbytes,
            rdcc_nslots=1009,   # prime number, >> expected distinct chunks per file
        )
        self._handles[hv] = (file_obj, h5f)
        return h5f

    def close_all(self):
        for file_obj, h5f in self._handles.values():
            for closer in (h5f, file_obj):
                try:
                    closer.close()
                except Exception:
                    pass
        self._handles.clear()

def _regrid_viirs(
    lis_grid,
    method: str,
    current_date,
    max_tiles: int = None,
    pad_px: int = 60,
    **kwargs,
) -> dict:
    """
    Memory-bounded VIIRS regridding with tile-seam correction.

    Each domain tile is regridded individually (as in the original
    tile-by-tile accumulate loop), but padded with a THIN border
    (pad_px pixels) pulled from its up-to-8 neighbors via targeted HDF5
    slicing -- never the neighbor's full array. Peak memory is therefore
    bounded by roughly one tile's padded footprint + its pyresample
    KD-tree, not the sum of every domain + halo tile.

    pad_px should comfortably exceed whatever spatial support your chosen
    `method` needs at the tile edge (a few pixels for nearest/bilinear,
    more for EWA). 60 px (~30 km at 500 m) is a generous default.

    NOTE on pad_px clamping: each tile independently clamps its own
    EFFECTIVE padding to that tile's own HDF5 chunk size via
    _resolve_pad_px(). The *requested* pad_px argument is never mutated,
    so a smaller clamp on one tile cannot silently shrink the padding
    used by subsequent tiles in the loop.
    """
    date_str = current_date.strftime("%Y-%m-%d")
    log.info(f"Starting VIIRS processing for {date_str} (requested pad_px={pad_px})...")

    try:
        if hasattr(lis_grid, "get_lonlats"):
            lons, lats = lis_grid.get_lonlats()
            min_lon, max_lon = float(lons.min()), float(lons.max())
            min_lat, max_lat = float(lats.min()), float(lats.max())
        else:
            min_lon = float(lis_grid.lon.min().item())
            max_lon = float(lis_grid.lon.max().item())
            min_lat = float(lis_grid.lat.min().item())
            max_lat = float(lis_grid.lat.max().item())
    except Exception as e:
        log.warning(f"Could not extract bounding box from lis_grid: {e}. Using default region.")
        min_lon, min_lat, max_lon, max_lat = -116.1, 35.32, -89.46, 50.59

    bounding_box = (min_lon, min_lat, max_lon, max_lat)
    short_name = "VJ110A1F"

    log.info(f"Searching Earthdata for {short_name} granules on {date_str} within {bounding_box}...")
    raw_results = earthaccess.search_data(
        short_name=short_name, bounding_box=bounding_box, temporal=(date_str, date_str)
    )
    if not raw_results:
        log.warning(f"No VIIRS granules found for {date_str}. Skipping VIIRS processing.")
        return {}

    lis_area_obj = getattr(lis_grid, "area", lis_grid)

    domain_tiles: set[tuple[int, int]] = set()
    # Sentinel values: key absent = never searched, False = confirmed missing.
    granule_by_hv: dict[tuple[int, int], object] = {}
    skipped = 0
    for granule in raw_results:
        links = granule.data_links()
        if not links:
            continue
        hv = _parse_tile_hv(links[0].split("/")[-1])
        if hv is None:
            continue
        h, v = hv
        if not _viirs_tile_overlaps_area(h, v, lis_area_obj):
            skipped += 1
            continue
        domain_tiles.add(hv)
        granule_by_hv[hv] = granule

    if skipped:
        log.info(f"VIIRS: skipped {skipped} tile(s) outside precise LIS domain AreaDefinition")
    if not domain_tiles:
        log.warning("VIIRS: no tiles overlap the LIS domain — skipping")
        return {}

    # For logging / --max-viirs-tiles only -- halo tiles below are only ever
    # partially read, so this is a bound on API calls, not raw memory.
    required_tiles = _expand_tile_set_with_halo(domain_tiles, radius=1)
    log.info(
        f"VIIRS: {len(domain_tiles)} domain tile(s); up to "
        f"{len(required_tiles) - len(domain_tiles)} neighbor tile(s) may be "
        f"touched for up to {pad_px}px thin-border context (never fully loaded)."
    )
    if max_tiles is not None and len(required_tiles) > max_tiles:
        raise RuntimeError(
            f"VIIRS: {len(required_tiles)} tile(s) (domain + neighbors) would be "
            f"touched but --max-viirs-tiles={max_tiles}. Increase only if this looks "
            "correct for the domain."
        )

    cache = _Hdf5TileCache()
    transformer = pyproj.Transformer.from_crs(_SIN_CRS, "EPSG:4326", always_xy=True)
    target_shape = lis_area_obj.shape
    combined_snow = np.full(target_shape, np.nan, dtype=np.float32)
    # Each LIS cell takes its value only from the tile it lies in. A tile's
    # padding reaches pad_px into its neighbours, so without this, later tiles
    # overwrite cells that belong to earlier ones.
    lis_lon, lis_lat = lis_area_obj.get_lonlats()
    cell_x, cell_y = transformer.transform(lis_lon, lis_lat, direction="INVERSE")

    try:
        sorted_tiles = sorted(domain_tiles)
        for i, (h, v) in enumerate(sorted_tiles):
            try:
                h5f = cache.get_h5((h, v), granule_by_hv[(h, v)])
                ds = h5f[_HDFEOS_DATA_PATH]
                tile_rows, tile_cols = ds.shape

                # Compute THIS tile's effective padding locally -- never
                # reassign the outer `pad_px` parameter, so a clamp here
                # cannot leak into the next tile's iteration.
                effective_pad_px = _resolve_pad_px(ds, pad_px)
                if effective_pad_px != pad_px:
                    log.info(
                        f"  -> Tile h{h:02d}v{v:02d}: using effective_pad_px="
                        f"{effective_pad_px} (requested {pad_px}) for this tile only."
                    )

                x_coords = h5f[_HDFEOS_XDIM_PATH][:]
                y_coords = h5f[_HDFEOS_YDIM_PATH][:]
                pixel_size_x = float(abs(x_coords[1] - x_coords[0]))
                pixel_size_y = float(abs(y_coords[1] - y_coords[0]))

                own_data = ds[:].astype(np.float32)   # full read -- unavoidable, it's our output
                own_data[own_data > 100] = np.nan

                if np.all(np.isnan(own_data)):
                    log.info(f"  -> Tile h{h:02d}v{v:02d} is entirely NaN, skipping.")
                    del own_data
                    continue

                padded = np.full(
                    (tile_rows + 2 * effective_pad_px, tile_cols + 2 * effective_pad_px),
                    np.nan, dtype=np.float32,
                )
                padded[
                    effective_pad_px:effective_pad_px + tile_rows,
                    effective_pad_px:effective_pad_px + tile_cols,
                ] = own_data
                del own_data  # free the un-padded copy right away

                for dh in (-1, 0, 1):
                    for dv in (-1, 0, 1):
                        if dh == 0 and dv == 0:
                            continue
                        hh = (h + dh) % (_VIIRS_H_MAX + 1)
                        vv = v + dv
                        if vv < 0 or vv > _VIIRS_V_MAX:
                            continue

                        neighbor_granule = granule_by_hv.get((hh, vv), "unsearched")
                        if neighbor_granule == "unsearched":
                            neighbor_granule = _search_tile_granule(short_name, hh, vv, date_str)
                            granule_by_hv[(hh, vv)] = neighbor_granule  # cache hit OR miss
                        if not neighbor_granule:
                            continue  # confirmed no data -- border stays NaN

                        try:
                            n_h5 = cache.get_h5((hh, vv), neighbor_granule)
                            n_ds = n_h5[_HDFEOS_DATA_PATH]
                            row_n, row_d = _axis_slices(dv, effective_pad_px, tile_rows)
                            col_n, col_d = _axis_slices(dh, effective_pad_px, tile_cols)

                            # *** The ONLY part of the neighbor ever read ***
                            border = n_ds[row_n, col_n].astype(np.float32)
                            border[border > 100] = np.nan
                            padded[row_d, col_d] = border
                            del border
                        except Exception as e:
                            log.debug(f"VIIRS: border read failed h{hh:02d}v{vv:02d}: {e}")

                x_coords_1d, y_coords_1d = _padded_tile_geolocation(
                    h, v, tile_rows, tile_cols, pixel_size_x, pixel_size_y, effective_pad_px
                )
                x2d, y2d = np.meshgrid(x_coords_1d, y_coords_1d)
                lon2d, lat2d = transformer.transform(x2d, y2d)
                del x2d, y2d, x_coords_1d, y_coords_1d

                source_def = SwathDefinition(
                    lons=xr.DataArray(lon2d, dims=["y", "x"]),
                    lats=xr.DataArray(lat2d, dims=["y", "x"]),
                )
                del lon2d, lat2d
                da_tile = xr.DataArray(padded, dims=["y", "x"])
                da_tile.attrs["area"] = source_def

                log.info(
                    f"  -> Regridding tile {i+1}/{len(sorted_tiles)} "
                    f"(h{h:02d}v{v:02d}, pad={effective_pad_px}px)..."
                )
                rg_tile = regrid(da_tile, source_def, lis_area_obj, method=method)

                tile_result = rg_tile.values
                x_min, y_max = _viirs_tile_corner_xy(h, v)
                in_tile = ((cell_x >= x_min) & (cell_x < x_min + _MODIS_SIN_TILE_SIZE_M)
                           & (cell_y <= y_max) & (cell_y > y_max - _MODIS_SIN_TILE_SIZE_M))
                valid_mask = ~np.isnan(tile_result) & in_tile
                combined_snow[valid_mask] = tile_result[valid_mask]

                del padded, da_tile, rg_tile, tile_result, valid_mask, source_def
            except Exception as e:
                log.warning(f"Could not process VIIRS tile h{h:02d}v{v:02d}: {e}")
            finally:
                gc.collect()
    finally:
        cache.close_all()

    if np.all(np.isnan(combined_snow)):
        log.warning("VIIRS: no valid data produced for any tile — skipping")
        return {}

    result = {
        "viirs_cgf_ndsi_snow_cover": xr.DataArray(
            combined_snow,
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "VIIRS Cloud Gap Filled NDSI Snow Cover",
                "units": "percentage",
                "source": "NASA Earthdata (VJ110A1F via earthaccess)",
                "note": (
                    f"Regridded tile-by-tile; each tile padded with up to "
                    f"{pad_px}px (clamped per-tile to its own HDF5 chunk size, "
                    "never below the next tile's full requested padding) "
                    "pulled from neighboring tiles via chunk-level HDF5 reads "
                    "(full neighbor tiles never loaded) to avoid seams at tile "
                    "boundaries and at the LIS domain edge, bounding peak memory "
                    "to roughly one tile at a time."
                ),
            },
        )
    }
    log.info("VIIRS tile-by-tile streaming + thin-border regridding completed successfully.")
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
        # FORCE creation of a fresh filesystem object using the latest os.environ keys
        # Do not reuse the one passed in `fs` to prevent ExpiredToken errors on long jobs
        fresh_fs = make_fs() 
        
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            ds_out.to_netcdf(tmp_path, engine="h5netcdf", encoding=encoding)
            with open(tmp_path, "rb") as f:
                data = f.read()
            with fresh_fs.open(output_path, "wb") as fout:
                fout.write(data)
        finally:
            os.unlink(tmp_path)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(output_path, engine="h5netcdf", encoding=encoding)

def verify_area_matches_grid(lis_area: AreaDefinition, lis_grid: xr.Dataset, atol: float = 1e-3) -> None:
    """
    Sanity-check that the reconstructed pyresample AreaDefinition produces
    the same per-pixel lat/lon as the LIS file's own stored lat/lon arrays.

    AMSR2/CEDA are regridded onto lis_grid's REAL lat/lon arrays (ground
    truth from the file). VIIRS/ICESat-2 are regridded onto lis_area (a
    RECONSTRUCTED geometry). Both ultimately get written under the SAME
    lat/lon coordinates in the combined output file. If these two
    representations disagree even slightly, VIIRS/ICESat-2 data will be
    subtly mis-colocated relative to AMSR2/CEDA without any obvious error.
    """
    area_lons, area_lats = lis_area.get_lonlats()
    file_lons = lis_grid["lon"].values
    file_lats = lis_grid["lat"].values

    if area_lons.shape != file_lons.shape:
        raise ValueError(
            f"lis_area shape {area_lons.shape} != lis_grid shape {file_lons.shape} "
            "-- these two grid representations are not even compatible in size."
        )

    lon_diff = np.abs(area_lons - file_lons)
    lat_diff = np.abs(area_lats - file_lats)
    log.info(
        f"LIS grid cross-check: max |lon diff|={lon_diff.max():.6f}deg, "
        f"max |lat diff|={lat_diff.max():.6f}deg (tolerance={atol}deg)"
    )
    if lon_diff.max() > atol or lat_diff.max() > atol:
        log.error(
            "LIS AreaDefinition (used for VIIRS/ICESat-2) does NOT match the "
            "LIS file's own lat/lon arrays (used for AMSR2/CEDA) within "
            f"{atol}deg. VIIRS/ICESat-2 outputs may be mis-colocated relative "
            "to AMSR2/CEDA in the combined output. Check earth-model/corner/"
            "orientation assumptions in build_lis_area_definition()."
        )

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
    parser.add_argument("--viirs-pad-px", type=int, default=60, metavar="N",
                    help="Width (in native VIIRS pixels) of the neighbor-tile border "
                         "pulled in around each domain tile to prevent seams, without "
                         "loading full neighbor tiles. 60px (~30km @ 500m) is generous "
                         "for nearest/bilinear; increase for wide-kernel methods like EWA.")   
    ns = parser.parse_args()

    # ---------------------------------------------------------
    # [NEW] Authenticate with NASA Earthdata Login
    # ---------------------------------------------------------
    log.info("Authenticating with NASA Earthdata...")
    try:
        earthaccess.login(persist=True)
        log.info("Earthdata authentication successful.")
    except Exception as e:
        log.error(f"Earthdata login failed. Cannot access VIIRS data: {e}")
        return

    # ---------------------------------------------------------
    # JAXA G-Portal SFTP authentication -- only needed when AMSR2 is requested
    # ---------------------------------------------------------
    sftp_fs = None
    if ns.amsr2_dir is not None:
        print("=== JAXA G-Portal Authentication ===")
        sftp_host = "sftp.gportal.jaxa.jp"
        sftp_user = input("Enter JAXA G-Portal Username: ").strip()
        if not sftp_user:
            print("Error: Username cannot be empty.")
            return
        private_key = input("Private Key Path [/home/jovyan/privatekey.key]: ").strip() or "/home/jovyan/privatekey.key"
        if not os.path.exists(private_key):
            print(f"Error: Private key file not found at {private_key}")
            return
        print("Connecting to SFTP using private key...")
        try:
            sftp_fs = fsspec.filesystem(
                "sftp", host=sftp_host, username=sftp_user,
                key_filename=private_key, look_for_keys=False,
            )
            sftp_fs.ls("/")
            print("SFTP Connection Successful.")
        except Exception as e:
            print(f"Failed to connect to SFTP: {e}")
            return

    # Build a shared fsspec store if any S3 paths are present
    any_s3 = any(
        _is_s3(str(p))
        for p in [ns.lis_path, ns.amsr2_dir, ns.ceda_dir, ns.viirs_dir]
        if p is not None
    )
    s3_fs = make_fs() if any_s3 else None

    log.info("Loading LIS grid and AreaDefinition...")
    lis_grid = load_lis_grid(ns.lis_path, fs=s3_fs)    
    lis_area = build_lis_area_definition(ns.lis_path, fs=s3_fs, cache_dir=ns.weights_dir, overwrite=ns.overwrite_weights)
    verify_area_matches_grid(lis_area, lis_grid) 
    # Generate a list of daily dates
    dates = pd.date_range(start=ns.start_date, end=ns.end_date, freq='D')

    # ---------------------------------------------------------
    # Pre-flight check: is the SlideRule API server reachable?
    # This also performs the one-time client initialization --
    # do not call icesat2.init() again later in the script.
    # ---------------------------------------------------------
    icesat2_enabled = _check_sliderule_availability("slideruleearth.io", verbose=True)
    if not icesat2_enabled:
        print("\nWARNING: The SlideRule API server version check failed -- it does not appear to be reachable.")
        print("ICESat-2 ATL06 snow-depth/height data cannot be fetched without it.")
        answer = input("Continue running the workflow WITHOUT ICESat-2 data for all dates? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            log.error("Aborting workflow at user request (SlideRule API unavailable).")
            return
        log.warning("Proceeding WITHOUT ICESat-2/SlideRule data for the entire date range.")
    
    # -------------------------------------------------------------------------
    # MAIN TIME LOOP
    # -------------------------------------------------------------------------
    for current_date in dates:
        # 1. Refresh global AWS credentials (boto3/os.environ)
        refresh_aws_credentials()
        
        # 2. Re-instantiate the fsspec FileSystem so it picks up the fresh os.environ variables
        if any_s3:
            s3_fs = make_fs() 
            
        # Use %Y%m%d to get '20190101' instead of '2019-01-01'
        date_str = current_date.strftime('%Y%m%d') 
        log.info(f"--- Processing date: {date_str} ---")

        # =====================================================================
        # Pre-populate data_vars with dummy variables to satisfy LIS assimilation
        # =====================================================================
        data_vars: dict[str, xr.DataArray] = {}
        found_any_data = False # Track if AT LEAST ONE source had valid data today
        
        lis_shape = (lis_grid.sizes["north_south"], lis_grid.sizes["east_west"])
        def get_dummy(name, units):
            """Returns an empty DataArray filled with -9999.0 for the LIS grid."""
            return xr.DataArray(
                np.full(lis_shape, -9999.0, dtype=np.float32),
                dims=["north_south", "east_west"],
                attrs={"long_name": f"{name} (No Data)", "units": units, "source": "Missing"}
            )

        # 1. Initialize Default Schema Dummies 
        if ns.amsr2_dir is not None:
            data_vars["amsr2_snow_depth_mean"] = get_dummy("AMSR2 snow depth mean", "mm")
            data_vars["amsr2_snow_depth_std"] = get_dummy("AMSR2 snow depth standard deviation", "mm")
            
        if ns.ceda_dir is not None:
            data_vars["ceda_swe"] = get_dummy("CEDA ESA CCI snow water equivalent", "mm")
            data_vars["ceda_swe_std"] = get_dummy("CEDA ESA CCI SWE standard deviation", "mm")
            
        if ns.viirs_dir is not None:
            data_vars["viirs_cgf_ndsi_snow_cover"] = get_dummy("VIIRS Cloud Gap Filled NDSI Snow Cover", "percentage")
            
        # ICESat-2 (SlideRule virtual fetch is always attempted in this script)
        data_vars["icesat2_snow_depth"] = get_dummy("ICESat-2 estimated snow depth", "meters")
        data_vars["icesat2_h_mean"] = get_dummy("ICESat-2 ATL06 land-ice surface height", "meters")
        data_vars["icesat2_3dep_dem_10m"] = get_dummy("ICESat-2 Sliderule 3DEP DEM 10M", "meters")

         # =====================================================================
        # Attempt to fetch and regrid real data
        # =====================================================================
        if ns.amsr2_dir is not None:
            amsr2_res = _regrid_amsr2(
                input_dir=ns.amsr2_dir,
                lis_grid=lis_grid,
                method=ns.amsr2_method,
                weights_dir=ns.weights_dir,
                current_date=current_date,
                overwrite_weights=ns.overwrite_weights,
                fs=sftp_fs,
            )
            #amsr2_res = _regrid_amsr2(ns.amsr2_dir, lis_grid, ns.amsr2_method, ns.weights_dir, current_date, overwrite_weights=ns.overwrite_weights, fs=fs)
            if amsr2_res:
                data_vars.update(amsr2_res)
                found_any_data = True
    
        if ns.ceda_dir is not None:
            ceda_res = _regrid_ceda(ns.ceda_dir, lis_grid, ns.ceda_method, ns.weights_dir, current_date, overwrite_weights=ns.overwrite_weights, fs=s3_fs)
            if ceda_res:
                data_vars.update(ceda_res)
                found_any_data = True
    
        if ns.viirs_dir is not None:
            viirs_res = _regrid_viirs(
                lis_area, ns.viirs_method, current_date,
                max_tiles=ns.max_viirs_tiles,
                pad_px=ns.viirs_pad_px,
            )
            if viirs_res:
                data_vars.update(viirs_res)
                found_any_data = True
    
        # =========================================================
        # VIRTUAL READ FROM SLIDERULE API (Replacing Parquet Read)
        # =========================================================
        try:
            if icesat2_enabled:
                try:
                    log.info(f"Fetching ICESat-2 data virtually from SlideRule API for {date_str}...")
                    ...
                except Exception as e:
                    log.error(f"Failed to virtually fetch or process SlideRule data for {date_str}: {e}")
            else:
                log.info(f"Skipping ICESat-2/SlideRule fetch for {date_str} (server unavailable, user chose to continue).")
            
            # 1. Define LIS domain polygon dynamically from your LIS grid
            lat_min, lat_max = float(lis_grid.lat.min()), float(lis_grid.lat.max())
            lon_min, lon_max = float(lis_grid.lon.min()), float(lis_grid.lon.max())
            
            polygon = [
                {"lon": lon_min, "lat": lat_min},
                {"lon": lon_max, "lat": lat_min},
                {"lon": lon_max, "lat": lat_max},
                {"lon": lon_min, "lat": lat_max},
                {"lon": lon_min, "lat": lat_min},
            ]
            
            # 2. Set daily time bounds for SlideRule matching fetch_atl06_mosaic-r1
            t0 = current_date.strftime("%Y-%m-%dT00:00:00Z")
            t1 = current_date.strftime("%Y-%m-%dT23:59:59Z")
            
            # 3. SlideRule parameter schema mapped exactly from fetch script
            parms = {
                "poly": polygon,
                "t0": t0,
                "t1": t1,
                "atl08_class": "atl08_ground",
                "srt": icesat2.SRT_LAND,
                "cnf": 4,
                "len": 40,
                "res": 20,
                "samples": {
                    "mosaic": {
                        "asset": "usgs3dep-10meter-dem",
                        "radius": 10.0,
                        "zonal_stats": True
                    }
                }
            }
            
            # 4. Fetch virtually into memory
            icesat2_df = icesat2.atl06p(parms)
            
            if len(icesat2_df) > 0:
                log.info(f"SlideRule returned {len(icesat2_df)} observations for {date_str}.")
                
                icesat2_df['lon'] = icesat2_df.geometry.x
                icesat2_df['lat'] = icesat2_df.geometry.y
                
                icesat2_vars = _regrid_icesat2(
                    icesat2_df,
                    lis_area,
                    current_date,
                    source_path="SlideRule API (Virtual)"
                )
                if icesat2_vars:
                    data_vars.update(icesat2_vars)
                    found_any_data = True
            else:
                log.info(f"ICESat-2: No SlideRule observations found for {date_str}.")
                
        except Exception as e:
            log.error(f"Failed to virtually fetch or process SlideRule data for {date_str}: {e}")
    
        # Check if we actually found ANY real data for today
        if not found_any_data:
            log.warning(f"No valid data found from any source for {date_str}. Skipping file generation.")
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

# Build the dynamic output filename (Handles both Local and S3 paths safely)
        if _is_s3(ns.output_path):
            if ns.output_path.endswith('.nc'):
                # Extract filename vs path for S3
                base, name = ns.output_path.rsplit('/', 1)
                stem, suffix = name.rsplit('.', 1)
                daily_out_path = f"{base}/{stem}_{date_str}.{suffix}"
            else:
                daily_out_path = f"{ns.output_path.rstrip('/')}/swe_combined_{date_str}.nc"
        else:
            # Handle Local File System Path
            out_path_obj = Path(ns.output_path)
            if out_path_obj.suffix == '.nc':
                daily_out_path = str(out_path_obj.parent / f"{out_path_obj.stem}_{date_str}{out_path_obj.suffix}")
            else:
                out_path_obj.mkdir(parents=True, exist_ok=True)
                daily_out_path = str(out_path_obj / f"swe_combined_{date_str}.nc")

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
        
        # Use the S3-aware _write_output helper function defined earlier in the script!
        _write_output(ds_day, daily_out_path, encoding, fs=s3_fs if _is_s3(ns.output_path) else None)
        
        # Explicitly close and clean up to prevent memory spikes
        ds_day.close()
        del ds_day
        
        log.info(f"Finished {date_str} successfully!")

    log.info("Workflow complete. All valid daily files have been generated.")

if __name__ == "__main__":
    main()