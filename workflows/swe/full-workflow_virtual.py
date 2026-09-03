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
        --icesat2-parquet /data/icesat2/atl06.parquet \\
        --output-path /data/swe_combined.nc

Usage example (S3)
──────────────────
    python full-workflow.py \\
        --lis-path s3://my-bucket/lis_input_NMP_1000m_missouri.nc \\
        --amsr2-dir s3://my-bucket/amsr2 \\
        --ceda-dir  s3://my-bucket/ceda \\
        --viirs-dir s3://my-bucket/viirs \\
        --icesat2-parquet s3://my-bucket/icesat2/atl06.parquet \\
        --output-path /data/swe_combined.nc

"""



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

import icechunk as ic
from icechunk import Repository, s3_storage
import xcdat as xc

import obstore

# Use boto3 to grab the working credentials
session = boto3.Session()
creds = session.get_credentials()

if creds:
    frozen_creds = creds.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen_creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen_creds.secret_key
    if frozen_creds.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen_creds.token
        
    # Prevent obstore from trying to hit the EC2 metadata endpoint
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    
    print("Successfully injected boto3 credentials into environment for obstore.")
else:
    print("Warning: boto3 could not find AWS credentials.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    force=True
)
log = logging.getLogger(__name__)

AMSR2_GLOB = "**/*.h5"
CEDA_GLOB = "**/*.nc"
VIIRS_GLOB = "**/*.h5"


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

def list_s3(store, prefix):
    # Dummy helper matching user's original logic or obstore listing
    return [meta['path'] for meta in store.list(prefix=prefix)]
    
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
    """Regrid AMSR2 dataset using icechunk and return {var_name: DataArray}."""

    # 1. Initialize icechunk store
    log.info("Initializing icechunk repository...")
    jaxa_gportal_url = "https://gportal.jaxa.jp/"
    
    repo = Repository.open(
        s3_storage(
            bucket="airborne-smce-prod-user-bucket",
            prefix="JOIN/icechunk-stores/GCOM-W1-AMSR2-L3-SND"
        ),
        
    )

    # 2. Open a read-only session for the main branch
    session = repo.readonly_session("main")

    # 3. Read dataset
    log.info("Opening Zarr store via icechunk session...")
    ds = xr.open_zarr(session.store, mask_and_scale=False)

    # 4. Standardize the longitude automatically using xcdat
    log.info("Standardizing longitude to [-180, 180] using xcdat...")
    ds = xc.swap_lon_axis(ds, to=(-180, 180))

    # Extract target variable (assuming "Geophysical Data" based on your original file)
    da = ds["Geophysical Data"]

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
            raise ValueError("Could not find lat/lon in LIS grid!")

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
            attrs={"long_name": long_name, "units": units, "source": "icechunk/gcom-w1-amsr2-l3-snd"},
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

def _regrid_ceda(
    input_dir: str,
    lis_grid: xr.Dataset,
    method: str,
    weights_dir: str,    
    current_date: pd.Timestamp,
    overwrite_weights: bool = False,    
    fs=None,
) -> dict[str, xr.DataArray]:
    """Regrid the CEDA icechunk virtual store and return {var_name: DataArray}."""
    
    bucket = "airborne-smce-prod-user-bucket"
    region = "us-west-2"
    src_bucket_url = f"s3://{bucket}"
    
    log.info("Configuring S3 Object Store...")
    # Initialize object store and registry
    src_store = obstore.store.S3Store(bucket=bucket, config=obstore.store.s3_store_config())
    registry = obstore.store.ObjectStoreRegistry({src_bucket_url: src_store})

    # Get list of CEDA files
    log.info(f"Listing CEDA files from {src_bucket_url}/JOIN/CEDA ...")
    ceda_files = sorted(f for f in list_s3(src_store, "JOIN/CEDA") if f.endswith("nc"))

    if not ceda_files:
        log.warning("No CEDA files found in S3 store — skipping")
        return {}

    # Build virtual zarr store of all the CEDA data
    parser = HDFParser()
    ceda_urls = [f"{src_bucket_url}/{f}" for f in ceda_files]
    vds_all = open_virtual_mfdataset(ceda_urls, parser=parser, registry=registry)

    # Create / Open icechunk store
    dst_prefix = "JOIN/icechunk-stores/CEDA"
    dst_store = ic.s3_storage(
        bucket=bucket,
        prefix=dst_prefix,
        region=region,
        from_env=True
    )

    dst_prefix_url = f"{src_bucket_url}/JOIN/CEDA/"
    config = ic.RepositoryConfig.default()
    config.set_virtual_chunk_container(ic.VirtualChunkContainer(
        dst_prefix_url,
        ic.storage.s3_store(region=region),
        name="ceda-s3"
    ))
    credentials = ic.credentials.containers_credentials(
        {dst_prefix_url: ic.credentials.s3_credentials(from_env=True)}
    )

    # Open or create the icechunk repository
    log.info("Opening/Creating Icechunk Repository for CEDA...")
    repo = ic.Repository.open_or_create(dst_store, config, credentials)
    repo.save_config()

    # Create writable session and commit virtualization
    session = repo.writable_session("main")
    vds_all.vz.to_icechunk(session.store)
    commit_id = session.commit("Virtualize CEDA data in airborne-smce-prod-user-bucket")
    log.info(f"Committed icechunk store: {commit_id}")

    # Open read-only session for querying
    session_ro = repo.readonly_session("main")
    ds_all = xr.open_zarr(session_ro.store)

    # Select the specific date requested for the pipeline
    try:
        # Assuming the dataset has a 'time' dimension we can select against
        ds = ds_all.sel(time=current_date, method="nearest")
    except KeyError:
        log.warning("Time dimension missing or differs in CEDA store, falling back to full store")
        ds = ds_all

    log.info("CEDA dataset loaded via virtualized icechunk store.")

    # ---------------------------------------------------------
    # Regridding Logic
    # ---------------------------------------------------------
    lat_vals = ds["lat"].values if "lat" in ds else ds["y"].values
    lon_vals = ds["lon"].values if "lon" in ds else ds["x"].values
    if lat_vals.ndim == 2:
        lat_vals = lat_vals[:, 0]
    if lon_vals.ndim == 2:
        lon_vals = lon_vals[0, :]
        
    source_grid = xr.Dataset(
        coords={"lat": np.sort(np.unique(lat_vals)), "lon": np.sort(np.unique(lon_vals))}
    )

    from join_scratch.regrid.regular_to_regular import compute_weights, load_regridder
    
    weights_local_dir = _ensure_local_dir(weights_dir)
    weights_path = weights_local_dir / f"ceda-lis-weights-{method}.nc"
    
    compute_weights(source_grid, lis_grid, weights_path, method=method, overwrite=overwrite_weights)
    regridder = load_regridder(source_grid, lis_grid, weights_path, method=method)

    # Restore lat/lon as dim names for xESMF if necessary
    if "y" in ds.dims and "x" in ds.dims:
        ds_xesmf = ds.swap_dims({"y": "lat", "x": "lon"})
    else:
        ds_xesmf = ds

    log.info("CEDA: regridding swe and swe_std …")
    rg = regridder(ds_xesmf)

    def _da(arr, long_name, units):
        return xr.DataArray(
            arr.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={"long_name": long_name, "units": units, "source": "icechunk/ceda-virtualized"},
        )

    return {
        "ceda_swe": _da(rg["swe"], "CEDA ESA CCI snow water equivalent", "mm"),
        "ceda_swe_std": _da(rg["swe_std"], "CEDA ESA CCI SWE standard deviation", "mm"),
    }

# def _viirs_tile_bbox(h: int, v: int) -> tuple[float, float, float, float]:
#     """Return (lon_min, lat_min, lon_max, lat_max) for a MODIS/VIIRS h/v tile.

#     The MODIS sinusoidal tile grid divides the globe into 36 × 18 tiles.
#     Each tile spans exactly 10° of latitude and ~10° equivalent in the
#     sinusoidal projection.  Tile (h=0, v=0) is the top-left (north-west) tile.
#     """
#     lat_max = 90.0 - v * 10.0
#     lat_min = lat_max - 10.0
#     # Longitude extent depends on latitude; use the wider of top/bottom edge
#     # sin-projection x-extent per tile is (2*pi*R/36) metres, but for bbox
#     # purposes we just use the geographic span at each latitude edge.
#     import math
#     def _lon_half_width(lat_deg):
#         lat_r = math.radians(abs(lat_deg))
#         cos_lat = math.cos(lat_r) if lat_r < math.pi / 2 else 1e-9
#         return 10.0 / cos_lat  # degrees longitude for 10° equivalent arc
#     hw = max(_lon_half_width(lat_min), _lon_half_width(lat_max))
#     lon_centre = -180.0 + (h + 0.5) * (360.0 / 36.0)
#     return lon_centre - hw, lat_min, lon_centre + hw, lat_max

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
    input_dir: str,
    lis_area,
    method: str,
    current_date: pd.Timestamp,
    fs=None,    
    max_tiles: int | None = None,
) -> dict[str, xr.DataArray]:
    """Regrid the first VIIRS date group found and return {var_name: DataArray}."""
    from pyresample.geometry import SwathDefinition

    # Format the date to match how it appears in your filenames
    # Example format: "20190101"
    date_str = current_date.strftime("%Y%j") 
    path_str = current_date.strftime("%Y/%m/%d/")
    
    input_dir = f"{input_dir}VJ110A1F/{path_str}"
    all_files = _list_files(input_dir, ".h5", fs=fs)
    # Remove duplicates from the entire directory search
    all_files = list(dict.fromkeys(all_files))
    
    if not all_files:
        log.warning("No VIIRS files found under %s — skipping", input_dir)
        return {}
    # Filter for files that belong to this specific date
    files = [f for f in all_files if date_str in f]
    
    date_groups: dict[str, list[str]] = defaultdict(list)
    for p in files:
        stem = _path_name(p).rsplit(".", 1)[0] if "." in _path_name(p) else _path_name(p)
        parts = stem.split(".")
        date_key = parts[1] if len(parts) > 1 else stem
        date_groups[date_key].append(p)

    date_key, paths = next(iter(sorted(date_groups.items())))

    # Filter to tiles that spatially overlap the LIS domain
    filtered = []
    skipped = 0
    for p in paths:
        fname = _path_name(p)
        m = __import__("re").search(r"\.h(\d{2})v(\d{2})\.", fname)
        if m is not None:
            h, v = int(m.group(1)), int(m.group(2))
            if not _viirs_tile_overlaps_area(h, v, lis_area):
                skipped += 1
                continue
        filtered.append(p)
    if skipped:
        log.info("VIIRS: skipped %d tile(s) outside LIS domain bbox", skipped)
    paths = filtered
    if not paths:
        log.warning("VIIRS: no tiles overlap the LIS domain — skipping")
        return {}
    if max_tiles is not None and len(paths) > max_tiles:
        raise RuntimeError(
            f"VIIRS: {len(paths)} tile(s) would be loaded but --max-viirs-tiles={max_tiles}. "
            "Aborting to prevent OOM. Either the spatial filter is not working correctly "
            "or the domain is unusually large. Increase --max-viirs-tiles only if expected."
        )
    log.info("VIIRS: using date %s (%d tile(s))", date_key, len(paths))

    all_data, all_lons, all_lats = [], [], []
    for path in paths:
        if _is_s3(path):
            handler = handler_from_s3(ViirsFileHandler, path, fs=fs)
        else:
            handler = ViirsFileHandler.from_path(path)
        da = handler.get_dataset()
        swath_def = da.attrs["area"]
        all_data.append(da.values)
        all_lons.append(swath_def.lons.values)
        all_lats.append(swath_def.lats.values)

    composite_data = np.concatenate(all_data, axis=0)
    composite_lons = np.concatenate(all_lons, axis=0)
    composite_lats = np.concatenate(all_lats, axis=0)

    lons_da = xr.DataArray(composite_lons, dims=["y", "x"])
    lats_da = xr.DataArray(composite_lats, dims=["y", "x"])
    source_def = SwathDefinition(lons=lons_da, lats=lats_da)
    composite_da = xr.DataArray(composite_data, dims=["y", "x"])

    log.info("VIIRS: regridding with method=%s …", method)
    rg = regrid(composite_da, source_def, lis_area, method=method)

    return {
        "viirs_cgf_ndsi_snow_cover": xr.DataArray(
            rg.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "VIIRS CGF NDSI snow cover",
                "units": "1",
                "source": f"date_key={date_key}",
            },
        )
    }

# def _regrid_icesat2(
#     parquet_path: str,
#     lis_area,
#     lis_grid: xr.Dataset,
#     fs=None,
# ) -> dict[str, xr.DataArray]:
#     """Regrid ICESat-2 ATL06 point cloud and return {var_name: DataArray}.

#     For S3 URIs, the path is passed directly to geopandas.read_parquet which
#     delegates to pyarrow's native S3 support (no FSFile wrapper needed).
#     For local paths, existence is checked before proceeding.
#     """
#     if not _is_s3(parquet_path) and not Path(parquet_path).exists():
#         log.warning("ICESat-2 Parquet not found at %s — skipping", parquet_path)
#         return {}
#     # Icesat2FileHandler.get_dataset calls geopandas.read_parquet(str(self.filename))
#     # which supports s3:// URIs natively via pyarrow, so from_path without fs works.
#     handler = Icesat2FileHandler.from_path(parquet_path)

#     log.info("ICESat-2: loading from %s", parquet_path)
#     da = handler.get_dataset()
#     source_area = da.attrs["area"]
#     log.info("ICESat-2: regridding %d observations using mean …", len(da))
#     rg = regrid(da, source_area, lis_area, method="mean")
#     return {
#         "icesat2_h_li": xr.DataArray(
#             rg.values.astype(np.float32),
#             dims=["north_south", "east_west"],
#             attrs={
#                 "long_name": "ICESat-2 ATL06 land-ice surface height (mean per LIS pixel)",
#                 "units": "meters",
#                 "source": _path_name(parquet_path),
#             },
#         )
#     }

# def _regrid_icesat2(
#     # parquet_path: str,
#     da_full: xr.DataArray,
#     lis_area,
#     lis_grid: xr.Dataset,
#     current_date: pd.Timestamp,
#     source_path: str,
#     fs=None,
# ) -> dict[str, xr.DataArray]:
#     """Regrid ICESat-2 ATL06 data (attempts to compute h_li - 3DEP DEM)."""
#     import xarray as xr
#     import numpy as np
#     from pathlib import Path

#     print(da_full)

#     date_str = current_date.strftime('%Y-%m-%d')
#     start_of_day = f"{date_str} 00:00:00"
#     end_of_day = f"{date_str} 23:59:59"
    
#     # Slice the dataset for the entire 24-hour period of the current date
#     try:
#         da = da_full.sel(time=slice(start_of_day, end_of_day))
#     except KeyError:
#         log.warning(f"ICESat-2: 'time' dimension missing or invalid.")
#         return {}
        
#     # If the slice is empty, skip
#     if da.size == 0:
#         log.info(f"ICESat-2: No data points found for {date_str}")
#         return {}

#     log.info(f"ICESat-2: Found {da.size} points for {date_str}")

#     source_area = da.attrs["area"]
#     lons = source_area.lons.values
#     lats = source_area.lats.values
#     h_li = da.values

#     # --- 1. FILTER POINTS TO LIS BOUNDING BOX ---
#     import math as _math
#     corners = lis_area.outer_boundary_corners
#     corner_lons = [_math.degrees(c.lon) for c in corners]
#     corner_lats = [_math.degrees(c.lat) for c in corners]
    
#     lon_min, lon_max = min(corner_lons) - 0.1, max(corner_lons) + 0.1
#     lat_min, lat_max = min(corner_lats) - 0.1, max(corner_lats) + 0.1
    
#     log.info("Filtering ICESat-2 points to LIS domain: Lon [%.2f, %.2f], Lat [%.2f, %.2f]", lon_min, lon_max, lat_min, lat_max)
    
#     domain_mask = (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)
#     points_in_domain = np.sum(domain_mask)
    
#     log.info("Found %d of %d points inside LIS domain", points_in_domain, len(lons))
    
#     # Start with h_li as our baseline
#     final_values = h_li.copy()

#     # --- 2. ATTEMPT SLIDERULE 3DEP DEM EXTRACTION ---
#     sliderule_success = False
#     if points_in_domain > 0:
#         log.info("Initializing SlideRule for 3DEP DEM extraction...")
#         from sliderule import icesat2, raster
#         import time
#         import pandas as pd
        
#         try:
#             icesat2.init("slideruleearth.io", verbose=False)
            
#             # 1. Vectorized filtering and coordinate generation
#             filtered_lons = lons[domain_mask]
#             filtered_lats = lats[domain_mask]
            
#             # np.column_stack is significantly faster than zip + list comprehension
#             coords_array = np.column_stack((filtered_lons, filtered_lats))
#             total_points = len(coords_array)
            
#             chunk_size = 50000
            
#             # 2. Pre-allocate NumPy array to avoid dynamic list resizing overhead
#             dem_elevations = np.full(total_points, np.nan, dtype=np.float32)
            
#             log.info("Sampling %d points in chunks of %d using asset 'usgs3dep-10meter-dem'...", total_points, chunk_size)
            
#             for i in range(0, total_points, chunk_size):
#                 end_idx = min(i + chunk_size, total_points)
                
#                 if (i // chunk_size) % 10 == 0:
#                     log.info("  -> Requesting chunk %d to %d...", i, end_idx)
                
#                 # Convert only the current chunk to a nested Python list
#                 chunk = coords_array[i:end_idx].tolist()
                
#                 # Attempt pure request without the buggy 'poly' hint
#                 result = raster.sample("usgs3dep-10meter-dem", chunk)
                
#                 # 3. Streamlined result parsing
#                 vals = []
#                 if isinstance(result, pd.DataFrame) and not result.empty:
#                     # Select the first numeric column directly
#                     num_cols = result.select_dtypes(include=['number'])
#                     if not num_cols.empty:
#                         vals = num_cols.iloc[:, 0].values
#                 elif isinstance(result, list) and result:
#                     if isinstance(result[0], dict):
#                         keys = list(result[0].keys())
#                         vals = [item.get("value", item.get(keys[0], np.nan)) for item in result]
#                     else:
#                         vals = result
                
#                 # Convert extracted values to numpy array and assign to pre-allocated array
#                 vals = np.array(vals, dtype=np.float32) if len(vals) > 0 else np.array([])
                
#                 # Handle size mismatches efficiently
#                 valid_len = min(len(vals), end_idx - i)
#                 if valid_len > 0:
#                     dem_elevations[i : i + valid_len] = vals[:valid_len]
                
#                 time.sleep(0.1)
                
#             # Vectorized masking
#             valid_mask = (dem_elevations > -9000) & (~np.isnan(dem_elevations))
            
#             if np.any(valid_mask): # Faster than np.sum(valid_mask) > 0
#                 valid_count = np.count_nonzero(valid_mask)
#                 log.info("Successfully fetched %d valid DEM elevations!", valid_count)
                
#                 # 4. Map back to the original domain efficiently
#                 # Get the indices where domain_mask is True
#                 domain_indices = np.where(domain_mask)[0] 
                
#                 # Update final_values only where we have valid DEM elevations
#                 valid_domain_indices = domain_indices[valid_mask]
#                 final_values[valid_domain_indices] -= dem_elevations[valid_mask]
                
#                 sliderule_success = True
#             else:
#                 log.warning("SlideRule successfully processed but returned entirely NoData (NaNs) for this region. Falling back to raw h_li.")
                
#         except Exception as e:
#             log.error("SlideRule API Error: %s", e)
#             log.warning("SlideRule failed or is offline. Falling back to raw h_li for ICESat-2.")

#     # --- 3. REGRID TO LIS GRID ---
#     da.values = final_values
    
#     # We only want to regrid the points that actually fall in the domain
#     valid_points_count = points_in_domain
    
#     output_name = "icesat2_snow_depth" if sliderule_success else "icesat2_h_li"
#     long_name = "ICESat-2 ATL06 Snow Depth (h_li - USGS 3DEP)" if sliderule_success else "ICESat-2 ATL06 land-ice surface height"
    
#     log.info("ICESat-2: regridding %d total observations (%d in domain) using mean …", len(da), valid_points_count)
    
#     rg = regrid(da, source_area, lis_area, method="mean")

#     return {
#         output_name: xr.DataArray(
#             rg.values.astype(np.float32),
#             dims=["north_south", "east_west"],
#             attrs={
#                 "long_name": long_name,
#                 "units": "m",
#                 "source": _path_name(parquet_path)
#             },
#         )
#     }

def _regrid_icesat2(
    icesat2_full_da: xr.DataArray,
    lis_area,
    lis_dem: xr.DataArray,
    current_date,  # pd.Timestamp
    source_path: str = None
) -> dict[str, xr.DataArray]:
    """
    Slices the pre-loaded ICESat-2 dataset for the current_date, regrids to the 
    LIS area, and subtracts the pre-computed static 3DEP DEM to compute snow depth.
    """
    log = logging.getLogger(__name__)

    # 1. Filter for the current day
    start_of_day = str(current_date.date())
    end_of_day = str((current_date + pd.Timedelta(days=1)).date())
    
    try:
        da_daily = icesat2_full_da.sel(time=slice(start_of_day, end_of_day))
    except KeyError:
        da_daily = []

    if len(da_daily) == 0:
        log.warning(f"ICESat-2: No data found for {start_of_day}. Skipping.")
        return {}

    # 2. Rebuild SwathDefinition using the SLICED coordinates
    # This ensures the lat/lon arrays exactly match the size of the filtered data
    source_area = SwathDefinition(lons=da_daily.lon, lats=da_daily.lat)
    
    # 3. Regrid the raw h_li (surface height) to the LIS grid
    log.info(f"ICESat-2: Regridding {len(da_daily)} observations for {start_of_day} to LIS grid...")
    rg_h_li = regrid(da_daily, source_area, lis_area, method="mean")
    
    # 4. Subtract the static LIS DEM to compute snow depth
    log.info("ICESat-2: Subtracting static 3DEP DEM to compute snow depth...")
    snow_depth_vals = rg_h_li.values - lis_dem.values
    
    dims = lis_dem.dims
    source_name = source_path if source_path else "ICESat-2 Parquet"

    # 5. Return both the derived snow depth and the raw regridded height
    return {
        "icesat2_snow_depth": xr.DataArray(
            snow_depth_vals.astype(np.float32),
            dims=dims,
            attrs={
                "long_name": "ICESat-2 estimated snow depth (h_li - 3DEP DEM)",
                "units": "meters",
                "source": f"{source_name} + USGS 3DEP 10m DEM"
            },
        ),
        "icesat2_h_li": xr.DataArray(
            rg_h_li.values.astype(np.float32),
            dims=dims,
            attrs={
                "long_name": "ICESat-2 ATL06 land-ice surface height (mean per LIS pixel)",
                "units": "meters",
                "source": source_name
            },
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



# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    
    
    parser = argparse.ArgumentParser(
        description=(
            "Regrid all SWE input datasets to the LIS grid and combine into "
            "a single NetCDF file.  All paths may be local or s3:// URIs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    
    parser.add_argument("--lis-path", required=True,
                        help="Path to the LIS input NetCDF file (local or s3://).")
    parser.add_argument("--amsr2-dir", default=None,
                        help="Directory containing AMSR2 HDF5 files (local or s3://).")
    parser.add_argument("--ceda-dir", default=None,
                        help="Directory containing CEDA ESA CCI SWE NetCDF files (local or s3://).")
    parser.add_argument("--viirs-dir", default=None,
                        help="Directory containing VIIRS CGF snow cover HDF5 files (local or s3://).")
    parser.add_argument("--icesat2-parquet", default=None,
                        help="Path to the cached ICESat-2 ATL06 Parquet file (local or s3://).")
    parser.add_argument("--weights-dir", default="_data/weights",
                        help="Local directory for xESMF weights files.")
    parser.add_argument("--output-path", default="_data/swe_combined.nc",
                        help="Output combined NetCDF path (local or s3://).")
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

    # Build a shared fsspec store if any S3 paths are present
    any_s3 = any(
        _is_s3(str(p))
        for p in [ns.lis_path, ns.amsr2_dir, ns.ceda_dir, ns.viirs_dir, ns.icesat2_parquet]
        if p is not None
    )
    fs = make_fs() if any_s3 else None

    lis_grid = load_lis_grid(ns.lis_path, fs=fs)    
    lis_area = build_lis_area_definition(ns.lis_path, fs=fs, cache_dir=ns.weights_dir, overwrite=ns.overwrite_weights)

     # =========================================================================
    # --- NEW: LOAD PRE-COMPUTED DEM ---
    # =========================================================================
    dem_parquet_path = Path("./_data/dem/lis_dem.parquet")
    if dem_parquet_path.exists():
        log.info("Loading pre-computed LIS DEM from %s", dem_parquet_path)
        import pandas as pd
        dem_df = pd.read_parquet(dem_parquet_path)
        
        # Reshape the 1D DEM column back into the 2D LIS grid shape
        dem_2d = dem_df["dem"].values.reshape(lis_grid.lon.shape)
        
        # Reconstruct the DataArray using the LIS grid's exact dimensions
        lis_dem = xr.DataArray(
            dem_2d,
            dims=lis_grid.lon.dims,  # e.g., ('north_south', 'east_west')
            coords={"lat": lis_grid.lat, "lon": lis_grid.lon},
            attrs={"long_name": "USGS 3DEP 10m DEM", "units": "meters"}
        )
    else:
        log.error("DEM Parquet not found at %s. Please run get_lis_dem.py first!", dem_parquet_path)
        raise FileNotFoundError(f"Missing {dem_parquet_path}")
    # =========================================================================

    # Pre-load ICESat-2 data (if provided) to avoid reloading it on every date loop    
    icesat2_full_da = None
    if ns.icesat2_parquet is not None:
        log.info(f"Pre-loading ICESat-2 data from {ns.icesat2_parquet}...")
        if not _is_s3(ns.icesat2_parquet) and not Path(ns.icesat2_parquet).exists():
            log.warning(f"ICESat-2 Parquet not found at {ns.icesat2_parquet} — skipping ICESat-2.")
        else:
            handler = Icesat2FileHandler.from_path(ns.icesat2_parquet)
            icesat2_full_da = handler.get_dataset()

    # Generate a list of daily dates
    dates = pd.date_range(start=ns.start_date, end=ns.end_date, freq='D')
    
    all_daily_datasets = []

    for current_date in dates:
        log.info(f"--- Processing date: {current_date.strftime('%Y-%m-%d')} ---")    
    
        data_vars: dict[str, xr.DataArray] = {}

        try:
            log.info("Starting AMSR2 Icechunk Regridding...")
            amsr2_vars = _regrid_amsr2(
                input_dir=None,  # No longer needed, configured internally 
                lis_grid=lis_grid,
                method="bilinear",
                weights_dir=ns.weights_dir,
                current_date=current_date,
                overwrite_weights=ns.overwrite_weights,
                fs=None
            )
            data_vars.update(amsr2_vars)
        except Exception as e:
            log.error(f"AMSR2 processing failed for {current_date}: {e}")
        
        # try:
        #     log.info("Starting CEDA Icechunk Regridding...")
        #     ceda_vars = _regrid_ceda(
        #         input_dir=None,  # No longer needed, configured internally
        #         lis_grid=lis_grid,
        #         method="bilinear",
        #         weights_dir=ns.weights_dir,
        #         current_date=current_date,
        #         overwrite_weights=ns.overwrite_weights,
        #         fs=None
        #     )
        #     data_vars.update(ceda_vars)
        # except Exception as e:
        #     log.error(f"CEDA processing failed for {current_date}: {e}")
            
        # if ns.viirs_dir is not None:
        #     data_vars.update(_regrid_viirs(ns.viirs_dir, lis_area, ns.viirs_method, current_date, fs=fs, max_tiles=ns.max_viirs_tiles))
    
        # # Pass the PRE-LOADED dataset instead of the file path
        # if icesat2_full_da is not None:
        #     icesat2_vars = _regrid_icesat2(
        #         icesat2_full_da,   # Pass the actual DataArray, not the string path
        #         lis_area,
        #         lis_dem,           # Pass the DEM, not lis_grid
        #         current_date,
        #         source_path=ns.icesat2_parquet
        #     )
        #     data_vars.update(icesat2_vars)
    
        if not data_vars:
            log.error(
                "No input directories provided or no data found. "
                "Pass at least one of --amsr2-dir, --ceda-dir, --viirs-dir, --icesat2-parquet."
            )
            raise SystemExit(1)

        # Build a daily dataset AND add the 'time' dimension
        ds_day = xr.Dataset(
            data_vars,
            coords={
                "lat": lis_grid["lat"],
                "lon": lis_grid["lon"],
                "time": [current_date]  
            }
        )
        all_daily_datasets.append(ds_day)

    if not all_daily_datasets:
        log.error("No data processed for any dates!")
        raise SystemExit(1)

    # Concatenate all daily datasets along the new 'time' dimension
    log.info("Concatenating all dates along the time dimension...")
    ds_out = xr.concat(all_daily_datasets, dim="time")

    # Write the final combined NetCDF
    encoding = {
        var: {"dtype": "float32", "_FillValue": np.float32("nan"), "zlib": True}
        for var in ds_out.data_vars
    }
    
    log.info(f"Writing combined output to {ns.output_path} …")
    ds_out.to_netcdf(ns.output_path, encoding=encoding)
    log.info("Done!")

    # # Build combined Dataset with shared LIS lat/lon coordinates
    # ds_out = xr.Dataset(
    #     data_vars,
    #     coords={
    #         "lat": lis_grid["lat"],
    #         "lon": lis_grid["lon"],
    #     },
    #     attrs={
    #         "description": (
    #             "Combined SWE and snow-cover observations regridded to the "
    #             "LIS 1 km Lambert Conformal grid (Missouri/NMP domain)."
    #         ),
    #         "conventions": "CF-1.8",
    #     },
    # )

    # encoding = {
    #     var: {"dtype": "float32", "_FillValue": np.float32("nan")}
    #     for var in data_vars
    # }

    # log.info("Writing combined output to %s …", ns.output_path)
    # _write_output(ds_out, ns.output_path, encoding, fs=fs)
    # log.info("Done: %s", ns.output_path)

    # log.info("Variables written:")
    # for var in data_vars:
    #     shape = data_vars[var].shape
    #     log.info("  %-45s %s", var, shape)

            
if __name__ == "__main__":
    main()