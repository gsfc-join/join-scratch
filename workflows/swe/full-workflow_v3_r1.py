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
from pyresample.kd_tree import resample_nearest

import os
import boto3
import time

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
    
    # Grab the raw 1D coordinate arrays
    raw_lons = handler._lon.copy()
    raw_lats = handler._lat.copy()
    
    # FORCE -180 to 180 longitude convention
    raw_lons = np.where(raw_lons > 180, raw_lons - 360, raw_lons)
    
    # 2. Fix the 180-degree phase shift in BOTH data and coordinates
    # The L3 global grids usually need both the data and the longitude array rolled!
    log.info("Applying a physical 180-degree roll to the AMSR2 data matrix and coordinates...")
    half_width = len(raw_lons) // 2
    
    # Roll the data array
    da = da.roll(lon=half_width, roll_coords=False)
    
    # Roll the longitude array so it matches the newly shifted data
    rolled_lons = np.roll(raw_lons, half_width)
    
    # Strictly attach the fixed, rolled, -180/180 coordinates!
    da = da.assign_coords(lat=raw_lats, lon=rolled_lons)

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

            print(f'vmax is {vmax}')
                
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
    #compute_weights(source_grid, lis_grid, weights_path, method=method, overwrite=True)
    regridder = load_regridder(source_grid, lis_grid, weights_path, method=method)

    # 6. Extract the data and apply the regridder
    ds_mean = da.isel(inner=0).drop_vars("inner", errors="ignore").to_dataset(name="Geophysical Data")
    ds_unc  = da.isel(inner=1).drop_vars("inner", errors="ignore").to_dataset(name="Geophysical Data")

    # Quick diagnostic: does the native data even have values here?
    valid_native = np.count_nonzero(~np.isnan(ds_mean["Geophysical Data"].values))
    log.info(f"AMSR2 native matrix has {valid_native} valid pixels before regridding.")

    log.info("AMSR2: regridding mean (inner=0) …")
    rg_mean = regridder(ds_mean)["Geophysical Data"]
    
    # Re-attach LIS coordinates to the output array for accurate map plotting
    rg_mean = rg_mean.assign_coords(lat=lis_grid.lat, lon=lis_grid.lon)

    # Find the max value from the regridded data to match colorbars
    plot_vmax = float(np.nanmax(rg_mean.values))
    log.info(f"Setting plot colorbar maximum to {plot_vmax:.2f}")

    # PLOT 1 (Delayed): Native Data using the regridded max value
    da_plot_native = da_mean_native.sortby("lon")
    
    _debug_plot(da_plot_native, "AMSR2 Native Grid (Central U.S.)", "amsr2_01_native.png", extents=lis_extents, vmax=plot_vmax)

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
    
def _regrid_ceda(
    input_dir: str,
    lis_grid: xr.Dataset,
    method: str,
    weights_dir: str,    
    current_date: pd.Timestamp,
    overwrite_weights: bool = False,    
    fs=None,
) -> dict[str, xr.DataArray]:
    """Regrid the file found and return {var_name: DataArray}."""
    # Format the date to match how it appears in your filenames
    # Example format: "20190101"
    date_str = current_date.strftime("%Y%m%d") 
    
    all_files = _list_files(input_dir, ".nc", fs=fs)
    if not all_files:
        log.warning("No CEDA files found under %s — skipping", input_dir)
        return {}

    files = [f for f in all_files if date_str in f]
    
    path = files[0]
    log.info("CEDA: using %s", _path_name(path))

    if _is_s3(path):
        handler = handler_from_s3(CedaFileHandler, path, fs=fs)
    else:
        handler = CedaFileHandler.from_path(path)

    ds = handler.get_dataset()

    lat_vals = ds["lat"].values if "lat" in ds else ds["y"].values
    lon_vals = ds["lon"].values if "lon" in ds else ds["x"].values
    if lat_vals.ndim == 2:
        lat_vals = lat_vals[:, 0]
    if lon_vals.ndim == 2:
        lon_vals = lon_vals[0, :]
    source_grid = xr.Dataset(
        coords={"lat": np.sort(np.unique(lat_vals)), "lon": np.sort(np.unique(lon_vals))}
    )

    weights_local_dir = _ensure_local_dir(weights_dir)
    weights_path = weights_local_dir / f"ceda-lis-weights-{method}.nc"
    compute_weights(source_grid, lis_grid, weights_path, method=method, overwrite=overwrite_weights)
    regridder = load_regridder(source_grid, lis_grid, weights_path, method=method)

    # Restore lat/lon as dim names for xESMF
    ds_xesmf = ds.swap_dims({"y": "lat", "x": "lon"})

    log.info("CEDA: regridding swe and swe_std …")
    rg = regridder(ds_xesmf)

    def _da(arr, long_name, units):
        return xr.DataArray(
            arr.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={"long_name": long_name, "units": units, "source": _path_name(path)},
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
    df_full,  # Now taking a pandas DataFrame
    lis_area,
    current_date: pd.Timestamp,
    source_path: str,
    fs=None,
) -> dict[str, xr.DataArray]:

    print("starting the _regrid_icesat2")
    
    """Regrid ICESat-2 ATL06 data and compute snow depth (h_li - mosaic.stats.median)."""
    
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
        h_li = pd.to_numeric(df_daily["h_li"], errors="coerce").to_numpy(dtype=np.float64)
        
        # Unpack the list structure in mosaic.stats.median
        if "mosaic.stats.median" in df_daily.columns:
            # Custom function to grab the first item if it's a list, otherwise NaN
            def extract_val(x):
                if isinstance(x, (list, np.ndarray)):
                    return x[0] if len(x) > 0 else np.nan
                return x # In case it's already a float
                
            # Apply unpacking, then coerce to float64
            dem_vals_unpacked = df_daily["mosaic.stats.median"].apply(extract_val)
            dem_vals = pd.to_numeric(dem_vals_unpacked, errors="coerce").to_numpy(dtype=np.float64)
        else:
            log.warning("ICESat-2: 'mosaic.stats.median' column not found in Parquet. Cannot compute snow depth.")
            return {}
            
    except Exception as e:
        log.error(f"ICESat-2: Error extracting column values: {e}")
        return {}
    
    # Calculate snow depth
    # Calculate snow depth
    snow_depth_vals = h_li - dem_vals
    
    # ---------------------------------------------------------
    # NEW DIAGNOSTICS: Inspect the 1D track before regridding
    # ---------------------------------------------------------
    # 1. How many points are exactly zero?
    # exact_zeros = np.count_nonzero(snow_depth_vals == 0.0)
    
    # 2. How many points have identical h_li and dem_vals? 
    # (Should match exact_zeros unless NaNs are involved)
    # identical_vals = np.count_nonzero(h_li == dem_vals)
    
    # 3. What is the physical range of the non-zero snow depths?
    # valid_mask = ~np.isnan(snow_depth_vals)
    # if np.any(valid_mask):
    #     snow_min = np.nanmin(snow_depth_vals)
    #     snow_max = np.nanmax(snow_depth_vals)
    #     snow_mean = np.nanmean(snow_depth_vals)
    # else:
    #     snow_min, snow_max, snow_mean = np.nan, np.nan, np.nan
        
    # log.info(f"ICESat-2 Track Diagnostics:")
    # log.info(f"  -> Exact Zeros (h_li == dem): {exact_zeros} points")
    # log.info(f"  -> Non-Zero Valid Points: {np.count_nonzero(valid_mask) - exact_zeros}")
    # log.info(f"  -> Snow Depth Range: Min={snow_min:.2f}m, Max={snow_max:.2f}m, Mean={snow_mean:.2f}m")
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    # FILTER PHYSICAL ANOMALIES (Clouds, Canopy, Deep Canyons)
    # ---------------------------------------------------------
    # Realistic snow depth limits (e.g., -5m to +25m to account for some DEM error/drifts)
    snow_depth_vals = np.where(
        (snow_depth_vals >= -10.0) & (snow_depth_vals <= 30.0),
        snow_depth_vals,
        np.nan
    )
    
    valid_filtered = np.count_nonzero(~np.isnan(snow_depth_vals))
    log.info(f"ICESat-2: Retained {valid_filtered} realistic snow depth points after physical filtering.")
    
    # # QUICK DEBUG: Check valid counts
    # valid_hli = np.count_nonzero(~np.isnan(h_li))
    # valid_dem = np.count_nonzero(~np.isnan(dem_vals))
    # valid_snow = np.count_nonzero(~np.isnan(snow_depth_vals))
    # log.info(f"ICESat-2 Debug: valid h_li={valid_hli}, valid dem={valid_dem}, valid snow={valid_snow}")

    # Create the source area definition for Pyresample from the flat arrays
    lons_da = xr.DataArray(lons, dims=["obs"])
    lats_da = xr.DataArray(lats, dims=["obs"])
    source_def = SwathDefinition(lons=lons_da, lats=lats_da)

    # Wrap the pure float64 data arrays AND attach the source_def to their metadata
    da_snow = xr.DataArray(snow_depth_vals, dims=["obs"])
    da_snow.attrs["area"] = source_def

    da_h_li = xr.DataArray(h_li, dims=["obs"])
    da_h_li.attrs["area"] = source_def

    da_dem = xr.DataArray(dem_vals, dims=["obs"])
    da_dem.attrs["area"] = source_def

    # Use the existing regrid wrapper
    from join_scratch.regrid import regrid
    
    log.info("ICESat-2: Regridding snow depth...")
    rg_snow_depth = regrid(da_snow, source_def, lis_area, method="mean")
    
    log.info("ICESat-2: Regridding h_li...")
    rg_h_li = regrid(da_h_li, source_def, lis_area, method="mean")

    log.info("ICESat-2: Regridding 3DEP DEM...")
    rg_dem = regrid(da_dem, source_def, lis_area, method="mean")

    log.info("ICESat-2: Regridding snow depth...")
    rg_snow_depth = regrid(da_snow, source_def, lis_area, method="mean")
    
    log.info("ICESat-2: Regridding h_li...")
    rg_h_li = regrid(da_h_li, source_def, lis_area, method="mean")

    # ---------------------------------------------------------
    # NEW DIAGNOSTICS: Inspect the 2D grid after regridding
    # ---------------------------------------------------------
    # grid_hli_valid = np.count_nonzero(~np.isnan(rg_h_li.values))
    # grid_snow_valid = np.count_nonzero(~np.isnan(rg_snow_depth.values))
    # grid_snow_zeros = np.count_nonzero(rg_snow_depth.values == 0.0)
    
    # log.info(f"ICESat-2 Grid Diagnostics:")
    # log.info(f"  -> Gridded h_li pixels: {grid_hli_valid}")
    # log.info(f"  -> Gridded snow pixels (total valid): {grid_snow_valid}")
    # log.info(f"  -> Gridded snow pixels (exactly zero): {grid_snow_zeros}")
    # ---------------------------------------------------------

    return {
        "icesat2_snow_depth": xr.DataArray(
            rg_snow_depth.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "ICESat-2 estimated snow depth (h_li - mosaic.stats.median)",
                "units": "meters",
                "source": f"{source_path}",
            }
        ),
        "icesat2_h_li": xr.DataArray(
            rg_h_li.values.astype(np.float32),
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

    # Build a shared fsspec store if any S3 paths are present
    any_s3 = any(
        _is_s3(str(p))
        for p in [ns.lis_path, ns.amsr2_dir, ns.ceda_dir, ns.viirs_dir, ns.icesat2_parquet]
        if p is not None
    )
    fs = make_fs() if any_s3 else None

    log.info("Loading LIS grid and AreaDefinition...")
    lis_grid = load_lis_grid(ns.lis_path, fs=fs)    
    lis_area = build_lis_area_definition(ns.lis_path, fs=fs, cache_dir=ns.weights_dir, overwrite=ns.overwrite_weights)
  
    # Pre-load ICESat-2 data (if provided) to avoid reloading it on every date loop    
    icesat2_df = None
    if ns.icesat2_parquet is not None:
        log.info(f"Pre-loading ICESat-2 data from {ns.icesat2_parquet}...")
        if not _is_s3(ns.icesat2_parquet) and not Path(ns.icesat2_parquet).exists():
            log.warning(f"ICESat-2 file not found at {ns.icesat2_parquet} — skipping ICESat-2.")
        else:
            try:
                icesat2_df = pd.read_parquet(ns.icesat2_parquet)  
                # Check what columns actually exist in the dataframe
                log.info(f"ICESat-2 Parquet columns: {list(icesat2_df.columns)}")
                
                # Check if the time information is already the index
                if icesat2_df.index.name in ['time', 'time_ns']:
                    # Convert the index to datetime (handles both string and nanosecond integer types)
                    icesat2_df.index = pd.to_datetime(icesat2_df.index)
                
                # Otherwise, check if it's a column and set it as the index
                elif 'time_ns' in icesat2_df.columns:
                    icesat2_df.index = pd.to_datetime(icesat2_df['time_ns'])
                    icesat2_df.index.name = 'time_ns'
                elif 'time' in icesat2_df.columns:
                    icesat2_df.index = pd.to_datetime(icesat2_df['time'])
                    icesat2_df.index.name = 'time'
                else:
                    log.warning("ICESat-2 data does not contain a recognizable time column/index for daily slicing.")

                # SORT THE INDEX to fix the "non-monotonic" slicing error
                if icesat2_df is not None and not icesat2_df.index.is_monotonic_increasing:
                    log.info("Sorting ICESat-2 data by time...")
                    icesat2_df.sort_index(inplace=True)
        
            except Exception as e:
                log.error(f"Failed to read ICESat-2 parquet file: {e}")
                icesat2_df = None

    # Generate a list of daily dates
    dates = pd.date_range(start=ns.start_date, end=ns.end_date, freq='D')
    
    # -------------------------------------------------------------------------
    # MAIN TIME LOOP
    # -------------------------------------------------------------------------
    for current_date in dates:
        # Use %Y%m%d to get '20190101' instead of '2019-01-01'
        date_str = current_date.strftime('%Y%m%d') 
        log.info(f"--- Processing date: {date_str} ---")     
    
        data_vars: dict[str, xr.DataArray] = {}

        if ns.amsr2_dir is not None:
            data_vars.update(_regrid_amsr2(ns.amsr2_dir, lis_grid, ns.amsr2_method, ns.weights_dir, current_date, overwrite_weights=ns.overwrite_weights, fs=fs))
    
        if ns.ceda_dir is not None:
            data_vars.update(_regrid_ceda(ns.ceda_dir, lis_grid, ns.ceda_method, ns.weights_dir, current_date, overwrite_weights=ns.overwrite_weights, fs=fs))
    
        if ns.viirs_dir is not None:
            data_vars.update(_regrid_viirs(ns.viirs_dir, lis_area, ns.viirs_method, current_date, fs=fs, max_tiles=ns.max_viirs_tiles))
    
        if icesat2_df is not None:
            icesat2_vars = _regrid_icesat2(
                icesat2_df,
                lis_area,
                current_date,
                source_path=ns.icesat2_parquet
            )
            data_vars.update(icesat2_vars)
    
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
