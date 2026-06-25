#!/usr/bin/env python

from pathlib import Path
import subprocess
import tempfile

from typing import Sequence

import xarray as xr
import numpy as np
import icechunk
import pandas as pd

from tqdm import tqdm

from join_scratch.utils.s3 import ic_s3_credentials
from join_scratch.regrid.utils import (
    create_scrip_grid,
    load_weights,
    regrid_slice
)

import cartopy.crs as ccrs

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

tmp_dir = Path("/tmp")

bucket = "airborne-smce-prod-user-bucket"
src_prefix = "JOIN/icechunk-stores/VIIRS/VJ110A1F"

esmf_out_s3 = f"s3://{bucket}/JOIN/cached-weights/VIIRS/lis-1km-missouri.zarr"
esmf_out_tmp = tmp_dir / "viirs-lis-weights.nc4"

s3_prelim_output = f"s3://{bucket}/JOIN/outputs-preliminary/viirs-swe-regridded.zarr"

time_values = pd.date_range(start="2019-01-01", end="2019-01-07", freq="D")

################################################################################
# Read the LIS grid

lis_prefix = "JOIN/lis_input_NMP_1000m_missouri.nc"
lis_url = f"s3://{bucket}/{lis_prefix}"
lis = xr.open_dataset(lis_url, engine="h5netcdf")

lis_lats = lis["lat"].values
lis_lons = lis["lon"].values

assert lis_lats.shape == lis_lons.shape

lis_scrip = create_scrip_grid(lis_lats, lis_lons, title = "LIS NMP 1000m Missouri")
lis_scrip_nc4 = tmp_dir / "lis-scrip.nc4"
lis_scrip.to_netcdf(lis_scrip_nc4, format="NETCDF4")

################################################################################
# Create the bounding box on the sinusoidal grid. Filter to the bounding box 
# first to reduce the amount of data that needs to be loaded into memory for the 
# regrid weights generation.

viirs_sine = ccrs.Sinusoidal()

# First, get the corners of the LIS grid in lat-lon space.
lis_lat_min = lis_lats.min()
lis_lat_max = lis_lats.max()
lis_lon_min = lis_lons.min()
lis_lon_max = lis_lons.max()
lis_bbox_latlon = np.array([
    (lis_lon_min, lis_lat_min),
    (lis_lon_min, lis_lat_max),
    (lis_lon_max, lis_lat_min),
    (lis_lon_max, lis_lat_max)
])

# Next, transform the corners to the VIIRS sinusoidal grid
lis_bbox_pts_viirs = viirs_sine.transform_points(
    ccrs.PlateCarree(),
    lis_bbox_latlon[:,0],
    lis_bbox_latlon[:,1]
)

# VIIRS is really big. But regridding is naiveley parallelizable in terms of 
# destination pixels. So break it up into 4x4 tiles so each one fits into 
# memory and then combine them later.

lis_bbox_xmin = lis_bbox_pts_viirs[:,0].min()
lis_bbox_xmax = lis_bbox_pts_viirs[:,0].max()
lis_bbox_ymin = lis_bbox_pts_viirs[:,1].min()
lis_bbox_ymax = lis_bbox_pts_viirs[:,1].max()

lis_bbox_xspace = np.linspace(lis_bbox_xmin, lis_bbox_xmax, 5)
lis_bbox_yspace = np.linspace(lis_bbox_ymax, lis_bbox_ymin, 5)

from itertools import pairwise
lis_xslices = list(slice(*pair) for pair in pairwise(lis_bbox_xspace))
lis_yslices = list(slice(*pair) for pair in pairwise(lis_bbox_yspace))

###
# lis_xslice = lis_xslices[0]
# lis_yslice = lis_yslices[0]
# del lis_xslice, lis_yslice

# Now, extract the maximum points on each dimension to get the bounding box.
# lis_xslice = slice(lis_bbox_pts_viirs[:,0].min(), lis_bbox_pts_viirs[:,0].max())
# # NOTE: VIIRS has to be sliced from top (max) to bottom (min)
# lis_yslice = slice(lis_bbox_pts_viirs[:,1].max(), lis_bbox_pts_viirs[:,1].min())

################################################################################
# Open icechunk store
credentials = icechunk.credentials.containers_credentials(
    {f"s3://{bucket}/JOIN/VIIRS/": ic_s3_credentials()}
)
repo = icechunk.Repository.open(
    icechunk.s3_storage(
        bucket=bucket,
        prefix=src_prefix,
    ),
    authorize_virtual_chunk_access=credentials
)
session = repo.readonly_session("main")
src_ds = xr.open_zarr(session.store)

##############################
def mk_scrip(src_ds, lis_xslice, lis_yslice):
    src_ds_sub = src_ds.sel(XDim = lis_xslice, YDim = lis_yslice)
    src_xs = src_ds_sub["XDim"].values
    src_ys = src_ds_sub["YDim"].values
    src_xs_2d, src_ys_2d = np.meshgrid(src_xs, src_ys)

    src_latlon = ccrs.PlateCarree().transform_points(viirs_sine, src_xs_2d, src_ys_2d)

    src_lons_2d = src_latlon[:,:,0]
    src_lats_2d = src_latlon[:,:,1]

    src_scrip = create_scrip_grid(src_lats_2d, src_lons_2d, title = "viirs-sinusoidal")
    src_scrip_nc4 = tempfile.NamedTemporaryFile(delete=False, suffix=".nc4", prefix="viirs-scrip-")
    src_scrip.to_netcdf(src_scrip_nc4.name, format="NETCDF4")
    return src_scrip_nc4


def slice_weights(
    src_ds: xr.Dataset,
    lis_xslice: slice,
    lis_yslice: slice
):

    src_scrip_nc4 = mk_scrip(src_ds, lis_xslice, lis_yslice)
    esmf_out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".nc4", prefix="viirs-weights-")
    cmd = [
        "ESMF_RegridWeightGen",
        "--source", str(src_scrip_nc4.name),
        "--destination", str(lis_scrip_nc4),
        "--weight", str(esmf_out_tmp.name),
        "-r",                   # Regional -- not global!
        "--method", "bilinear",
        "--src_type", "SCRIP",
        "--dst_type", "SCRIP",
        "--ignore_unmapped",
        "--netcdf4"
    ]
    _ = subprocess.run(cmd, capture_output=True, text=True)
    return esmf_out_tmp


# NOTE: This can be parallelized
# NOTE: If I rearrange the y and x loops, I can use scipy.sparse.bmat directly 
# because the results will be in row-major order already (not column-major).
esmf_outputs = []
for lis_xslice in tqdm(lis_xslices, desc="x"):
    for lis_yslice in tqdm(lis_yslices, desc="y", leave=False):
        result = slice_weights(src_ds, lis_xslice, lis_yslice)
        esmf_outputs.append(result)

# Now, merge the output files back together
esmf_files = list(Path("~/viirs-weights").expanduser().glob("*.nc4"))

################################################################################
################################################################################

import numpy as np
import xarray as xr
from pathlib import Path


def infer_tile_position(ds: xr.Dataset) -> tuple[float, float]:
    """
    Return (min_lat, min_lon) of the source grid for sorting tiles
    into their 2D grid position.
    """
    return (float(ds["yc_a"].min()), float(ds["xc_a"].min()))


def arrange_tiles(datasets: list[xr.Dataset], nrows: int = 4, ncols: int = 4):
    """
    Sort datasets into a (nrows, ncols) grid by their spatial extent.
    Returns a 2D list tile_grid[row_idx][col_idx] of datasets,
    where row 0 is southernmost and col 0 is westernmost.
    """
    # Attach corner info to each dataset for sorting
    tagged = [
        (ds, float(ds["yc_a"].min()), float(ds["xc_a"].min()))
        for ds in datasets
    ]

    # Get unique lat/lon breakpoints by clustering min corners
    min_lats = sorted(set(round(t[1], 4) for t in tagged))
    min_lons = sorted(set(round(t[2], 4) for t in tagged))

    assert len(min_lats) == nrows, (
        f"Expected {nrows} distinct latitude bands, found {len(min_lats)}: {min_lats}"
    )
    assert len(min_lons) == ncols, (
        f"Expected {ncols} distinct longitude bands, found {len(min_lons)}: {min_lons}"
    )

    tile_grid = [[None] * ncols for _ in range(nrows)]
    for ds, lat, lon in tagged:
        r = min_lats.index(round(lat, 4))
        c = min_lons.index(round(lon, 4))
        assert tile_grid[r][c] is None, f"Duplicate tile at grid position ({r}, {c})"
        tile_grid[r][c] = ds

    assert all(
        tile_grid[r][c] is not None for r in range(nrows) for c in range(ncols)
    ), "Some tile grid positions are unfilled — check tile count or corner rounding"

    return tile_grid, min_lats, min_lons


def merge_esmf_weights_2d(
    weight_files: Sequence[Path | str],
    output_path: Path | str,
    nrows: int = 4,
    ncols: int = 4,
) -> xr.Dataset:
    """
    Merge ESMF weight files from a (nrows x ncols) tiled source grid into a
    single weight file. Tile positions are inferred from source grid corners.

    Assumes:
      - Source grid (n_a) was tiled in a 2D row/col layout
      - Destination grid (n_b) is identical across all tiles
      - Source grid is row-major (C order) flattened into n_a
      - ESMF 1-based indexing in col/row arrays

    Parameters
    ----------
    weight_files : paths to all 16 tile weight files (any order)
    output_path  : path for merged output
    nrows, ncols : tile grid dimensions (default 4x4)
    """
    datasets = [xr.open_dataset(f) for f in weight_files]
    assert len(datasets) == nrows * ncols, (
        f"Expected {nrows * ncols} files, got {len(datasets)}"
    )

    tile_grid, min_lats, min_lons = arrange_tiles(datasets, nrows, ncols)

    # --- Per-tile source grid shape ---
    # Each tile's src_grid_dims gives [ncols_tile, nrows_tile] (ESMF is col-major
    # in its dims convention: first dim is "x"/longitude, second is "y"/latitude)
    tile_na_nrows = np.array(
        [[tile_grid[r][c]["src_grid_dims"].values[1] for c in range(ncols)]
         for r in range(nrows)],
        dtype=int,
    )
    tile_na_ncols = np.array(
        [[tile_grid[r][c]["src_grid_dims"].values[0] for c in range(ncols)]
         for r in range(nrows)],
        dtype=int,
    )

    # Global source grid dimensions
    global_src_nrows = tile_na_nrows[:, 0].sum()   # sum down latitude bands
    global_src_ncols = tile_na_ncols[0, :].sum()   # sum across longitude bands

    # Validate tile consistency: all tiles in same row band have same nrows,
    # all tiles in same col band have same ncols
    for r in range(nrows):
        assert np.all(tile_na_nrows[r, :] == tile_na_nrows[r, 0]), (
            f"Inconsistent tile row heights in row band {r}"
        )
    for c in range(ncols):
        assert np.all(tile_na_ncols[:, c] == tile_na_ncols[0, c]), (
            f"Inconsistent tile col widths in col band {c}"
        )

    # Row/col offsets into the global flattened n_a index for each tile.
    # In a row-major grid of shape (global_nrows, global_ncols), the flattened
    # index of a point at local position (local_row, local_col) within tile
    # (tr, tc) is:
    #   row_offset(tr) * global_ncols + col_offset(tc) + local_row * global_ncols + local_col
    # But since col/row in the weight file are already flattened local indices,
    # we reconstruct local (i, j) and re-flatten into global space.
    row_band_offsets = np.concatenate([[0], np.cumsum(tile_na_nrows[:, 0])])
    col_band_offsets = np.concatenate([[0], np.cumsum(tile_na_ncols[0, :])])

    # --- Process sparse triplets tile by tile ---
    col_list, row_list, S_list = [], [], []

    for tr in range(nrows):
        for tc in range(ncols):
            ds = tile_grid[tr][tc]
            tile_ncols = int(tile_na_ncols[tr, tc])
            row_off = int(row_band_offsets[tr])
            col_off = int(col_band_offsets[tc])

            # Convert 1-based local flat index -> 0-based local (i_row, i_col)
            local_flat = ds["col"].values - 1          # 0-based local flat index
            local_i = local_flat // tile_ncols         # local row within tile
            local_j = local_flat % tile_ncols          # local col within tile

            # Re-flatten into global 0-based index, then back to 1-based
            global_flat = (
                (row_off + local_i) * global_src_ncols
                + (col_off + local_j)
                + 1                                    # back to 1-based
            )
            col_list.append(global_flat.astype(np.int32))

            # dst index (row) is the same grid for all tiles — no offsetting needed
            row_list.append(ds["row"].values.astype(np.int32))
            S_list.append(ds["S"].values)

    col_merged = np.concatenate(col_list)
    row_merged = np.concatenate(row_list)
    S_merged   = np.concatenate(S_list)

    # --- Concatenate n_a variables (source grid metadata) ---
    # Concatenate row-band by row-band, then col within each band.
    # For each row band, concatenate tiles left to right along n_a.
    na_scalar_vars = ["yc_a", "xc_a", "area_a", "frac_a"]
    na_corner_vars = (["yv_a", "xv_a"] if "yv_a" in datasets[0] else [])
    if "mask_a" in datasets[0]:
        na_scalar_vars.append("mask_a")

    all_na_vars = na_scalar_vars + na_corner_vars

    # Build row-band arrays: for each row band, concat tiles left→right
    row_bands = []
    for tr in range(nrows):
        row_bands.append(
            xr.concat(
                [tile_grid[tr][tc] for tc in range(ncols)],
                dim="n_a"
            )
        )

    # Then concat row bands south→north
    merged_na_ds = xr.concat(row_bands, dim="n_a")

    # --- n_b variables: just take from any tile (all identical) ---
    nb_vars = ["yc_b", "xc_b", "area_b", "frac_b"]
    if "yv_b" in datasets[0]:
        nb_vars += ["yv_b", "xv_b"]
    if "mask_b" in datasets[0]:
        nb_vars.append("mask_b")

    # --- Assemble final dataset ---
    merged = xr.Dataset(
        {
            "src_grid_dims": xr.DataArray(
                np.array([global_src_ncols, global_src_nrows], dtype=np.int32),
                dims=["src_grid_rank"],
            ),
            "dst_grid_dims": xr.DataArray(
                tile_grid[0][0]["dst_grid_dims"].values,
                dims=["dst_grid_rank"],
            ),
            **{v: merged_na_ds[v] for v in all_na_vars},
            **{v: tile_grid[0][0][v] for v in nb_vars},
            "col": xr.DataArray(col_merged, dims=["n_s"]),
            "row": xr.DataArray(row_merged, dims=["n_s"]),
            "S":   xr.DataArray(S_merged,   dims=["n_s"]),
        },
        attrs=datasets[0].attrs,
    )

    for ds in datasets:
        ds.close()

    merged.to_netcdf(output_path)
    return merged


datasets = [xr.open_dataset(f) for f in esmf_files]
min_lons = sorted(float(ds["xc_a"].min()) for ds in datasets)
print("Min lons:", min_lons)
print("Gaps:", [round(b - a, 3) for a, b in zip(min_lons, min_lons[1:])])

merged = merge_esmf_weights_2d(esmf_files, Path("~/viirs-weights_merged.nc4").expanduser())

# files = sorted(Path(".").glob("weights_*.nc"))
# assert len(files) == 16, f"Expected 16 weight files, found {len(files)}"
# merged = merge_esmf_weights_2d(files, "weights_merged.nc")
# print(merged)

################################################################################
################################################################################

d0 = xr.open_dataset(esmf_files[0])
d0i = d0.set_index(n_a = "yc_a", n_b = "yc_b")

tiles = [load_weights(f.name) for f in esmf_outputs]

import scipy
W0 = scipy.sparse.hstack(tiles[0:4])
W1 = scipy.sparse.hstack(tiles[4:8])
W2 = scipy.sparse.hstack(tiles[8:12])
W3 = scipy.sparse.hstack(tiles[12:16])
W = scipy.sparse.vstack([W0, W1, W2, W3])

grid_size = 4
blocks = []
for y in range(grid_size):
    row_blocks = [tiles[y + x * grid_size] for x in range(grid_size)]
    blocks.append(row_blocks)

W0 = scipy.sparse.hstack(blocks[0])
W = scipy.sparse.bmat(blocks, format="csr")

# try:
#     log.info("Trying to load existing weights at %s", str(esmf_out_s3))
#     W = load_weights(esmf_out_s3)
#     log.info("Loaded existing weights!")
# except ValueError as e:
#     log.warning(
#         f"Could not read existing weights with error {str(e)}. ",
#         "Recreating weights locally."
#     )
#     cmd = [
#         "ESMF_RegridWeightGen",
#         "--source", str(src_scrip_nc4),
#         "--destination", str(lis_scrip_nc4),
#         "--weight", str(esmf_out_tmp),
#         "-r",                   # Regional -- not global!
#         "--method", "bilinear",
#         "--src_type", "SCRIP",
#         "--dst_type", "SCRIP",
#         "--ignore_unmapped",
#         "--netcdf4"
#     ]
#     log.info("Running %s", " ".join(cmd))
#     result = subprocess.run(cmd, capture_output=True, text=True)
#     # Cache the weights to zarr on S3
#     log.info("Saving weights to S3 %s", " ".join(cmd))
#     xr.open_dataset(esmf_out_tmp, engine="h5netcdf").to_zarr(esmf_out_s3)
#     W = load_weights(esmf_out_tmp)

################################################################################
# Create result dataset

result = (lis[["lat", "lon", "lat_b", "lon_b"]]
          .copy()
          .set_coords(["lon", "lat", "lon_b", "lat_b"]))

reference = result["lat"]
dst_shape = reference.shape
regridded_slices = []

# NOTE: This can be naively parallelized
# NOTE: May need to loop over multiple variables as well (which can also be parallelized)
log.info("Regridding time steps")
for t in tqdm(time_values):
    dat = (src_ds["swe"].sel(time=t, method="nearest"))
    regridded = regrid_slice(dat.values, W, dst_shape)
    reference.coords
    da_slice = xr.DataArray(
        data=regridded,
        coords=reference.coords,
        dims=reference.dims,
        attrs=dat.attrs
    )
    regridded_slices.append(da_slice)

result["swe_ceda"] = xr.concat(regridded_slices, dim=pd.Index(time_values, name="time"))

log.info("Writing result to %s", s3_prelim_output)
result.to_zarr(s3_prelim_output, mode="w")

