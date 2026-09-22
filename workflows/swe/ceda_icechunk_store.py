#!/usr/bin/env python

import os
# Silence icechunk rust warnings. Must be set before importing icechunk.
os.environ.setdefault("RUST_LOG", "error")

import icechunk
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

import obstore
from obspec_utils.registry import ObjectStoreRegistry

import logging
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from zarr.codecs import Zlib, Shuffle # Use Zarr v3 native codecs
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# CF time reference
TIME_UNITS = "days since 1970-01-01"
TIME_CALENDAR = "proleptic_gregorian"
TIME_EPOCH = pd.Timestamp("1970-01-01")

CEDA_BASE = "https://dap.ceda.ac.uk/neodc/esacci/snow/data/swe/MERGED/v4.0"
FNAME_TMPL = "{date}-ESACCI-L3C_SNOW-SWE-SSMIS_DMSP-fv4.0.nc"

CEDA_URL = "https://dap.ceda.ac.uk/"


def ceda_url(date: pd.Timestamp) -> str:
    """Return the CEDA HTTPS URL for a date."""
    return f"{CEDA_BASE}/{date.year:04d}/{date.month:02d}/" + FNAME_TMPL.format(
        date=date.strftime("%Y%m%d")
    )


def _check_file_exists(url: str) -> bool:
    """Check if the file exists by performing a HEAD request."""
    try:
        store = obstore.store.HTTPStore(CEDA_URL)
        relative_path = url.replace(CEDA_URL, "")
        store.head(relative_path)
        return True
    except Exception as e:
        log.warning(f"File not found or inaccessible: {url} ({e})")
        return False


def _get_manifests(chunk_url: str, registry: ObjectStoreRegistry, loadable_vars: list[str] = None) -> xr.Dataset:
    """Open a NetCDF file via VirtualiZarr and extract its manifest."""
    try:
        parser = HDFParser()
        vds = open_virtual_dataset(
            url=chunk_url,
            parser=parser,
            registry=registry,
            loadable_variables=loadable_vars or [],
        )
        return vds
    except Exception as exc:
        log.warning("Failed to open %s: %s", chunk_url, exc)
        raise


def get_codecs_v3(var) -> tuple:
    """Safely extract the exact compression and filter settings (like Shuffle) for Zarr v3."""
    compressors = None
    filters = None
    
    # 1. Read from xarray encoding (standard for NetCDF variables)
    zlib = var.encoding.get('zlib', False)
    complevel = var.encoding.get('complevel', 9)
    shuffle = var.encoding.get('shuffle', False)
    
    # 2. Verify with VirtualiZarr's direct physical array metadata
    if hasattr(var, 'data') and hasattr(var.data, 'zarray'):
        zarray = var.data.zarray
        if zarray.compressor and zarray.compressor.get('id') == 'zlib':
            zlib = True
            complevel = zarray.compressor.get('level', complevel)
        if zarray.filters:
            for f in zarray.filters:
                if f.get('id') == 'shuffle':
                    shuffle = True
                    
    if zlib:
        compressors = [Zlib(level=complevel)]
    if shuffle:
        # Zarr v3 Shuffle codec takes elementsize
        filters = [Shuffle(elementsize=var.dtype.itemsize)]
        
    return compressors, filters


def _get_chunk_structure(var) -> tuple:
    """
    Safely determine the true physical chunk size from a VirtualiZarr variable.
    Returns a tuple of chunk sizes for each dimension.
    """
    # 1. Most accurate: VirtualiZarr ManifestArray zarray property
    if hasattr(var, 'data') and hasattr(var.data, 'zarray') and var.data.zarray.chunks:
        return tuple(var.data.zarray.chunks)
        
    # 2. Xarray tuple of tuples for chunks
    if var.chunks is not None:
        return tuple(c[0] for c in var.chunks)
        
    # 3. Fallback to encoding chunk sizes
    if 'chunksizes' in var.encoding:
        return tuple(var.encoding['chunksizes'])
        
    # 4. Old logic for manifest iter_refs as last resort (Restored from your original code!)
    if hasattr(var, 'data') and hasattr(var.data, 'manifest'):
        manifest = var.data.manifest
        max_indices = {}
        for chunk_key, _ in manifest.iter_refs():
            # Safely handle string keys like "0.0.1"
            if isinstance(chunk_key, str):
                parsed_key = [int(k) for k in chunk_key.split('.')]
            else:
                parsed_key = list(chunk_key)

            for i, idx in enumerate(parsed_key):
                if i not in max_indices:
                    max_indices[i] = idx
                else:
                    max_indices[i] = max(max_indices[i], idx)
        
        if max_indices:
            chunks = []
            # Note: We must compare against var.shape to determine chunk sizes.
            # If the manifest yields key '0.0.1', max index is 1, so there are 2 chunks.
            for i, dim_size in enumerate(var.shape):
                if i in max_indices:
                    num_chunks = max_indices[i] + 1
                    chunk_size = dim_size // num_chunks
                    chunks.append(chunk_size)
                else:
                    chunks.append(dim_size)
            return tuple(chunks)
            
    return var.shape


def initialize_store(
    session: icechunk.Session,
    sample_url: str,
    registry: ObjectStoreRegistry,
    commit_msg: str = "Initialize empty store",
) -> None:
    """Create the arrays in a new IceChunk repository based on a sample file."""
    sample_ds = _get_manifests(sample_url, registry, loadable_vars=["lat", "lon"])
    
    root = zarr.open_group(session.store, mode="w", zarr_format=3)

    # Create time dimension (will be extended as we add dates)
    root.require_array(
        "time",
        shape=(0,),
        chunks=(512,),
        dtype="int32",
        dimension_names=["time"],
        attributes={
            "units": TIME_UNITS,
            "calendar": TIME_CALENDAR,
            "long_name": "observation date",
        },
    )

    # Create spatial coordinates from sample
    for coord_name in ["lat", "lon"]:
        if coord_name in sample_ds:
            coord_data = sample_ds[coord_name].values
            root.require_array(
                coord_name,
                shape=coord_data.shape,
                chunks=coord_data.shape,
                dtype=coord_data.dtype,
                dimension_names=[coord_name],
                attributes=dict(sample_ds[coord_name].attrs),
            )
            root[coord_name][:] = coord_data

    # Determine data variables to store
    data_vars = [var for var in sample_ds.data_vars if var not in ["lat", "lon", "time"]]
    log.info(f"Found data variables: {data_vars}")
    
    for var_name in data_vars:
        var = sample_ds[var_name]
        dims = list(var.dims)
        has_time = "time" in dims
        
        full_chunks = _get_chunk_structure(var)
        
        if has_time:
            time_idx = var.dims.index("time")
            spatial_chunks = tuple(c for i, c in enumerate(full_chunks) if i != time_idx)
            spatial_shape = tuple(s for i, s in enumerate(var.shape) if i != time_idx)
            
            full_chunks = (1,) + spatial_chunks
            full_shape = (0,) + spatial_shape
            full_dims = ["time"] + [d for d in dims if d != "time"]
            log.info(f"Variable {var_name}: dims={full_dims}, shape={full_shape}, chunks={full_chunks}")
        else:
            full_shape = var.shape
            full_dims = dims
            log.info(f"Variable {var_name} (no time): dims={full_dims}, shape={full_shape}, chunks={full_chunks}")
            
        attrs = dict(var.attrs)
        
        # Extract precise _FillValue
        fill_val = var.encoding.get("_FillValue", attrs.get("_FillValue", None))
        if fill_val is not None:
            fill_val = np.array(fill_val, dtype=var.dtype).item()

        # Extract precise Codec pipeline using native Zarr v3 codecs
        compressors, filters = get_codecs_v3(var)
        
        # Enforce Zlib to avoid Zarr falling back to Zstd
        if compressors is None and has_time:
            compressors = [Zlib(level=9)]
            
        root.require_array(
            var_name,
            shape=full_shape,
            chunks=full_chunks,
            dtype=var.dtype,
            fill_value=fill_val,
            compressors=compressors,
            filters=filters,
            dimension_names=full_dims,
            attributes=attrs,
        )

    session.commit(commit_msg)
    log.info("Initialized empty store")


def stored_days(repo: icechunk.Repository) -> list[int]:
    """Return stored day values in their current storage order."""
    session = repo.readonly_session("main")
    try:
        root = zarr.open_group(session.store, mode="r", zarr_format=3)
        return root["time"][:].tolist()
    except zarr.errors.GroupNotFoundError:
        log.warning("Store exists but is not initialized (no root group found)")
        return []


def insert_date(date: pd.Timestamp, repo: icechunk.Repository, registry: ObjectStoreRegistry) -> None:
    """Insert a date using virtual references to the source NetCDF chunks."""
    day_val = int((date - TIME_EPOCH).days)
    date_iso = date.strftime("%Y-%m-%d")
    current = stored_days(repo)

    if day_val in current:
        slot = current.index(day_val)
        log.info("%s skipped (already present, slot %d)", date_iso, slot)
        return

    chunk_url = ceda_url(date)

    if not _check_file_exists(chunk_url):
        log.warning("File does not exist for %s, skipping.", date_iso)
        return

    try:
        vds = _get_manifests(chunk_url, registry)
    except Exception as exc:
        log.warning("Could not fetch %s: %s", date_iso, exc)
        return

    new_slot = len(current)
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="r+", zarr_format=3)

    # Extend time dimension
    root["time"].resize((new_slot + 1,))
    root["time"][new_slot] = day_val

    data_vars = [var for var in vds.data_vars if var not in ["lat", "lon", "time"]]

    for var_name in data_vars:
        if var_name not in root:
            log.warning(f"Variable {var_name} not found in store, skipping")
            continue
            
        var = vds[var_name]
        
        # If the variable is a 0-D scalar (like spatial_ref), ignore it for future inserts.
        if "time" not in var.dims:
            continue
        
        current_shape = root[var_name].shape
        new_shape = (new_slot + 1,) + current_shape[1:]
        root[var_name].resize(new_shape)
        
        if hasattr(var, 'data') and hasattr(var.data, 'manifest'):
            manifest = var.data.manifest
            
            for chunk_key, reference in manifest.iter_refs():
                time_idx = var.dims.index('time')
                
                # Safely split standard string dot-separated keys (e.g. "0.0.0")
                if isinstance(chunk_key, str):
                    chunk_key_list = [int(k) for k in chunk_key.split('.')]
                else:
                    chunk_key_list = list(chunk_key)
                
                # Drop the time index only if the physical chunk explicitly contains it
                if len(chunk_key_list) == len(var.dims):
                    spatial_chunk_key = tuple(chunk_key_list[:time_idx] + chunk_key_list[time_idx+1:])
                else:
                    spatial_chunk_key = tuple(chunk_key_list)
                
                adjusted_key = (new_slot,) + spatial_chunk_key
                key_str = f"{var_name}/c/" + "/".join(map(str, adjusted_key))
                
                session.store.set_virtual_ref(
                    key_str,
                    reference["path"],
                    offset=reference["offset"],
                    length=reference["length"],
                    validate_container=False,
                )
                
    session.commit(f"Insert {date_iso} at slot {new_slot}")
    log.info("%s inserted at slot %d", date_iso, new_slot)


def select_time_range(ds: xr.Dataset, start: str, end: str) -> xr.Dataset:
    times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
    return ds.isel(time=np.flatnonzero(mask))


# Build object store registry for CEDA HTTPS access
store = obstore.store.HTTPStore(CEDA_URL)
registry = ObjectStoreRegistry({CEDA_URL: store})

# Store Prefix - bumped to v7 to ensure clean chunk metadata
STORE_PREFIX = "JOIN/icechunk-stores/CEDA_Store_v2"

try:
    repo = icechunk.Repository.open(
        icechunk.s3_storage(
            bucket="airborne-smce-prod-user-bucket",
            prefix=f"{STORE_PREFIX}/",
        ),
        authorize_virtual_chunk_access={CEDA_URL: icechunk.credentials.HttpAccess},
    )
    log.info("Found existing icechunk store")
    
    if not stored_days(repo) and len(stored_days(repo)) == 0:
        try:
            session = repo.readonly_session("main")
            zarr.open_group(session.store, mode="r", zarr_format=3)
        except zarr.errors.GroupNotFoundError:
            log.info("Store exists but is not initialized, will initialize now")
            store_needs_init = True
            
except icechunk.IcechunkError as exc:
    log.warning(
        "Failed to open existing repo with error: %s. "
        "Trying to create a fresh repo.",
        exc,
    )
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(CEDA_URL, icechunk.http_store())
    )
    repo = icechunk.Repository.create(
        icechunk.s3_storage(
            bucket="airborne-smce-prod-user-bucket",
            prefix=STORE_PREFIX,
        ),
        config=config,
    )
    store_needs_init = True

# Initialize the store if needed
if store_needs_init:
    sample_date = pd.Timestamp("2018-10-01")  # Matches start of your date sequence
    sample_url = ceda_url(sample_date)
    initialize_store(repo.writable_session("main"), sample_url, registry)


# --- DRIVER CODE ---

date_seq = pd.date_range(
    start="2018-10-01",
    end="2019-06-07",
    freq="D",
)

for date in tqdm(date_seq):
    insert_date(date, repo, registry)

log.info("Testing reading a subset of data")
session = repo.readonly_session("main")

# IMPORTANT: xarray needs to know this is a Zarr v3 store and not to look for consolidated v2 metadata
ds = xr.open_zarr(session.store, zarr_format=3, consolidated=False)

print("Available variables:", list(ds.data_vars))
print("Dataset structure:")
print(ds)

# Select a time range for testing
sub = select_time_range(ds, "2018-10-01", "2019-06-01")
print("\nTime range subset:")
print(sub)