import sys
import datetime
import pandas as pd
import xarray as xr
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
import obstore
from obspec_utils.registry import ObjectStoreRegistry
import icechunk
import s3fs
from icechunk.credentials import s3_credentials
import boto3
import os

# Use boto3 to grab the working credentials
session = boto3.Session()
creds = session.get_credentials()

# Initialize to None so we can safely check it later
static_creds = None

if creds:
    frozen_creds = creds.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen_creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen_creds.secret_key
    if frozen_creds.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen_creds.token
        
    # Prevent obstore from trying to hit the EC2 metadata endpoint
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    
    print("Successfully injected boto3 credentials into environment for obstore.")
    
    # FIX: Create the icechunk credentials using the helper function and frozen_creds
    static_creds = s3_credentials(
        access_key_id=frozen_creds.access_key,
        secret_access_key=frozen_creds.secret_key,
        session_token=frozen_creds.token
    )
else:
    print("Warning: boto3 could not find AWS credentials.")

# ==========================================
# 1. Configuration 
# ==========================================
bucket = "airborne-smce-prod-user-bucket"
source_prefix = "JOIN/VIIRS/VJ110A1F"
dest_prefix = "JOIN/icechunk-stores/VIIRS/VJ110A1F/V002/"

source_base_url = f"s3://{bucket}"

# Target date range (can be historical or future dates)
target_start_date = datetime.date(2018, 10, 1)
target_end_date = datetime.date(2019, 7, 1)

# AWS Region (modify if your SMCE bucket is not in us-east-1)
aws_region = "us-east-1"

# ==========================================
# 2. Connect to Icechunk & Check Existing Inventory
# ==========================================
print(f"Connecting to Icechunk repository at s3://{bucket}/{dest_prefix}...")
storage = icechunk.s3_storage(bucket=bucket, prefix=dest_prefix, region=aws_region)

config = icechunk.RepositoryConfig.default()

# Configure virtual chunk container for S3
container = icechunk.VirtualChunkContainer(
    name="viirs_archive",       
    url_prefix=f"{source_base_url}/",  
    store=icechunk.s3_store(region=aws_region)
)
config.set_virtual_chunk_container(container)

# FIX: Safely construct kwargs to authorize virtual chunk access
repo_kwargs = {
    "storage": storage,
    "config": config
}

if static_creds:
    repo_kwargs["authorize_virtual_chunk_access"] = {
        f"{source_base_url}/": static_creds
    }

repo = icechunk.Repository.open_or_create(**repo_kwargs)

session = repo.writable_session("main")

# Collect existing dates to avoid duplicates
existing_dates = set()
is_first_run = True

try:
    with xr.open_zarr(session.store, zarr_format=3) as ds_existing:
        if "time" in ds_existing:
            existing_times = pd.to_datetime(ds_existing.time.values)
            existing_dates = set(existing_times.date)
        is_first_run = False
        print(f"Existing dataset found! It currently contains {len(existing_dates)} days of data.")
except Exception:
    print("No existing dataset found (or store is empty). This will be an initial run.")

# ==========================================
# 3. Setup Registry & Discover S3 URLs
# ==========================================
print("\nInitializing object store registry for S3...")
parser = HDFParser()
# Use obstore S3 store integration
s3_store = obstore.store.S3Store.from_url(f"{source_base_url}/")
registry = ObjectStoreRegistry({f"{source_base_url}/": s3_store})

print("\nDiscovering missing files via s3fs...")
fs = s3fs.S3FileSystem(anon=False)

s3_paths_to_process = []
date_mapping = {}

current_date = target_start_date
while current_date <= target_end_date:
    if current_date in existing_dates:
        print(f"  -> Skipping {current_date}: Already in store.")
    else:
        year = current_date.strftime("%Y")
        month = current_date.strftime("%m")
        day = current_date.strftime("%d")
        
        # Glob to find the actual filenames because processing timestamps vary
        search_path = f"{bucket}/{source_prefix}/{year}/{month}/{day}/*.h5"
        found_files = fs.glob(search_path)
        
        if not found_files:
            print(f"  -> WARNING: No files found for {current_date} at {search_path}")
        else:
            for f in found_files:
                url = f"s3://{f}"
                s3_paths_to_process.append(url)
                date_mapping[url] = current_date
        
    current_date += datetime.timedelta(days=1)

if not s3_paths_to_process:
    print("\nAll requested dates are already present or no files were found. Nothing to process!")
    sys.exit(0)

# ==========================================
# 4. Virtualize and Concatenate Datasets
# ==========================================
print(f"\nVirtualizing {len(s3_paths_to_process)} missing daily files via obstore S3...")
vds_list = []
for url in s3_paths_to_process:
    try:
        vds = open_virtual_dataset(url=url, parser=parser, registry=registry)
        
        # VIIRS HDF5 files may not have a time dimension intrinsically. 
        # We expand dims here so xr.concat(dim="time") succeeds later.
        file_date = pd.to_datetime(date_mapping[url])
        vds = vds.expand_dims({"time": [file_date]})
        
        print(f"Successfully mapped: {url}")
        vds_list.append(vds)
    except Exception as e:
        print(f"  -> WARNING: Unexpected error mapping {url}: {e}")

if not vds_list:
    print("\nNo valid files could be mapped. Exiting gracefully.")
    sys.exit(0)

print("\nConcatenating virtual datasets along the 'time' dimension...")
# Note: If your directory contains multiple spatial tiles (e.g. h08v05 AND h09v05) per day, 
# a standard time concat will conflict. You may need to adapt this to concat spatially if so.
combined_vds = xr.concat(
    vds_list, 
    dim="time", 
    coords="minimal", 
    data_vars="minimal", 
    compat="override"
)

# ==========================================
# 5. Write to Icechunk and Commit
# ==========================================
if is_first_run:
    print("\nInitializing new dataset in Icechunk...")
    combined_vds.vz.to_icechunk(session.store, mode="w")
    action = "Initialized"
else:
    print("\nAppending new data to existing Icechunk dataset...")
    combined_vds.vz.to_icechunk(session.store, mode="a", append_dim="time")
    action = "Appended"

commit_message = f"{action} {len(vds_list)} VIIRS VJ110A1F records."
session.commit(commit_message)

print(f"\nSuccess! Data committed to Icechunk. Commit message: '{commit_message}'")