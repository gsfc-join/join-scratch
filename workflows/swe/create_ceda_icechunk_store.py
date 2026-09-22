import sys
import datetime
import pandas as pd
import xarray as xr
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
import obstore
from obspec_utils.registry import ObjectStoreRegistry
import icechunk

# ==========================================
# 1. Configuration 
# ==========================================
base_ceda_url = "https://dap.ceda.ac.uk/neodc/esacci/snow/data/swe/MERGED/v4.0/"

bucket = "airborne-smce-prod-user-bucket"
prefix = "JOIN/icechunk-stores/CEDA_Store_v3/"

# Target date range (can be historical or future dates)
target_start_date = datetime.date(2018, 10, 1)
target_end_date = datetime.date(2019, 7, 1)

# ==========================================
# 2. Connect to Icechunk & Check Existing Inventory
# ==========================================
print(f"Connecting to Icechunk repository at s3://{bucket}/{prefix}...")
storage = icechunk.s3_storage(bucket=bucket, prefix=prefix)

config = icechunk.RepositoryConfig.default()
clean_base_url = base_ceda_url.rstrip('/')
container = icechunk.VirtualChunkContainer(
    name="ceda_archive",       
    url_prefix=f"{clean_base_url}/",  
    store=icechunk.http_store()
)
config.set_virtual_chunk_container(container)

repo = icechunk.Repository.open_or_create(
    storage=storage, 
    config=config,
    authorize_virtual_chunk_access={f"{clean_base_url}/": icechunk.credentials.HttpAccess}
)

session = repo.writable_session("main")

# We will collect a set of dates already in the store to avoid duplicates
existing_dates = set()
is_first_run = True

try:
    with xr.open_zarr(session.store, zarr_format=3) as ds_existing:
        # Extract all existing time values and convert to a set of Python dates
        existing_times = pd.to_datetime(ds_existing.time.values)
        existing_dates = set(existing_times.date)
        
        is_first_run = False
        print(f"Existing dataset found! It currently contains {len(existing_dates)} days of data.")
except Exception:
    print("No existing dataset found (or store is empty). This will be an initial run.")

# ==========================================
# 3. Setup Registry & Generate Missing URLs
# ==========================================
print("\nInitializing object store registry...")
parser = HDFParser()
http_store = obstore.store.HTTPStore(f"{clean_base_url}/")
registry = ObjectStoreRegistry({f"{clean_base_url}/": http_store})

urls = []
current_date = target_start_date
while current_date <= target_end_date:
    if current_date in existing_dates:
        # Skip dates that are already in the Icechunk store
        print(f"  -> Skipping {current_date}: Already in store.")
    else:
        year = current_date.strftime("%Y")
        month = current_date.strftime("%m")
        date_str = current_date.strftime("%Y%m%d")
        
        url = f"{clean_base_url}/{year}/{month}/{date_str}-ESACCI-L3C_SNOW-SWE-SSMIS_DMSP-fv4.0.nc"
        urls.append(url)
        
    current_date += datetime.timedelta(days=1)

# If all requested dates were skipped, exit gracefully
if not urls:
    print("\nAll requested dates are already present in the store. Nothing to process!")
    sys.exit(0)

# ==========================================
# 4. Virtualize and Concatenate Datasets
# ==========================================
print(f"\nVirtualizing {len(urls)} missing daily files via obstore HTTPS...")
vds_list = []
for url in urls:
    try:
        vds = open_virtual_dataset(url=url, parser=parser, registry=registry)
        print(f"Successfully mapped: {url}")
        vds_list.append(vds)
    except FileNotFoundError:
        print(f"  -> WARNING: File missing at source (404), skipping: {url}")
    except Exception as e:
        print(f"  -> WARNING: Unexpected error mapping {url}: {e}")

if not vds_list:
    print("\nNo valid files could be downloaded/mapped. Exiting gracefully.")
    sys.exit(0)

print("\nConcatenating virtual datasets along the 'time' dimension...")
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

commit_message = f"{action} {len(vds_list)} daily CEDA SWE records."
session.commit(commit_message)

print(f"\nSuccess! Data committed to Icechunk. Commit message: '{commit_message}'")