import xarray as xr
import icechunk
import fsspec
from kerchunk.hdf import SingleHdf5ToZarr
from datetime import datetime

# 1. Define the G-Portal target
date_obj = datetime(2023, 1, 1)
yyyy = date_obj.strftime("%Y")
mm = date_obj.strftime("%m")
date_str = date_obj.strftime("%Y%m%d")
orbit = "A"

url = f"https://gportal.jaxa.jp/download/standard/GCOM-W/GCOM-W.AMSR2/L3.SND_10/2/{yyyy}/{mm}/GW1AM2_{date_str}_01D_{orbit}_L3SGSNDHG2210210.h5"

print(f"Fetching HTTP byte-ranges via Kerchunk for: {url}")

# 2. Use Kerchunk to read the HDF5 metadata over HTTP
#    This reads just the header/chunk tables without downloading the whole file
fs = fsspec.filesystem("https") # Add {"client_kwargs": {"auth": ...}} if JAXA requires login
with fs.open(url, "rb") as f:
    # Extract the chunk map
    h5_chunks = SingleHdf5ToZarr(f, url, inline_threshold=100)
    refs = h5_chunks.translate()

print("Successfully generated Kerchunk references. Creating virtual dataset...")

# 3. Open as a virtual dataset using Xarray's Kerchunk engine
#    (Your error output showed 'kerchunk' IS an available engine in your environment)
vds = xr.open_dataset(
    "reference://", 
    engine="kerchunk",
    backend_kwargs={
        "storage_options": {
            "fo": refs, 
            "remote_protocol": "https"
        }
    },
    drop_variables=["lat", "lon"], # Optional
    consolidated=False
)

# 4. Initialize Icechunk Repository
storage_config = icechunk.LocalConfig(path="./amsr2_icechunk_store")
repo = icechunk.Repository.create(storage_config)
session = repo.writable_session("main")

# 5. Write the virtual dataset to the Icechunk store
print("Committing virtual dataset to Icechunk...")
vds.virtualize.to_zarr(
    store=session.store, 
    mode="w"
)

commit_hash = session.commit("Added virtualized AMSR2 file from JAXA G-Portal via Kerchunk")
print(f"Committed to Icechunk: {commit_hash}")

# 6. Verify access
read_session = repo.readonly_session(commit_hash)
ds = xr.open_zarr(read_session.store, consolidated=False)
print("Success! Virtual dataset loaded:")
print(ds)