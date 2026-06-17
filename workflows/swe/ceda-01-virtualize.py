#!ceda-01-virtualize.py

from pathlib import Path
import os
# Silence icechunk rust warnings. Must be set before importing icechunk.
os.environ.setdefault("RUST_LOG", "error")

from join_scratch.utils.s3 import s3_store_config, list_s3

import obstore
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
from virtualizarr.parsers import HDFParser

import icechunk as ic

import xarray as xr

from matplotlib import pyplot as plt

bucket = "airborne-smce-prod-user-bucket"
region = "us-west-2"
src_bucket_url = f"s3://{bucket}"
dst_bucket_url = f"s3://{bucket}"

src_store = obstore.store.S3Store(bucket=bucket, config=s3_store_config())
registry = ObjectStoreRegistry({src_bucket_url: src_store})

# Get list of CEDA files
ceda_files = sorted(f for f in list_s3(src_store, "JOIN/CEDA") if f.endswith("nc"))

# Build virtual zarr store of all the CEDA data
parser = HDFParser()
ceda_urls = [f"{src_bucket_url}/{f}" for f in ceda_files]
vds_all = open_virtual_mfdataset(ceda_urls, parser=parser, registry=registry)

# Create icechunk store
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
    ic.storage.s3_store(region="us-west-2"),
    name = "ceda-s3"
))
credentials = ic.credentials.containers_credentials(
    {dst_prefix_url: ic.credentials.s3_credentials(from_env=True)}
)

repo = ic.Repository.open_or_create(dst_store, config, credentials)
repo.save_config()

session = repo.writable_session("main")
vds_all.vz.to_icechunk(session.store)
commit_id = session.commit("Virtualize CEDA data in airborne-smce-prod-user-bucket")
print(f"Committed: {commit_id}")

################################################################################
# Test reading the data
repo_ro = ic.Repository.open(
    dst_store,
    authorize_virtual_chunk_access=credentials
)
session_ro = repo_ro.readonly_session("main")

ds = xr.open_zarr(session_ro.store)

fig, ax = plt.subplots()
ds["swe"].sel(time="2019-01-15").plot(x="lon", y="lat", ax=ax)
fig.savefig(Path("~/snowmap.png").expanduser(), bbox_inches="tight", dpi=300)

fig, ax = plt.subplots()
ds["swe"].sel(lon=-93.63, lat=46.80, method="nearest").plot(x="time", ax=ax)
fig.savefig(Path("~/timeseries.png").expanduser(), bbox_inches="tight", dpi=300)
