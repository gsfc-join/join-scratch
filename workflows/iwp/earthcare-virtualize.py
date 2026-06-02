# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # EarthCARE — Virtual IceChunk Store (All Granules)
#
# Builds an IceChunk store on S3 with virtual references to **all** available
# EarthCARE granules in `s3://airborne-smce-prod-user-bucket/JOIN/EarthCARE/`.
#
# Two product types are stored as separate Zarr groups within one IceChunk
# repository:
#
# | Zarr group | Source product | Along-track dim | Key dimensions |
# |---|---|---|---|
# | `ac_clp` | `AC__CLP_2BS` | `nalong` | `nalong × nheight (206)` |
# | `aux_met` | `AUX_JSG_1DS` | `along_track` | `along_track × across_track (207) × height (242)` |
#
# **Store location:**
# `s3://airborne-smce-prod-user-bucket/JOIN/icechunk-stores/EarthCARE-subset`
#
# No science data is downloaded — only chunk byte-range metadata is written.
# Coordinates (time, lat, lon) are read eagerly once per granule.

# %% [markdown]
# ## Setup

# %%
import os
os.environ.setdefault("RUST_LOG", "error")

import io
import re
import h5py
import icechunk
import numpy as np
import xarray as xr
import obstore
from obstore.store import S3Store
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.parsers import HDFParser
from virtualizarr.writers.icechunk import virtual_dataset_to_icechunk

# %%
S3_BUCKET  = "airborne-smce-prod-user-bucket"
S3_REGION  = "us-west-2"

CLP_PREFIX = "JOIN/EarthCARE/AC__CLP"
MET_PREFIX = "JOIN/EarthCARE/AUX_MET"

STORE_BUCKET = S3_BUCKET
STORE_PREFIX = "JOIN/icechunk-stores/EarthCARE-subset"

# EarthCARE time epoch: TAI seconds since 2000-01-01 00:00:00 UTC
EC_EPOCH = np.datetime64("2000-01-01T00:00:00", "ns")

# %% [markdown]
# ## Discover granules

# %%
s3_store = S3Store.from_url(f"s3://{S3_BUCKET}/", region=S3_REGION)
registry  = ObjectStoreRegistry({f"s3://{S3_BUCKET}/": s3_store})


def list_s3_keys(prefix: str) -> list[str]:
    """Return sorted list of S3 object keys under *prefix*."""
    keys = []
    for batch in obstore.list(s3_store, prefix=prefix):
        for item in batch:
            keys.append(item["path"])
    return sorted(keys)


clp_keys = list_s3_keys(CLP_PREFIX)
met_keys = list_s3_keys(MET_PREFIX)
print(f"AC__CLP granules : {len(clp_keys)}")
print(f"AUX_MET granules : {len(met_keys)}")

# %%
# Match AC__CLP and AUX_MET granules by the shared orbit-id + direction token
# embedded in every EarthCARE filename, e.g. "09417B".
ORBIT_RE = re.compile(r"_(\d{5}[BD])_")


def orbit_id(key: str) -> str:
    m = ORBIT_RE.search(key)
    if not m:
        raise ValueError(f"Cannot extract orbit id from key: {key}")
    return m.group(1)


clp_by_orbit = {orbit_id(k): k for k in clp_keys}
met_by_orbit = {orbit_id(k): k for k in met_keys}
common_orbits = sorted(set(clp_by_orbit) & set(met_by_orbit))

print(f"Matched granule pairs : {len(common_orbits)}")
print(f"First orbit : {common_orbits[0]}  →  {clp_by_orbit[common_orbits[0]].split('/')[-1]}")
print(f"Last orbit  : {common_orbits[-1]}  →  {clp_by_orbit[common_orbits[-1]].split('/')[-1]}")

# %% [markdown]
# ## Helper functions

# %%
def read_ec_coords_clp(key: str) -> dict:
    """Eagerly read time, latitude, longitude from an AC__CLP granule."""
    buf = io.BytesIO(obstore.get(s3_store, key).bytes())
    with h5py.File(buf, "r") as f:
        time_sec = f["ScienceData/Geo/time"][:]
        lat      = f["ScienceData/Geo/latitude"][:]
        lon      = f["ScienceData/Geo/longitude"][:]
    time_dt = EC_EPOCH + (time_sec * 1e9).astype("int64").astype("timedelta64[ns]")
    return {"time": time_dt, "latitude": lat, "longitude": lon}


def read_ec_time_met(key: str) -> np.ndarray:
    """Eagerly read the along-track time array from an AUX_MET granule."""
    buf = io.BytesIO(obstore.get(s3_store, key).bytes())
    with h5py.File(buf, "r") as f:
        time_sec = f["ScienceData/time"][:]
    return EC_EPOCH + (time_sec * 1e9).astype("int64").astype("timedelta64[ns]")


# EarthCARE uses -9999 as a secondary missing-value sentinel in addition to the
# declared _FillValue (9.96921e+36).  Adding missing_value ensures xarray's
# mask_and_scale=True masks both.
EC_MISSING = -9999.0


def add_missing_value(vds: xr.Dataset) -> xr.Dataset:
    new_vars = {
        name: var.assign_attrs(dict(var.attrs) | {"missing_value": EC_MISSING})
        for name, var in vds.data_vars.items()
        if var.dtype.kind == "f" and "missing_value" not in var.attrs
    }
    return vds.assign(new_vars)

# %% [markdown]
# ## Build virtual datasets

# %%
geo_parser  = HDFParser(group="ScienceData/Geo")
data_parser = HDFParser(group="ScienceData/Data")
met_parser  = HDFParser(group="ScienceData")


def make_clp_vds(key: str) -> xr.Dataset:
    """Virtual dataset for one AC__CLP granule."""
    url = f"s3://{S3_BUCKET}/{key}"
    vds_geo  = geo_parser(url,  registry=registry).to_virtual_dataset()
    vds_data = data_parser(url, registry=registry).to_virtual_dataset()

    # Both groups share phony_dim_0 = nalong, phony_dim_1 = nheight;
    # ScienceData/Data adds phony_dim_2 for the 10-element quality flag axis.
    vds_geo  = vds_geo.rename( {"phony_dim_0": "nalong", "phony_dim_1": "nheight"})
    vds_data = vds_data.rename({"phony_dim_0": "nalong", "phony_dim_1": "nheight",
                                 "phony_dim_2": "nqflag"})

    vds = add_missing_value(xr.merge([vds_geo, vds_data]))
    coords = read_ec_coords_clp(key)
    return (
        vds
        .assign_coords(
            time     =("nalong", coords["time"],
                        {"long_name": "observation time", "timezone": "UTC"}),
            latitude =("nalong", coords["latitude"],
                        {"long_name": "latitude",  "units": "degrees_north"}),
            longitude=("nalong", coords["longitude"],
                        {"long_name": "longitude", "units": "degrees_east"}),
        )
        .set_index(nalong="time")
    )


def make_met_vds(key: str) -> xr.Dataset:
    """Virtual dataset for one AUX_MET granule.

    AUX_MET ScienceData already carries named HDF5 dimension scales
    (along_track, across_track, height, radar_times, lidar_times), so no
    phony_dim renaming is needed.  We attach a datetime64 time coordinate on
    along_track and set it as the dimension index.
    """
    url = f"s3://{S3_BUCKET}/{key}"
    vds  = met_parser(url, registry=registry).to_virtual_dataset()
    time = read_ec_time_met(key)
    return (
        vds
        .assign_coords(
            time=("along_track", time,
                  {"long_name": "observation time", "timezone": "UTC"}),
        )
        .set_index(along_track="time")
    )


# %%
print("Building AC__CLP virtual datasets…")
clp_vds_list = []
for orbit in common_orbits:
    vds = make_clp_vds(clp_by_orbit[orbit])
    clp_vds_list.append(vds)
    print(f"  {orbit}: nalong={vds.sizes['nalong']}, nheight={vds.sizes['nheight']}")

# %%
print("Building AUX_MET virtual datasets…")
met_vds_list = []
for orbit in common_orbits:
    vds = make_met_vds(met_by_orbit[orbit])
    met_vds_list.append(vds)
    print(f"  {orbit}: {dict(vds.sizes)}")

# %% [markdown]
# ## Create IceChunk store on S3

# %%
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        f"s3://{S3_BUCKET}/",
        icechunk.s3_store(region=S3_REGION),
    )
)
storage = icechunk.s3_storage(
    bucket=STORE_BUCKET, prefix=STORE_PREFIX,
    region=S3_REGION, from_env=True,
)
repo = icechunk.Repository.open_or_create(storage, config=config)
print(f"IceChunk store: s3://{STORE_BUCKET}/{STORE_PREFIX}")

# %% [markdown]
# ## Write `ac_clp` group

# %%
print(f"Writing ac_clp ({len(common_orbits)} granules)…")
for i, (orbit, vds) in enumerate(zip(common_orbits, clp_vds_list)):
    session = repo.writable_session("main")
    kwargs  = dict(group="ac_clp")
    if i > 0:
        kwargs["append_dim"] = "nalong"
    virtual_dataset_to_icechunk(vds, session.store, **kwargs)
    session.commit(f"ac_clp {orbit} ({i+1}/{len(common_orbits)})")
    if (i + 1) % 10 == 0 or i == len(common_orbits) - 1:
        print(f"  {i+1}/{len(common_orbits)} committed")
print("ac_clp done.")

# %% [markdown]
# ## Write `aux_met` group

# %%
print(f"Writing aux_met ({len(common_orbits)} granules)…")
for i, (orbit, vds) in enumerate(zip(common_orbits, met_vds_list)):
    session = repo.writable_session("main")
    kwargs  = dict(group="aux_met")
    if i > 0:
        kwargs["append_dim"] = "along_track"
    virtual_dataset_to_icechunk(vds, session.store, **kwargs)
    session.commit(f"aux_met {orbit} ({i+1}/{len(common_orbits)})")
    if (i + 1) % 10 == 0 or i == len(common_orbits) - 1:
        print(f"  {i+1}/{len(common_orbits)} committed")
print("aux_met done.")

# %% [markdown]
# ## Re-open and inspect

# %%
repo2 = icechunk.Repository.open(
    icechunk.s3_storage(
        bucket=STORE_BUCKET, prefix=STORE_PREFIX,
        region=S3_REGION, from_env=True,
    ),
    authorize_virtual_chunk_access={f"s3://{S3_BUCKET}/": None},
)
session_ro = repo2.readonly_session("main")

ds_clp = xr.open_zarr(session_ro.store, group="ac_clp",
                       consolidated=False, mask_and_scale=True)
ds_met = xr.open_zarr(session_ro.store, group="aux_met",
                       consolidated=False, mask_and_scale=True)

print("=== ac_clp ===")
print(ds_clp)
print(f"\nTime range : {str(ds_clp.nalong.values[0])[:19]} — {str(ds_clp.nalong.values[-1])[:19]}")
print(f"Total nalong footprints : {ds_clp.sizes['nalong']:,}")

print("\n=== aux_met ===")
print(ds_met)
print(f"\nTime range : {str(ds_met.along_track.values[0])[:19]} — {str(ds_met.along_track.values[-1])[:19]}")
print(f"Total along_track footprints : {ds_met.sizes['along_track']:,}")

# %%
# Time-based selection test
window = ds_clp.sel(nalong=slice("2026-01-24T04:00", "2026-01-24T04:10"))
print(f"\nAC__CLP footprints in 04:00–04:10 UTC window : {window.sizes['nalong']}")

# %% [markdown]
# ## Visualizations

# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ### Radar reflectivity curtain (orbit 09418)

# %%
seg  = ds_clp.sel(nalong=slice("2026-01-24T05:26", "2026-01-24T05:56"))
refl = np.where(np.isfinite(seg["cloud_radar_reflectivity_1km"].values),
                seg["cloud_radar_reflectivity_1km"].values, np.nan)
lat  = seg["latitude"].values
height_km = np.linspace(20, -0.5, refl.shape[1])

fig, ax = plt.subplots(figsize=(14, 4))
pcm = ax.pcolormesh(lat, height_km, refl.T,
                    cmap="RdYlBu_r", vmin=-30, vmax=20, shading="auto")
plt.colorbar(pcm, ax=ax, label="Radar reflectivity (dBZ)", pad=0.01)
ax.set_xlabel("Latitude (°N)")
ax.set_ylabel("Altitude (km)")
ax.set_title("EarthCARE AC__CLP — Attenuated Radar Reflectivity (1 km)\n"
             "Orbit 09418 · 2026-01-24 · ascending pass")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Cloud extinction curtain (ATLID, orbit 09418)

# %%
ext = seg["cloud_extinction_1km"].values
ext = np.clip(np.where(np.isfinite(ext), ext, np.nan), 0, 0.05)

fig, ax = plt.subplots(figsize=(14, 4))
pcm = ax.pcolormesh(lat, height_km, ext.T,
                    cmap="plasma", vmin=0, vmax=0.03, shading="auto")
plt.colorbar(pcm, ax=ax, label="Cloud extinction (/m)", pad=0.01)
ax.set_xlabel("Latitude (°N)")
ax.set_ylabel("Altitude (km)")
ax.set_title("EarthCARE AC__CLP — Cloud Extinction Coefficient (1 km)\n"
             "Orbit 09418 · 2026-01-24 · ascending pass")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Column cloud optical thickness (orbit 09418)

# %%
ot = np.where(np.isfinite(seg["optical_thickness_1km"].values),
              seg["optical_thickness_1km"].values, np.nan)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(lat, ot, color="steelblue", lw=0.8)
ax.set_xlabel("Latitude (°N)")
ax.set_ylabel("Optical thickness")
ax.set_title("EarthCARE AC__CLP — Column Cloud Optical Thickness\nOrbit 09418 · 2026-01-24")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Spacecraft altitude from AUX_MET (orbit 09418)

# %%
seg_met = ds_met.sel(along_track=slice("2026-01-24T05:26", "2026-01-24T05:56"))
if "sensor_altitude" in seg_met:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(seg_met["along_track"].values.astype("datetime64[s]"),
            seg_met["sensor_altitude"].values / 1e3,
            color="darkorange", lw=0.8)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("EarthCARE AUX_MET — Spacecraft Altitude\nOrbit 09418 · 2026-01-24")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Full-dataset temporal coverage

# %%
clp_t = ds_clp.nalong.values
met_t = ds_met.along_track.values

fig, ax = plt.subplots(figsize=(14, 2))
ax.scatter(clp_t[::50], np.ones(len(clp_t[::50])),
           s=2, c="steelblue", label="AC__CLP", alpha=0.6)
ax.scatter(met_t[::50], np.ones(len(met_t[::50])) * 1.4,
           s=2, c="darkorange", label="AUX_MET", alpha=0.6)
ax.set_yticks([1.0, 1.4])
ax.set_yticklabels(["AC__CLP", "AUX_MET"])
ax.set_xlabel("Time (UTC)")
ax.set_title(
    f"EarthCARE IceChunk Store — Temporal Coverage\n"
    f"{len(common_orbits)} orbit segments · "
    f"{str(clp_t[0])[:10]} – {str(clp_t[-1])[:10]}"
)
ax.grid(True, alpha=0.3, axis="x")
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

print(f"\nStore: s3://{STORE_BUCKET}/{STORE_PREFIX}")
print(f"  ac_clp  nalong       : {ds_clp.sizes['nalong']:,}")
print(f"  aux_met along_track  : {ds_met.sizes['along_track']:,}")
