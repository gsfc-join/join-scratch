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
# # GPM DPR — Virtual IceChunk Store (All Granules)
#
# Builds an IceChunk store on S3 with virtual references to **all** available
# GPM 2A-DPR V07C granules in
# `s3://airborne-smce-prod-user-bucket/JOIN/GPM-DPR/`.
#
# The store targets the `FS` (Full Scan, Ku+Ka, 49 rays) swath group and
# assembles variables from five shape families within `FS/SLV`:
#
# | Shape family | Variables | Extra dim |
# |---|---|---|
# | `(nscan, nray, nbin)` | `precipRate`, `precipWater`, … | — |
# | `(nscan, nray)` | `precipRateNearSurface`, `precipRateESurface`, … | — |
# | `(nscan, nray, nbin, nDSD=2)` | `paramDSD`, `zFactorFinal` | `nDSD` |
# | `(nscan, nray, nfreq=2)` | `piaFinal`, `zFactorFinalNearSurface`, … | `nfreq` |
# | `(nscan, nray, LS=2)` | `precipWaterIntegrated` | `LS` |
# | `(nscan, nray, nNUBF=3)` | `paramNUBF` | `nNUBF` |
#
# **Store location:**
# `s3://airborne-smce-prod-user-bucket/JOIN/icechunk-stores/GPM-DPR-subset`

# %% [markdown]
# ## Setup

# %%
import os
os.environ.setdefault("RUST_LOG", "error")

import io
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
DPR_PREFIX = "JOIN/GPM-DPR"

STORE_BUCKET = S3_BUCKET
STORE_PREFIX = "JOIN/icechunk-stores/GPM-DPR-subset"

# %% [markdown]
# ## Discover granules

# %%
s3_store = S3Store.from_url(f"s3://{S3_BUCKET}/", region=S3_REGION)
registry  = ObjectStoreRegistry({f"s3://{S3_BUCKET}/": s3_store})


def list_s3_keys(prefix: str) -> list[str]:
    keys = []
    for batch in obstore.list(s3_store, prefix=prefix):
        for item in batch:
            keys.append(item["path"])
    return sorted(keys)


dpr_keys = list_s3_keys(DPR_PREFIX)
print(f"GPM-DPR granules : {len(dpr_keys)}")
print(f"First : {dpr_keys[0].split('/')[-1]}")
print(f"Last  : {dpr_keys[-1].split('/')[-1]}")

# %% [markdown]
# ## Helper functions

# %%
def read_dpr_coords(key: str) -> dict:
    """Eagerly read time (datetime64) and 2-D lat/lon from an FS swath granule."""
    buf = io.BytesIO(obstore.get(s3_store, key).bytes())
    with h5py.File(buf, "r") as f:
        st  = f["FS/ScanTime"]
        yr  = st["Year"][:]
        mo  = st["Month"][:]
        dy  = st["DayOfMonth"][:]
        hr  = st["Hour"][:]
        mi  = st["Minute"][:]
        sc  = st["Second"][:]
        ms  = st["MilliSecond"][:]
        lat = f["FS/Latitude"][:]
        lon = f["FS/Longitude"][:]

    iso = [
        f"{int(y):04d}-{int(m):02d}-{int(d):02d}T"
        f"{int(h):02d}:{int(n):02d}:{int(s):02d}.{int(x):03d}"
        for y, m, d, h, n, s, x in zip(yr, mo, dy, hr, mi, sc, ms)
    ]
    return {
        "time":      np.array(iso, dtype="datetime64[ms]"),
        "latitude":  lat,
        "longitude": lon,
    }


# Variable families in FS/SLV grouped by shape
VARS_3D      = ["DFRforward1", "epsilon", "flagSLV", "precipRate", "precipWater"]
VARS_2D      = ["binEchoBottom", "phaseNearSurface", "precipRateAve24",
                "precipRateESurface", "precipRateNearSurface", "qualitySLV"]
VARS_4D_DSD  = ["paramDSD", "zFactorFinal"]
VARS_3D_FREQ = ["piaFinal", "piaOffset", "sigmaZeroCorrected",
                "zFactorFinalESurface", "zFactorFinalNearSurface"]
VARS_3D_LS   = ["precipWaterIntegrated"]
VARS_3D_NUBF = ["paramNUBF"]
ALL_VARS     = (VARS_3D + VARS_2D + VARS_4D_DSD +
                VARS_3D_FREQ + VARS_3D_LS + VARS_3D_NUBF)


def make_dpr_vds(key: str) -> xr.Dataset:
    """Virtual dataset for one 2A-DPR FS granule."""
    url = f"s3://{S3_BUCKET}/{key}"
    parts = []

    # (nscan, nray, nbin) and (nscan, nray)
    drop = [v for v in ALL_VARS if v not in VARS_3D + VARS_2D]
    v = HDFParser(group="FS/SLV", drop_variables=drop)(url, registry=registry).to_virtual_dataset()
    parts.append(v.rename({"phony_dim_0": "nscan", "phony_dim_1": "nray",
                            "phony_dim_2": "nbin"}))

    # (nscan, nray, nbin, nDSD)
    drop = [v for v in ALL_VARS if v not in VARS_4D_DSD]
    v = HDFParser(group="FS/SLV", drop_variables=drop)(url, registry=registry).to_virtual_dataset()
    parts.append(v.rename({"phony_dim_0": "nscan", "phony_dim_1": "nray",
                            "phony_dim_2": "nbin",  "phony_dim_3": "nDSD"}))

    # (nscan, nray, nfreq)
    drop = [v for v in ALL_VARS if v not in VARS_3D_FREQ]
    v = HDFParser(group="FS/SLV", drop_variables=drop)(url, registry=registry).to_virtual_dataset()
    parts.append(v.rename({"phony_dim_0": "nscan", "phony_dim_1": "nray",
                            "phony_dim_2": "nfreq"}))

    # (nscan, nray, LS)
    drop = [v for v in ALL_VARS if v not in VARS_3D_LS]
    v = HDFParser(group="FS/SLV", drop_variables=drop)(url, registry=registry).to_virtual_dataset()
    parts.append(v.rename({"phony_dim_0": "nscan", "phony_dim_1": "nray",
                            "phony_dim_2": "LS"}))

    # (nscan, nray, nNUBF)
    drop = [v for v in ALL_VARS if v not in VARS_3D_NUBF]
    v = HDFParser(group="FS/SLV", drop_variables=drop)(url, registry=registry).to_virtual_dataset()
    parts.append(v.rename({"phony_dim_0": "nscan", "phony_dim_1": "nray",
                            "phony_dim_2": "nNUBF"}))

    vds    = xr.merge(parts)
    coords = read_dpr_coords(key)
    return (
        vds
        .assign_coords(
            time=("nscan", coords["time"],
                  {"long_name": "scan time", "timezone": "UTC"}),
        )
        .assign(
            latitude =xr.DataArray(coords["latitude"],  dims=["nscan", "nray"],
                                   attrs={"long_name": "latitude",
                                          "units": "degrees_north"}),
            longitude=xr.DataArray(coords["longitude"], dims=["nscan", "nray"],
                                   attrs={"long_name": "longitude",
                                          "units": "degrees_east"}),
        )
        .set_index(nscan="time")
    )

# %% [markdown]
# ## Build virtual datasets for all granules

# %%
print(f"Building GPM-DPR virtual datasets ({len(dpr_keys)} granules)…")
dpr_vds_list = []
for key in dpr_keys:
    vds = make_dpr_vds(key)
    dpr_vds_list.append(vds)
    print(f"  {key.split('/')[-1]}: {dict(vds.sizes)}")

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
# ## Write granules (append along `nscan`)

# %%
print(f"Writing {len(dpr_keys)} granules…")
for i, (key, vds) in enumerate(zip(dpr_keys, dpr_vds_list)):
    session = repo.writable_session("main")
    kwargs  = {}
    if i > 0:
        kwargs["append_dim"] = "nscan"
    virtual_dataset_to_icechunk(vds, session.store, **kwargs)
    session.commit(f"granule {i+1}/{len(dpr_keys)}: {key.split('/')[-1]}")
    if (i + 1) % 5 == 0 or i == len(dpr_keys) - 1:
        print(f"  {i+1}/{len(dpr_keys)} committed")
print("Done.")

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
ds = xr.open_zarr(
    repo2.readonly_session("main").store,
    consolidated=False, mask_and_scale=True,
)
print(ds)
print(f"\nTotal scans  : {ds.sizes['nscan']:,}")
print(f"Time range   : {str(ds.nscan.values[0])[:19]} — {str(ds.nscan.values[-1])[:19]}")

# %%
# Cross-granule time-based selection test
window = ds.sel(nscan=slice("2026-01-19T03:40", "2026-01-19T03:55"))
print(f"\nScans in 03:40–03:55 UTC window : {window.sizes['nscan']}")
print(window[["latitude", "longitude", "precipRateNearSurface"]])

# %% [markdown]
# ## Visualizations

# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ### Near-surface precipitation rate (nadir ray, full dataset)

# %%
prate = ds["precipRateNearSurface"].isel(nray=24).values
lat   = ds["latitude"].isel(nray=24).values
prate = np.where(np.isfinite(prate) & (prate > 0), prate, np.nan)

fig, ax = plt.subplots(figsize=(13, 3))
sc = ax.scatter(lat, prate, c=prate, cmap="Blues", s=1,
                norm=mcolors.LogNorm(vmin=0.1, vmax=50))
plt.colorbar(sc, ax=ax, label="Near-surface precip rate (mm/hr)")
ax.set_xlabel("Latitude (°N)")
ax.set_ylabel("Precip rate (mm/hr)")
ax.set_title(
    f"GPM 2A-DPR FS — Near-Surface Precipitation Rate (nadir ray)\n"
    f"{len(dpr_keys)} granules · "
    f"{str(ds.nscan.values[0])[:10]} – {str(ds.nscan.values[-1])[:10]}"
)
ax.set_ylim(0, 60)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Precipitation rate vertical curtain (first granule, nadir ray)

# %%
t0   = str(ds.nscan.values[0])[:19]
t1   = str(ds.nscan.values[7987])[:19]  # ~7988 scans per granule
gran0 = ds.sel(nscan=slice(t0, t1))
pr    = gran0["precipRate"].isel(nray=24).values
lat0  = gran0["latitude"].isel(nray=24).values
pr    = np.where(np.isfinite(pr) & (pr > 0), pr, np.nan)
height_km = np.linspace(22, 0, pr.shape[1])

fig, ax = plt.subplots(figsize=(14, 4))
pcm = ax.pcolormesh(lat0, height_km, pr.T,
                    norm=mcolors.LogNorm(vmin=0.1, vmax=50),
                    cmap="Blues", shading="auto")
plt.colorbar(pcm, ax=ax, label="Precipitation rate (mm/hr)", pad=0.01)
ax.set_xlabel("Latitude (°N)")
ax.set_ylabel("Altitude (km)")
ax.set_title(
    f"GPM 2A-DPR FS — Precipitation Rate Profile (nadir ray)\n"
    f"Granule 1 · {t0[:10]}"
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Full-swath near-surface precip — 30-minute cross-track segment

# %%
# Use a 30-minute segment from a granule near the middle of the dataset
mid_t = ds.nscan.values[len(ds.nscan) // 2]
t_start = str(mid_t)[:16]
t_end   = str(mid_t + np.timedelta64(30, "m"))[:16]
seg = ds.sel(nscan=slice(t_start, t_end))
prs  = seg["precipRateNearSurface"].values
lats = seg["latitude"].values
lons = seg["longitude"].values
prs  = np.where(np.isfinite(prs) & (prs > 0), prs, np.nan)

fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(lons.ravel(), lats.ravel(), c=prs.ravel(), s=1,
                cmap="Blues", norm=mcolors.LogNorm(vmin=0.1, vmax=50))
plt.colorbar(sc, ax=ax, label="Near-surface precip rate (mm/hr)")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(
    f"GPM 2A-DPR FS — Full-Swath Near-Surface Precip Rate\n"
    f"{t_start} – {t_end} UTC"
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Full-dataset temporal coverage

# %%
times = ds.nscan.values
fig, ax = plt.subplots(figsize=(14, 1.5))
ax.scatter(times[::20], np.ones(len(times[::20])),
           s=2, c="steelblue", alpha=0.5)
ax.set_yticks([])
ax.set_xlabel("Time (UTC)")
ax.set_title(
    f"GPM-DPR IceChunk Store — Temporal Coverage\n"
    f"{len(dpr_keys)} granules · "
    f"{str(times[0])[:10]} – {str(times[-1])[:10]}"
)
ax.grid(True, alpha=0.3, axis="x")
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

print(f"\nStore: s3://{STORE_BUCKET}/{STORE_PREFIX}")
print(f"  Total nscan : {ds.sizes['nscan']:,}")
print(f"  nray        : {ds.sizes['nray']}")
print(f"  nbin        : {ds.sizes['nbin']}")
