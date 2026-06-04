"""
gpmdpr_virtualize.py — Build an IceChunk virtual store for GPM 2A-DPR data.

Usage
-----
  python gpmdpr_virtualize.py --execution-type local-test
  python gpmdpr_virtualize.py --execution-type prod

Execution types
---------------
  local-test : virtualizes --n-granules granules (default 3) from the first
               available day and writes to a temporary S3 prefix that is
               deleted at the end.  Intended for container smoke-tests.
  prod       : virtualizes every available 2A-DPR granule and writes to the
               permanent store prefix.
"""

import argparse
import io
import logging
import os
import time
import uuid

import h5py
import icechunk
import numpy as np
import obstore
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import S3Store
from virtualizarr.parsers import HDFParser
from virtualizarr.writers.icechunk import virtual_dataset_to_icechunk

os.environ.setdefault("RUST_LOG", "error")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_S3_BUCKET    = "airborne-smce-prod-user-bucket"
DEFAULT_S3_REGION    = "us-west-2"
DEFAULT_DPR_PREFIX   = "JOIN/GPM-DPR"
DEFAULT_STORE_PREFIX = "JOIN/icechunk-stores/GPM-DPR-subset"
DEFAULT_N_TEST       = 3

# FS/SLV variable families, grouped by shape
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--execution-type", required=True,
        choices=["local-test", "prod"],
        help="'local-test': small subset + temporary store (deleted after). "
             "'prod': all granules + permanent store.",
    )
    p.add_argument("--s3-bucket",    default=DEFAULT_S3_BUCKET)
    p.add_argument("--s3-region",    default=DEFAULT_S3_REGION)
    p.add_argument("--dpr-prefix",   default=DEFAULT_DPR_PREFIX,
                   help="S3 key prefix for 2A-DPR granules")
    p.add_argument("--store-bucket", default=None,
                   help="S3 bucket for the IceChunk store (default: same as --s3-bucket)")
    p.add_argument("--store-prefix", default=DEFAULT_STORE_PREFIX,
                   help="S3 key prefix for the IceChunk store")
    p.add_argument("--n-granules",   type=int, default=DEFAULT_N_TEST,
                   help="Number of granules to process in local-test mode (default: %(default)s)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def list_s3_keys(s3_store, prefix: str) -> list[str]:
    keys = []
    for batch in obstore.list(s3_store, prefix=prefix):
        for item in batch:
            keys.append(item["path"])
    return sorted(keys)


def delete_s3_prefix(s3_store, prefix: str) -> None:
    keys = []
    for batch in obstore.list(s3_store, prefix=prefix):
        for item in batch:
            keys.append(item["path"])
    if keys:
        log.info("Deleting %d objects under s3://.../%s", len(keys), prefix)
        obstore.delete(s3_store, keys)


# ---------------------------------------------------------------------------
# Coordinate reader
# ---------------------------------------------------------------------------
def read_dpr_coords(s3_store, key: str) -> dict:
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


# ---------------------------------------------------------------------------
# Virtual dataset builder
# ---------------------------------------------------------------------------
def make_dpr_vds(s3_store, registry, bucket: str, key: str) -> xr.Dataset:
    url   = f"s3://{bucket}/{key}"
    parts = []

    # (nscan, nray, nbin) + (nscan, nray)
    drop = [v for v in ALL_VARS if v not in VARS_3D + VARS_2D]
    v = (HDFParser(group="FS/SLV", drop_variables=drop)(url, registry=registry)
         .to_virtual_dataset()
         .rename({"phony_dim_0": "nscan", "phony_dim_1": "nray",
                  "phony_dim_2": "nbin"}))
    parts.append(v)

    # (nscan, nray, nbin, nDSD)
    drop = [v for v in ALL_VARS if v not in VARS_4D_DSD]
    v = (HDFParser(group="FS/SLV", drop_variables=drop)(url, registry=registry)
         .to_virtual_dataset()
         .rename({"phony_dim_0": "nscan", "phony_dim_1": "nray",
                  "phony_dim_2": "nbin",  "phony_dim_3": "nDSD"}))
    parts.append(v)

    # (nscan, nray, nfreq)
    drop = [v for v in ALL_VARS if v not in VARS_3D_FREQ]
    v = (HDFParser(group="FS/SLV", drop_variables=drop)(url, registry=registry)
         .to_virtual_dataset()
         .rename({"phony_dim_0": "nscan", "phony_dim_1": "nray",
                  "phony_dim_2": "nfreq"}))
    parts.append(v)

    # (nscan, nray, LS)
    drop = [v for v in ALL_VARS if v not in VARS_3D_LS]
    v = (HDFParser(group="FS/SLV", drop_variables=drop)(url, registry=registry)
         .to_virtual_dataset()
         .rename({"phony_dim_0": "nscan", "phony_dim_1": "nray",
                  "phony_dim_2": "LS"}))
    parts.append(v)

    # (nscan, nray, nNUBF)
    drop = [v for v in ALL_VARS if v not in VARS_3D_NUBF]
    v = (HDFParser(group="FS/SLV", drop_variables=drop)(url, registry=registry)
         .to_virtual_dataset()
         .rename({"phony_dim_0": "nscan", "phony_dim_1": "nray",
                  "phony_dim_2": "nNUBF"}))
    parts.append(v)

    vds    = xr.merge(parts)
    coords = read_dpr_coords(s3_store, key)
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


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------
def make_icechunk_config(bucket: str, region: str) -> icechunk.RepositoryConfig:
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            f"s3://{bucket}/",
            icechunk.s3_store(region=region),
        )
    )
    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    store_bucket = args.store_bucket or args.s3_bucket

    s3       = S3Store.from_url(f"s3://{args.s3_bucket}/", region=args.s3_region)
    registry = ObjectStoreRegistry({f"s3://{args.s3_bucket}/": s3})

    # Determine store prefix (temp for local-test)
    if args.execution_type == "local-test":
        store_prefix = f"JOIN/icechunk-stores/test-gpmdpr-{uuid.uuid4().hex[:8]}"
        log.info("[local-test] Temporary store prefix: %s", store_prefix)
    else:
        store_prefix = args.store_prefix

    # Discover granules
    log.info("Listing DPR keys from s3://%s/%s …", args.s3_bucket, args.dpr_prefix)
    dpr_keys = list_s3_keys(s3, args.dpr_prefix)
    log.info("Found %d granules", len(dpr_keys))

    # Restrict for local-test
    if args.execution_type == "local-test":
        dpr_keys = dpr_keys[: args.n_granules]
        log.info("[local-test] Processing %d granule(s)", len(dpr_keys))
        for k in dpr_keys:
            log.info("  %s", k.split("/")[-1])

    # Build virtual datasets
    log.info("Building virtual datasets …")
    vds_list = []
    for key in dpr_keys:
        vds = make_dpr_vds(s3, registry, args.s3_bucket, key)
        vds_list.append(vds)
        log.info("  %s: %s", key.split("/")[-1], dict(vds.sizes))

    # Create IceChunk store
    config  = make_icechunk_config(args.s3_bucket, args.s3_region)
    storage = icechunk.s3_storage(
        bucket=store_bucket, prefix=store_prefix,
        region=args.s3_region, from_env=True,
    )
    repo = icechunk.Repository.open_or_create(storage, config=config)
    log.info("IceChunk store: s3://%s/%s", store_bucket, store_prefix)

    def find_resume_idx():
        """Return (start_idx, already_has_data) for the root group / nscan dim."""
        try:
            ro = repo.readonly_session("main")
            ds = xr.open_zarr(ro.store, consolidated=False, mask_and_scale=False)
            current_size = ds.sizes["nscan"]
        except Exception:
            return 0, False

        cumulative = 0
        for idx, vds in enumerate(vds_list):
            cumulative += vds.sizes["nscan"]
            if cumulative >= current_size:
                log.info(
                    "Resuming from granule %d/%d (store has %d nscan, expected %d after granule %d)",
                    idx + 1, len(vds_list), current_size, cumulative, idx,
                )
                return idx, True
        log.info("Store already complete (%d nscan) — skipping writes.", current_size)
        return len(vds_list), True

    try:
        log.info("Writing %d granules …", len(dpr_keys))
        t0 = time.monotonic()
        start_i, has_data = find_resume_idx()
        for i, (key, vds) in enumerate(zip(dpr_keys, vds_list)):
            if i < start_i:
                continue
            session = repo.writable_session("main")
            kwargs  = {}
            if i > 0 or has_data:
                kwargs["append_dim"] = "nscan"
            virtual_dataset_to_icechunk(vds, session.store, **kwargs)
            session.commit(f"granule {i+1}/{len(dpr_keys)}: {key.split('/')[-1]}")
            if (i + 1) % 5 == 0 or i == len(dpr_keys) - 1:
                log.info("  %d/%d committed", i + 1, len(dpr_keys))
        log.info("All granules written in %.1f s", time.monotonic() - t0)

        # Verify store is readable
        log.info("Verifying store …")
        repo2 = icechunk.Repository.open(
            icechunk.s3_storage(
                bucket=store_bucket, prefix=store_prefix,
                region=args.s3_region, from_env=True,
            ),
            authorize_virtual_chunk_access={f"s3://{args.s3_bucket}/": None},
        )
        ro = repo2.readonly_session("main")
        ds  = xr.open_zarr(ro.store, consolidated=False, mask_and_scale=True)
        log.info("nscan: %d", ds.sizes["nscan"])
        log.info("Time range: %s – %s",
                 str(ds.nscan.values[0])[:19], str(ds.nscan.values[-1])[:19])
        log.info("Store verified OK.")

    finally:
        if args.execution_type == "local-test":
            log.info("[local-test] Cleaning up temporary store …")
            delete_s3_prefix(s3, store_prefix)
            log.info("[local-test] Cleanup complete.")

    log.info("Done. Store: s3://%s/%s", store_bucket, store_prefix)


if __name__ == "__main__":
    main()
