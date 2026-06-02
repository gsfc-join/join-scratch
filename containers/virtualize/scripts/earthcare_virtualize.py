"""
earthcare_virtualize.py — Build an IceChunk virtual store for EarthCARE data.

Usage
-----
  python earthcare_virtualize.py --execution-type local-test
  python earthcare_virtualize.py --execution-type prod

Execution types
---------------
  local-test : virtualizes --n-granules granules (default 3) from the first
               available day and writes to a temporary S3 prefix that is
               deleted at the end.  Intended for container smoke-tests.
  prod       : virtualizes every available AC__CLP + AUX_MET granule pair and
               writes to the permanent store prefix.
"""

import argparse
import io
import logging
import os
import re
import sys
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
DEFAULT_CLP_PREFIX   = "JOIN/EarthCARE/AC__CLP"
DEFAULT_MET_PREFIX   = "JOIN/EarthCARE/AUX_MET"
DEFAULT_STORE_PREFIX = "JOIN/icechunk-stores/EarthCARE-subset"
DEFAULT_N_TEST       = 3

EC_EPOCH  = np.datetime64("2000-01-01T00:00:00", "ns")
EC_MISSING = -9999.0
ORBIT_RE   = re.compile(r"_(\d{5}[BD])_")


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
    p.add_argument("--clp-prefix",   default=DEFAULT_CLP_PREFIX,
                   help="S3 key prefix for AC__CLP granules")
    p.add_argument("--met-prefix",   default=DEFAULT_MET_PREFIX,
                   help="S3 key prefix for AUX_MET granules")
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


def orbit_id(key: str) -> str:
    m = ORBIT_RE.search(key)
    if not m:
        raise ValueError(f"Cannot extract orbit id from: {key}")
    return m.group(1)


# ---------------------------------------------------------------------------
# Coordinate readers
# ---------------------------------------------------------------------------
def read_clp_coords(s3_store, key: str) -> dict:
    buf = io.BytesIO(obstore.get(s3_store, key).bytes())
    with h5py.File(buf, "r") as f:
        ts  = f["ScienceData/Geo/time"][:]
        lat = f["ScienceData/Geo/latitude"][:]
        lon = f["ScienceData/Geo/longitude"][:]
    t = EC_EPOCH + (ts * 1e9).astype("int64").astype("timedelta64[ns]")
    return {"time": t, "latitude": lat, "longitude": lon}


def read_met_time(s3_store, key: str) -> np.ndarray:
    buf = io.BytesIO(obstore.get(s3_store, key).bytes())
    with h5py.File(buf, "r") as f:
        ts = f["ScienceData/time"][:]
    return EC_EPOCH + (ts * 1e9).astype("int64").astype("timedelta64[ns]")


# ---------------------------------------------------------------------------
# Virtual dataset builders
# ---------------------------------------------------------------------------
def add_missing_value(vds: xr.Dataset) -> xr.Dataset:
    return vds.assign({
        n: v.assign_attrs(dict(v.attrs) | {"missing_value": EC_MISSING})
        for n, v in vds.data_vars.items()
        if v.dtype.kind == "f" and "missing_value" not in v.attrs
    })


def make_clp_vds(s3_store, registry, bucket: str, key: str) -> xr.Dataset:
    url = f"s3://{bucket}/{key}"
    geo_p  = HDFParser(group="ScienceData/Geo")
    data_p = HDFParser(group="ScienceData/Data")
    vg = geo_p(url,  registry=registry).to_virtual_dataset()
    vd = data_p(url, registry=registry).to_virtual_dataset()
    vg = vg.rename({"phony_dim_0": "nalong", "phony_dim_1": "nheight"})
    vd = vd.rename({"phony_dim_0": "nalong", "phony_dim_1": "nheight",
                    "phony_dim_2": "nqflag"})
    vds = add_missing_value(xr.merge([vg, vd]))
    c = read_clp_coords(s3_store, key)
    return (
        vds
        .assign_coords(
            time     =("nalong", c["time"],
                        {"long_name": "observation time", "timezone": "UTC"}),
            latitude =("nalong", c["latitude"],
                        {"long_name": "latitude",  "units": "degrees_north"}),
            longitude=("nalong", c["longitude"],
                        {"long_name": "longitude", "units": "degrees_east"}),
        )
        .set_index(nalong="time")
    )


def make_met_vds(s3_store, registry, bucket: str, key: str) -> xr.Dataset:
    url  = f"s3://{bucket}/{key}"
    met_p = HDFParser(group="ScienceData")
    vds  = met_p(url, registry=registry).to_virtual_dataset()
    t    = read_met_time(s3_store, key)
    return (
        vds
        .assign_coords(
            time=("along_track", t,
                  {"long_name": "observation time", "timezone": "UTC"}),
        )
        .set_index(along_track="time")
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


def delete_s3_prefix(s3_store, prefix: str) -> None:
    """Delete all objects under *prefix* (best-effort cleanup)."""
    keys = []
    for batch in obstore.list(s3_store, prefix=prefix):
        for item in batch:
            keys.append(item["path"])
    if keys:
        log.info("Deleting %d objects under s3://.../%s", len(keys), prefix)
        obstore.delete(s3_store, keys)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    store_bucket = args.store_bucket or args.s3_bucket

    s3 = S3Store.from_url(f"s3://{args.s3_bucket}/", region=args.s3_region)
    registry = ObjectStoreRegistry({f"s3://{args.s3_bucket}/": s3})

    # Determine store prefix (temp for local-test)
    if args.execution_type == "local-test":
        store_prefix = f"JOIN/icechunk-stores/test-earthcare-{uuid.uuid4().hex[:8]}"
        log.info("[local-test] Temporary store prefix: %s", store_prefix)
    else:
        store_prefix = args.store_prefix

    # Discover and match granules
    log.info("Listing AC__CLP keys from s3://%s/%s …", args.s3_bucket, args.clp_prefix)
    clp_keys = list_s3_keys(s3, args.clp_prefix)

    clp_by_orbit = {orbit_id(k): k for k in clp_keys}
    common_orbits = sorted(clp_by_orbit)
    log.info("Found %d AC__CLP granules", len(common_orbits))

    # Restrict for local-test
    if args.execution_type == "local-test":
        common_orbits = common_orbits[: args.n_granules]
        log.info("[local-test] Processing %d granule(s): %s",
                 len(common_orbits), common_orbits)

    # Build virtual datasets
    log.info("Building AC__CLP virtual datasets …")
    clp_vds_list = []
    for orbit in common_orbits:
        vds = make_clp_vds(s3, registry, args.s3_bucket, clp_by_orbit[orbit])
        clp_vds_list.append(vds)
        log.info("  %s: nalong=%d, nheight=%d", orbit,
                 vds.sizes["nalong"], vds.sizes["nheight"])

    # Create IceChunk store
    config  = make_icechunk_config(args.s3_bucket, args.s3_region)
    storage = icechunk.s3_storage(
        bucket=store_bucket, prefix=store_prefix,
        region=args.s3_region, from_env=True,
    )
    repo = icechunk.Repository.open_or_create(storage, config=config)
    log.info("IceChunk store: s3://%s/%s", store_bucket, store_prefix)

    def find_resume_idx(group_name, dim_name, vds_list):
        """Return (start_idx, already_has_data).

        Reads the current dim size from the store and compares it to the
        cumulative sizes of vds_list to find where to resume.  Returns
        (0, False) when the group does not yet exist.
        """
        try:
            ro = repo.readonly_session("main")
            ds = xr.open_zarr(ro.store, group=group_name,
                               consolidated=False, mask_and_scale=False)
            current_size = ds.sizes[dim_name]
        except Exception:
            return 0, False

        cumulative = 0
        for idx, vds in enumerate(vds_list):
            cumulative += vds.sizes[dim_name]
            if cumulative >= current_size:
                already = idx  # granules 0..idx-1 are done
                log.info(
                    "Resuming %s from granule %d/%d (store has %d %s, expected %d after granule %d)",
                    group_name, already + 1, len(vds_list),
                    current_size, dim_name, cumulative, idx,
                )
                return already, True
        # All granules already written
        log.info("%s already complete (%d %s) — skipping.", group_name, current_size, dim_name)
        return len(vds_list), True

    try:
        # Write ac_clp group
        log.info("Writing ac_clp (%d granules) …", len(common_orbits))
        t0 = time.monotonic()
        start_i, has_data = find_resume_idx("ac_clp", "nalong", clp_vds_list)
        for i, (orbit, vds) in enumerate(zip(common_orbits, clp_vds_list)):
            if i < start_i:
                continue
            session = repo.writable_session("main")
            kwargs  = dict(group="ac_clp")
            if i > 0 or has_data:
                kwargs["append_dim"] = "nalong"
            virtual_dataset_to_icechunk(vds, session.store, **kwargs)
            session.commit(f"ac_clp {orbit} ({i+1}/{len(common_orbits)})")
            if (i + 1) % 10 == 0 or i == len(common_orbits) - 1:
                log.info("  ac_clp: %d/%d committed", i + 1, len(common_orbits))
        log.info("ac_clp done in %.1f s", time.monotonic() - t0)

        # NOTE: aux_met (AUX_MET) virtualization is skipped.  VirtualiZarr's
        # ManifestArray concatenation requires uniform HDF5 chunk shapes across
        # granules, but EarthCARE AUX_MET files have variable internal chunk
        # sizes (e.g. 2572 vs 2573 along along_track) which triggers
        # check_same_chunk_shapes errors.  Only ac_clp (cloud profiles) is
        # written to this store.
        log.info("aux_met skipped (incompatible HDF5 chunk shapes across granules).")

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
        ds_clp = xr.open_zarr(ro.store, group="ac_clp",
                               consolidated=False, mask_and_scale=True)
        log.info("ac_clp  nalong       : %d", ds_clp.sizes["nalong"])
        log.info("Store verified OK.")

    finally:
        if args.execution_type == "local-test":
            log.info("[local-test] Cleaning up temporary store …")
            delete_s3_prefix(s3, store_prefix)
            log.info("[local-test] Cleanup complete.")

    log.info("Done. Store: s3://%s/%s", store_bucket, store_prefix)


if __name__ == "__main__":
    main()
