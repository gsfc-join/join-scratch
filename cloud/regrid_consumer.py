#!/usr/bin/env python
"""AMSR2 IceChunk → LIS regrid consumer.

Opens the pre-built IceChunk store and cached ESMF weight file directly from
S3, applies the sparse weight matrix with Dask, and writes a CF-compliant
NetCDF.

Usage
-----
    pixi run -e consumer regrid \\
        --start-date 20180101 \\
        --end-date   20180110 \\
        --output-path s3://airborne-smce-prod-user-bucket/JOIN/outputs/out.nc

All arguments have sensible defaults pointing at the shared S3 resources.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import dask
import dask.array as da
import icechunk
import netCDF4 as nc  # noqa: N813  — reads ESMF weight file
import numpy as np
import pandas as pd
import scipy.sparse
import xarray as xr
import zarr

# Initialise Rust logger before any icechunk usage.
os.environ.setdefault("RUST_LOG", "error")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants (mirror amsr2_regrid.py) ────────────────────────────────────────

S3_BUCKET = "airborne-smce-prod-user-bucket"
S3_JOIN_PREFIX = "JOIN"

DEFAULT_STORE_URI = (
    f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/icechunk-stores/GCOM-W1-AMSR2-L3-SND"
)
DEFAULT_WEIGHTS_URI = (
    f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/cached-weights"
    "/GCOM-W1-AMSR2-L3-SND/lis-1km-missouri.nc4"
)
DEFAULT_LIS_PATH = (
    f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/lis_input_NMP_1000m_missouri.nc"
)
DEFAULT_OUTPUT_PREFIX = f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/outputs"

TIME_UNITS = "days since 1970-01-01"
TIME_CALENDAR = "proleptic_gregorian"
TIME_EPOCH = pd.Timestamp("1970-01-01")

PREALLOC_START = pd.Timestamp("2012-06-01")
PREALLOC_END = pd.Timestamp("2030-12-31")
_prealloc_days = pd.date_range(PREALLOC_START, PREALLOC_END, freq="D")
_DAY_TO_SLOT: dict[int, int] = {
    int((d - TIME_EPOCH).days): i for i, d in enumerate(_prealloc_days)
}

NLAT, NLON = 1800, 3600
FILL_MISSING = np.int16(-32768)
FILL_NOOBS = np.int16(-32767)
SCALE_FACTOR = np.float32(0.1)

ORBIT_LABELS = ["Ascending", "Descending"]
BAND_LABELS = ["snow_depth", "quality_flag"]

# ── helpers ───────────────────────────────────────────────────────────────────


def _date_to_days(date: str) -> int:
    ts = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:]}")
    return int((ts - TIME_EPOCH).days)


def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")


def _parse_s3(uri: str) -> tuple[str, str]:
    without = uri[len("s3://"):]
    bucket, _, prefix = without.partition("/")
    return bucket, prefix


# ── 1. open IceChunk store ────────────────────────────────────────────────────


def open_store(store_uri: str) -> xr.Dataset:
    """Open the IceChunk store as a read-only xarray Dataset.

    Returns the dataset with ``mask_and_scale=False`` so raw int16 values are
    preserved for NaN-masked sparse multiply.
    """
    log.info("Opening IceChunk store: %s", store_uri)
    bucket, prefix = _parse_s3(store_uri)
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, region="us-west-2")
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "https://gportal.jaxa.jp/", icechunk.http_store()
        )
    )
    repo = icechunk.Repository.open(
        storage,
        config=config,
        authorize_virtual_chunk_access={"https://gportal.jaxa.jp/": None},
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(
        session.store,
        consolidated=False,
        zarr_format=3,
        mask_and_scale=False,
        chunks={},  # use stored chunk layout; Dask-backed
    )
    log.info("Store opened. Variables: %s", list(ds.data_vars))
    return ds


# ── 2. load ESMF weight matrix ────────────────────────────────────────────────


def load_weights(weights_uri: str, tmp_dir: Path) -> scipy.sparse.csr_matrix:
    """Download the ESMF weight file and return a CSR sparse matrix.

    Parameters
    ----------
    weights_uri:
        ``s3://`` URI or local path to the NetCDF4 ESMF weight file.
    tmp_dir:
        Scratch directory used when downloading from S3.
    """
    if _is_s3(weights_uri):
        import boto3

        local = tmp_dir / "weights.nc4"
        log.info("Downloading weights from %s …", weights_uri)
        bucket, key = _parse_s3(weights_uri)
        boto3.client("s3").download_file(bucket, key, str(local))
        weights_path = local
    else:
        weights_path = Path(weights_uri)

    log.info("Loading weight matrix from %s …", weights_path)
    with nc.Dataset(weights_path, "r") as ds:
        S = ds.variables["S"][:]
        row = ds.variables["row"][:] - 1   # 1-based → 0-based
        col = ds.variables["col"][:] - 1
        n_dst = int(ds.variables["dst_grid_dims"][:].prod())
        n_src = int(ds.variables["src_grid_dims"][:].prod())

    W = scipy.sparse.csr_matrix((S, (row, col)), shape=(n_dst, n_src))
    log.info(
        "Weight matrix: %d src → %d dst cells, %d non-zeros (%.1f MB)",
        n_src, n_dst, W.nnz,
        (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 1e6,
    )
    return W


# ── 3. load LIS grid coordinates ─────────────────────────────────────────────


def load_lis_grid(lis_path: str, tmp_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lat2d, lon2d)`` arrays for the LIS destination grid.

    Downloads from S3 if necessary.  Arrays are float32, shape ``(ny, nx)``.
    """
    if _is_s3(lis_path):
        import boto3

        local = tmp_dir / "lis_grid.nc"
        log.info("Downloading LIS grid from %s …", lis_path)
        bucket, key = _parse_s3(lis_path)
        boto3.client("s3").download_file(bucket, key, str(local))
        path = local
    else:
        path = Path(lis_path)

    ds = xr.open_dataset(path)
    lat2d = ds["lat"].values.astype("float32")
    lon2d = ds["lon"].values.astype("float32")
    log.info("LIS grid shape: %s", lat2d.shape)
    return lat2d, lon2d


# ── 4. apply weights with Dask ────────────────────────────────────────────────


def _regrid_block(
    block: np.ndarray,
    W: scipy.sparse.csr_matrix,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    """Apply sparse weight matrix to one (1,1,nlat,nlon,1) block.

    Fill values are masked to NaN before the multiply so they cannot infect
    neighbouring destination cells.  Destination cells with zero valid-weight
    contribution remain NaN.

    Returns float32 array of shape ``(1, 1, ny_lis, nx_lis, 1)``.
    """
    # block shape: (1, 1, NLAT, NLON, 1)
    raw = block[0, 0, :, :, 0]  # (NLAT, NLON) int16

    fdata = raw.astype("float32") * SCALE_FACTOR
    for fv in (int(FILL_MISSING), int(FILL_NOOBS)):
        fdata[raw == fv] = np.nan

    src = fdata.ravel()
    valid = np.isfinite(src)
    src_clean = np.where(valid, src, 0.0)
    valid_w = np.where(valid, 1.0, 0.0)

    dst_vals = W @ src_clean
    dst_wsum = W @ valid_w

    with np.errstate(invalid="ignore", divide="ignore"):
        dst = np.where(dst_wsum > 0, dst_vals / dst_wsum, np.nan)

    ny, nx = dst_shape
    return dst.reshape(1, 1, ny, nx, 1).astype("float32")


def apply_weights_dask(
    ic_ds: xr.Dataset,
    slots: list[int],
    W: scipy.sparse.csr_matrix,
    dst_shape: tuple[int, int],
    scheduler: str = "synchronous",
    n_workers: int = 4,
) -> da.Array:
    """Apply W to the IceChunk Dask array and return a Dask array of results.

    The sparse multiply is embarrassingly parallel over ``(time, orbit, band)``;
    each block is ``(1, 1, NLAT, NLON, 1)``.

    Parameters
    ----------
    ic_ds:
        IceChunk dataset (raw int16, Dask-backed).
    slots:
        Time-axis indices to select from the store.
    W:
        CSR weight matrix, shape ``(n_dst, n_src)``.
    dst_shape:
        ``(ny_lis, nx_lis)``.
    scheduler:
        Dask scheduler: ``"synchronous"`` (default, no extra deps),
        ``"threads"``, or ``"distributed"`` (requires a running cluster).
    n_workers:
        Used only when *scheduler* is ``"threads"``.

    Returns
    -------
    Dask array of shape ``(ntime, 2, ny_lis, nx_lis, 2)``, dtype float32.
    """
    ny, nx = dst_shape

    # Select the requested time slots and rechunk to one block per (t, orbit, band).
    raw = (
        ic_ds["geophysical_data"]
        .isel(time=slots)
        .chunk({"time": 1, "orbit": 1, "lat": NLAT, "lon": NLON, "band": 1})
    )
    src_arr = raw.data  # dask array, shape (ntime, 2, NLAT, NLON, 2)

    def _block_fn(block, block_info=None):
        return _regrid_block(block, W, dst_shape)

    out = da.map_blocks(
        _block_fn,
        src_arr,
        dtype="float32",
        chunks=(1, 1, ny, nx, 1),
    )
    return out  # (ntime, 2, ny, nx, 2)


# ── 5. build output Dataset and write ────────────────────────────────────────


def build_output_dataset(
    out_arr: np.ndarray,
    dates: list[str],
    lat2d: np.ndarray,
    lon2d: np.ndarray,
) -> xr.Dataset:
    """Wrap the regridded numpy array in an CF-compliant xarray Dataset."""
    ny, nx = lat2d.shape
    time_vals = pd.DatetimeIndex(
        [pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]}") for d in dates]
    )

    variables: dict[str, xr.DataArray] = {}
    for o_idx, orbit in enumerate(ORBIT_LABELS):
        for b_idx, band in enumerate(BAND_LABELS):
            var_name = f"{band}_{orbit.lower()}"
            data = out_arr[:, o_idx, :, :, b_idx]  # (ntime, ny, nx)
            variables[var_name] = xr.DataArray(
                data,
                dims=["time", "south_north", "west_east"],
                coords={
                    "time": time_vals,
                    "south_north": np.arange(ny),
                    "west_east": np.arange(nx),
                    "lat": (["south_north", "west_east"], lat2d),
                    "lon": (["south_north", "west_east"], lon2d),
                },
                attrs={
                    "long_name": (
                        f"AMSR2 {band.replace('_', ' ')} ({orbit} orbit)"
                    ),
                    "units": "cm" if band == "snow_depth" else "1",
                    "grid_mapping": "crs",
                    "_FillValue": np.float32(np.nan),
                    "source": (
                        "JAXA GCOM-W1 AMSR2 L3 Snow Depth "
                        "(bilinear regrid to LIS 1 km LCC)"
                    ),
                },
            )

    ds = xr.Dataset(
        variables,
        attrs={
            "Conventions": "CF-1.8",
            "title": (
                "AMSR2 L3 Snow Depth regridded to LIS 1 km Lambert Conformal grid"
            ),
            "source_product": "JAXA GCOM-W1/AMSR2 L3 Snow Depth (L3SGSNDHG2210210)",
            "regrid_method": "ESMF bilinear (weights pre-computed, applied via scipy.sparse)",
        },
    )
    ds["crs"] = xr.DataArray(
        np.int32(0),
        attrs={"grid_mapping_name": "lambert_conformal_conic"},
    )
    return ds


def _write_output(ds: xr.Dataset, output_path: str, tmp_dir: Path) -> None:
    if _is_s3(output_path):
        import boto3

        local = tmp_dir / "output.nc"
        ds.to_netcdf(local, engine="h5netcdf")
        bucket, key = _parse_s3(output_path)
        log.info("Uploading output to %s …", output_path)
        boto3.client("s3").upload_file(str(local), bucket, key)
        log.info("Upload complete.")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_path, engine="h5netcdf")
        log.info("Output written to %s.", output_path)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply pre-computed ESMF weights to AMSR2 IceChunk data."
    )
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument(
        "--end-date", default=None,
        help="YYYYMMDD inclusive (default: same as --start-date)",
    )
    parser.add_argument("--store-uri", default=DEFAULT_STORE_URI)
    parser.add_argument("--weights-uri", default=DEFAULT_WEIGHTS_URI)
    parser.add_argument("--lis-path", default=DEFAULT_LIS_PATH)
    parser.add_argument("--output-path", default=None)
    parser.add_argument(
        "--scheduler", default="threads",
        choices=["synchronous", "threads", "distributed"],
        help="Dask scheduler (default: threads)",
    )
    parser.add_argument(
        "--n-workers", type=int, default=4,
        help="Thread-pool workers when --scheduler=threads (default: 4)",
    )
    ns = parser.parse_args()

    start = ns.start_date
    end = ns.end_date or start
    dates = [
        d.strftime("%Y%m%d")
        for d in pd.date_range(
            pd.Timestamp(f"{start[:4]}-{start[4:6]}-{start[6:]}"),
            pd.Timestamp(f"{end[:4]}-{end[4:6]}-{end[6:]}"),
            freq="D",
        )
    ]
    log.info("Dates to process: %s … %s (%d total)", dates[0], dates[-1], len(dates))

    output_path = ns.output_path
    if output_path is None:
        output_path = (
            f"{DEFAULT_OUTPUT_PREFIX}/GCOM-W1-AMSR2-L3-SND-{start}-{end}.nc"
        )
    log.info("Output path: %s", output_path)

    slots = [_DAY_TO_SLOT[_date_to_days(d)] for d in dates]

    with tempfile.TemporaryDirectory(prefix="amsr2_consumer_") as tmp_str:
        tmp_dir = Path(tmp_str)

        # -- open resources -------------------------------------------------
        ic_ds = open_store(ns.store_uri)
        W = load_weights(ns.weights_uri, tmp_dir)
        lat2d, lon2d = load_lis_grid(ns.lis_path, tmp_dir)
        dst_shape = lat2d.shape  # (ny_lis, nx_lis)

        # -- build and compute Dask graph -----------------------------------
        log.info(
            "Building Dask graph (scheduler=%s, workers=%d) …",
            ns.scheduler, ns.n_workers,
        )
        out_dask = apply_weights_dask(
            ic_ds, slots, W, dst_shape,
            scheduler=ns.scheduler,
            n_workers=ns.n_workers,
        )

        if ns.scheduler == "distributed":
            from distributed import Client, LocalCluster
            with LocalCluster(n_workers=ns.n_workers, threads_per_worker=1) as cluster:
                with Client(cluster) as client:
                    W_future = client.scatter(W, broadcast=True)
                    # Recompute with scattered W so workers don't re-serialise it.
                    out_dask = apply_weights_dask(
                        ic_ds, slots, W_future.result(), dst_shape,
                        scheduler=ns.scheduler,
                    )
                    log.info("Computing with distributed scheduler …")
                    out_np = out_dask.compute(scheduler=client)
        elif ns.scheduler == "threads":
            log.info("Computing with thread-pool scheduler …")
            with dask.config.set(scheduler="threads", num_workers=ns.n_workers):
                out_np = out_dask.compute()
        else:
            log.info("Computing synchronously …")
            out_np = out_dask.compute(scheduler="synchronous")

        log.info("Computation complete. Output shape: %s", out_np.shape)

        # -- write output ---------------------------------------------------
        ds_out = build_output_dataset(out_np, dates, lat2d, lon2d)
        _write_output(ds_out, output_path, tmp_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
