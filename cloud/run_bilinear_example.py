#!/usr/bin/env python
"""Run the bilinear regridding example via AWS Batch.

Mirrors the logic in misc-docs/esmf-regridding-examples/01_bilinear_regridding.qmd
but delegates weight generation to the ESMF regrid Batch job.

Steps
-----
1. Submit the regrid Batch job with SRC_GRID_SPEC + DST_GRID_SPEC
   (or skip if --skip-batch is set and weights already exist on S3).
2. Poll until the job SUCCEEDS or FAILS.
3. Download the weight file from S3.
4. Build synthetic temperature source data (same as the .qmd example).
5. Apply W @ src_flat (sparse matmul).
6. Save the regridded field as a NetCDF file.

Usage
-----
    python run_bilinear_example.py
    python run_bilinear_example.py --src-res 10 --dst-res 1
    python run_bilinear_example.py --skip-batch --output out.nc
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import boto3
import netCDF4 as nc  # noqa: N813
import numpy as np
import scipy.sparse
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────────

S3_BUCKET = "airborne-smce-prod-user-bucket"
TF_DIR = Path(__file__).parent / "terraform"

DEFAULT_WEIGHTS_URI = (
    f"s3://{S3_BUCKET}/JOIN/cached-weights/bilinear-example/weights.nc4"
)

# ── terraform output helpers ──────────────────────────────────────────────────


def _tf_output(key: str) -> str:
    """Read a single value from terraform output."""
    result = subprocess.run(
        ["terraform", f"-chdir={TF_DIR}", "output", "-raw", key],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"terraform output -raw {key} failed:\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


# ── S3 helpers ────────────────────────────────────────────────────────────────


def _parse_s3(uri: str) -> tuple[str, str]:
    without = uri[len("s3://"):]
    bucket, _, key = without.partition("/")
    return bucket, key


def _s3_exists(uri: str) -> bool:
    bucket, key = _parse_s3(uri)
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False
    except Exception:
        return False


def _s3_download(uri: str, local: Path) -> None:
    bucket, key = _parse_s3(uri)
    log.info("Downloading s3://%s/%s → %s", bucket, key, local)
    boto3.client("s3").download_file(bucket, key, str(local))


# ── grid spec helpers ─────────────────────────────────────────────────────────


def _grid_spec(res: float) -> dict:
    """Build a regular global lat/lon grid spec for a given resolution in degrees."""
    half = res / 2.0
    nlat = int(180.0 / res)
    nlon = int(360.0 / res)
    return {
        "nlat": nlat,
        "nlon": nlon,
        "lat_min": -90.0 + half,
        "lat_max":  90.0 - half,
        "lon_min": -180.0 + half,
        "lon_max":  180.0 - half,
    }


def _grid_centers(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (lats, lons) 1-D center arrays from a grid spec."""
    lats = np.linspace(spec["lat_min"], spec["lat_max"], spec["nlat"])
    lons = np.linspace(spec["lon_min"], spec["lon_max"], spec["nlon"])
    return lats, lons


# ── Batch job submission & polling ────────────────────────────────────────────


def _submit_batch_job(
    job_queue: str,
    job_definition: str,
    src_spec: dict,
    dst_spec: dict,
    weights_uri: str,
    method: str,
) -> str:
    """Submit the regrid Batch job and return the job ID."""
    client = boto3.client("batch")

    response = client.submit_job(
        jobName="bilinear-example",
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides={
            "environment": [
                {"name": "SRC_GRID_SPEC",    "value": json.dumps(src_spec)},
                {"name": "DST_GRID_SPEC",    "value": json.dumps(dst_spec)},
                {"name": "WEIGHTS_URI",      "value": weights_uri},
                {"name": "METHOD",           "value": method},
                {"name": "FORCE_REGENERATE", "value": "true"},
                {"name": "EXTRA_ARGS",       "value": "--no_log --ignore_unmapped"},
                # LIS_PATH not needed — using DST_GRID_SPEC
                {"name": "LIS_PATH",         "value": ""},
            ],
        },
    )
    job_id = response["jobId"]
    log.info("Submitted Batch job: %s", job_id)
    return job_id


def _poll_batch_job(job_id: str, poll_interval: int = 30) -> str:
    """Poll until the job reaches a terminal state. Returns final status."""
    client = boto3.client("batch")
    while True:
        resp = client.describe_jobs(jobs=[job_id])
        job = resp["jobs"][0]
        status = job["status"]
        reason = job.get("statusReason", "")
        log.info("Job %s status: %s%s", job_id, status, f" — {reason}" if reason else "")
        if status in ("SUCCEEDED", "FAILED"):
            return status
        time.sleep(poll_interval)


# ── synthetic temperature field (mirrors the .qmd example) ───────────────────


def _synthetic_temperature(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Latitudinal gradient + wave anomalies, shape (nlat, nlon)."""
    lon2d, lat2d = np.meshgrid(lons, lats)
    T = 30.0 * np.cos(np.deg2rad(lat2d))
    T += 5.0 * np.sin(np.deg2rad(2 * lon2d)) * np.cos(np.deg2rad(lat2d))
    T += 2.0 * np.cos(np.deg2rad(3 * lon2d + lat2d))
    return T.astype("float32")


# ── load weights ──────────────────────────────────────────────────────────────


def _load_weights(path: Path, n_src: int, n_dst: int) -> scipy.sparse.csr_matrix:
    with nc.Dataset(path, "r") as ds:
        S   = ds.variables["S"][:]
        row = ds.variables["row"][:] - 1  # 0-based
        col = ds.variables["col"][:] - 1
    W = scipy.sparse.csr_matrix((S, (row, col)), shape=(n_dst, n_src))
    log.info(
        "Weight matrix: (%d, %d), %d non-zeros",
        n_dst, n_src, W.nnz,
    )
    return W


# ── apply weights ─────────────────────────────────────────────────────────────


def _apply_weights(
    W: scipy.sparse.csr_matrix,
    src_data: np.ndarray,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    """Apply W @ src_flat and reshape to dst_shape."""
    dst_flat = W @ src_data.ravel()
    return dst_flat.reshape(dst_shape).astype("float32")


# ── write output NetCDF ───────────────────────────────────────────────────────


def _write_output(
    dst_data: np.ndarray,
    dst_lats: np.ndarray,
    dst_lons: np.ndarray,
    src_res: float,
    dst_res: float,
    output_path: Path,
) -> None:
    ds = xr.Dataset(
        {
            "temperature": xr.DataArray(
                dst_data,
                dims=["lat", "lon"],
                coords={"lat": dst_lats, "lon": dst_lons},
                attrs={
                    "long_name": "Synthetic temperature (bilinear regridded)",
                    "units": "degC",
                    "regrid_method": "ESMF bilinear",
                    "source_resolution_deg": src_res,
                    "dest_resolution_deg": dst_res,
                },
            )
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"Bilinear regrid example: {src_res}° → {dst_res}°",
        },
    )
    ds.to_netcdf(output_path)
    log.info("Output written to %s  shape=%s", output_path, dst_data.shape)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the bilinear regridding example via AWS Batch."
    )
    parser.add_argument(
        "--src-res", type=float, default=10.0,
        help="Source grid resolution in degrees (default: 10)",
    )
    parser.add_argument(
        "--dst-res", type=float, default=2.0,
        help="Destination grid resolution in degrees (default: 2)",
    )
    parser.add_argument(
        "--method", default="bilinear",
        help="ESMF regridding method (default: bilinear)",
    )
    parser.add_argument(
        "--weights-uri", default=DEFAULT_WEIGHTS_URI,
        help=f"S3 URI for the weight file (default: {DEFAULT_WEIGHTS_URI})",
    )
    parser.add_argument(
        "--job-queue", default=None,
        help="Batch job queue name (default: read from terraform output)",
    )
    parser.add_argument(
        "--job-definition", default=None,
        help="Batch job definition name (default: read from terraform output)",
    )
    parser.add_argument(
        "--skip-batch", action="store_true",
        help="Skip Batch submission; download existing weights from S3 directly",
    )
    parser.add_argument(
        "--output", default="bilinear_example_output.nc",
        help="Output NetCDF path (default: bilinear_example_output.nc)",
    )
    ns = parser.parse_args()

    src_spec = _grid_spec(ns.src_res)
    dst_spec = _grid_spec(ns.dst_res)
    src_lats, src_lons = _grid_centers(src_spec)
    dst_lats, dst_lons = _grid_centers(dst_spec)

    log.info(
        "Source grid : %d×%d (%.1f°)",
        src_spec["nlat"], src_spec["nlon"], ns.src_res,
    )
    log.info(
        "Dest grid   : %d×%d (%.1f°)",
        dst_spec["nlat"], dst_spec["nlon"], ns.dst_res,
    )

    # ── 1. Weight generation via Batch ────────────────────────────────────────
    if ns.skip_batch:
        log.info("--skip-batch set — skipping Batch submission.")
    else:
        job_queue = ns.job_queue or _tf_output("batch_job_queue_name")
        job_definition = ns.job_definition or _tf_output("regrid_job_definition_name")
        log.info("Job queue      : %s", job_queue)
        log.info("Job definition : %s", job_definition)

        job_id = _submit_batch_job(
            job_queue=job_queue,
            job_definition=job_definition,
            src_spec=src_spec,
            dst_spec=dst_spec,
            weights_uri=ns.weights_uri,
            method=ns.method,
        )
        status = _poll_batch_job(job_id)
        if status != "SUCCEEDED":
            log.error("Batch job %s failed — aborting.", job_id)
            sys.exit(1)
        log.info("Batch job succeeded.")

    # ── 2. Download weights ───────────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="bilinear_example_") as tmp_str:
        weights_local = Path(tmp_str) / "weights.nc4"
        _s3_download(ns.weights_uri, weights_local)

        n_src = src_spec["nlat"] * src_spec["nlon"]
        n_dst = dst_spec["nlat"] * dst_spec["nlon"]
        W = _load_weights(weights_local, n_src, n_dst)

    # ── 3. Build synthetic source data ────────────────────────────────────────
    src_T = _synthetic_temperature(src_lons, src_lats)
    log.info("Source data shape: %s  min=%.2f  max=%.2f", src_T.shape, src_T.min(), src_T.max())

    # ── 4. Apply weights ──────────────────────────────────────────────────────
    dst_T = _apply_weights(W, src_T, (dst_spec["nlat"], dst_spec["nlon"]))
    log.info("Regridded data shape: %s  min=%.2f  max=%.2f", dst_T.shape, dst_T.min(), dst_T.max())

    # ── 5. Write output ───────────────────────────────────────────────────────
    _write_output(
        dst_data=dst_T,
        dst_lats=dst_lats,
        dst_lons=dst_lons,
        src_res=ns.src_res,
        dst_res=ns.dst_res,
        output_path=Path(ns.output),
    )


if __name__ == "__main__":
    main()
