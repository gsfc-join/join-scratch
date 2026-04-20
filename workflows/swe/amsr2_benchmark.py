#!/usr/bin/env python
"""Benchmark AMSR2 regridding: compare nearest vs bilinear methods."""

import sys
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from join_scratch.datasets import Amsr2FileHandler
from join_scratch.regrid.regular_to_regular import compute_weights, load_regridder
from join_scratch.utils import _time_call, BenchmarkResult, render_report
from lis_grid import load_lis_grid
from s3_utils import make_store, make_fs, handler_from_s3

import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

AMSR2_GLOB = "**/*.h5"
# S3 defaults
S3_BUCKET = "airborne-smce-prod-user-bucket"
S3_JOIN_PREFIX = "JOIN"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AMSR2 regridding methods.")
    parser.add_argument(
        "--lis-path",
        default=f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/lis_input_NMP_1000m_missouri.nc",
        help="Local path or s3:// URI to the LIS NetCDF file.",
    )
    parser.add_argument(
        "--input-prefix",
        default=f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/AMSR2",
        help="S3 URI prefix (s3://bucket/prefix) or local directory containing AMSR2 HDF5 files.",
    )
    parser.add_argument("--weights-dir", type=Path, default=Path("_data/amsr2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write timestamped report file (default: print to stdout only)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help=(
            "Force regeneration of weights files even if they already exist, "
            "and include weights-generation time in the report."
        ),
    )
    ns = parser.parse_args()

    # Resolve file(s) to benchmark
    input_prefix = str(ns.input_prefix)
    if input_prefix.startswith("s3://"):
        # Parse bucket and prefix from the URI
        without_scheme = input_prefix[len("s3://"):]
        bucket, _, key_prefix = without_scheme.partition("/")
        store = make_store(bucket, prefix=key_prefix)
        fs = make_fs()
        keys = [k for k in (obj["path"] for page in store.list(None) for obj in page) if k.endswith(".h5")]
        keys.sort()
        if not keys:
            raise FileNotFoundError(f"No AMSR2 HDF5 files found at {input_prefix}")
        first_key = keys[0]
        first_url = f"s3://{bucket}/{key_prefix}/{first_key}" if key_prefix else f"s3://{bucket}/{first_key}"
        log.info("Benchmarking on %s", first_url)
        handler = handler_from_s3(Amsr2FileHandler, first_url, fs=fs)
    else:
        local_dir = Path(input_prefix)
        amsr2_files = sorted(local_dir.glob(AMSR2_GLOB))
        if not amsr2_files:
            raise FileNotFoundError(f"No AMSR2 HDF5 files found under {local_dir}")
        log.info("Benchmarking on %s", amsr2_files[0].name)
        handler = Amsr2FileHandler.from_path(amsr2_files[0])
        fs = None

    lis_grid = load_lis_grid(ns.lis_path)

    lat_asc = np.sort(handler._lat)
    lon_asc = np.sort(handler._lon)
    source_grid = xr.Dataset(coords={"lat": lat_asc, "lon": lon_asc})

    da = handler.get_dataset().rename({"y": "lat", "x": "lon"}).sortby("lat")
    ds = da.to_dataset(name="Geophysical Data")
    source_shape = (len(lat_asc), len(lon_asc))

    results: list[BenchmarkResult] = []

    for method in ("nearest_s2d", "bilinear"):
        weights_path = ns.weights_dir / f"amsr2-lis-weights-{method}.nc"
        ns.weights_dir.mkdir(parents=True, exist_ok=True)

        if ns.no_cache:
            # Delete existing weights file so compute_weights always regenerates
            if weights_path.exists():
                weights_path.unlink()
                log.info("--no-cache: removed existing weights file %s", weights_path)
            weights_elapsed, weights_rss = _time_call(
                compute_weights, source_grid, lis_grid, weights_path, method=method
            )
            log.info(
                "%s weights generation: %.2f s, %.1f MiB",
                method,
                weights_elapsed,
                weights_rss,
            )
            results.append(
                BenchmarkResult(
                    label=f"AMSR2 {method} [weights gen]",
                    source_shape=source_shape,
                    elapsed_s=weights_elapsed,
                    rss_delta_mib=weights_rss,
                    notes="weights generation only",
                )
            )
        else:
            compute_weights(source_grid, lis_grid, weights_path, method=method)

        regridder = load_regridder(source_grid, lis_grid, weights_path, method=method)

        elapsed, rss_delta = _time_call(regridder, ds)
        results.append(
            BenchmarkResult(
                label=f"AMSR2 {method}",
                source_shape=source_shape,
                elapsed_s=elapsed,
                rss_delta_mib=rss_delta,
            )
        )
        log.info("%s: %.2f s, %.1f MiB", method, elapsed, rss_delta)

    print(render_report(results, title="AMSR2 Regrid Benchmark"))

    if ns.output_dir is not None:
        ts = datetime.now(tz=timezone.utc)
        ns.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = ns.output_dir / f"amsr2_benchmark_{ts.strftime('%Y%m%dT%H%M%SZ')}.txt"
        report_path.write_text(render_report(results, title="AMSR2 Regrid Benchmark", timestamp=ts))
        log.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
