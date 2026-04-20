#!/usr/bin/env python
"""Benchmark CEDA SWE regridding: compare nearest vs bilinear methods."""

import sys
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from join_scratch.datasets import CedaFileHandler
from join_scratch.regrid.regular_to_regular import compute_weights, load_regridder
from join_scratch.utils import _time_call, BenchmarkResult, render_report
from lis_grid import load_lis_grid
from s3_utils import make_store, make_fs, handler_from_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

CEDA_GLOB = "**/*.nc"
S3_BUCKET = "airborne-smce-prod-user-bucket"
S3_JOIN_PREFIX = "JOIN"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CEDA SWE regridding methods."
    )
    parser.add_argument(
        "--lis-path",
        default=f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/lis_input_NMP_1000m_missouri.nc",
        help="Local path or s3:// URI to the LIS NetCDF file.",
    )
    parser.add_argument(
        "--input-prefix",
        default=f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/CEDA",
        help="S3 URI prefix (s3://bucket/prefix) or local directory containing CEDA NetCDF files.",
    )
    parser.add_argument("--weights-dir", type=Path, default=Path("_data/ceda"))
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
        without_scheme = input_prefix[len("s3://"):]
        bucket, _, key_prefix = without_scheme.partition("/")
        store = make_store(bucket, prefix=key_prefix)
        fs = make_fs()
        keys = [k for k in (obj["path"] for page in store.list(None) for obj in page) if k.endswith(".nc")]
        keys.sort()
        if not keys:
            raise FileNotFoundError(f"No CEDA NetCDF files found at {input_prefix}")
        first_key = keys[0]
        first_url = f"s3://{bucket}/{key_prefix}/{first_key}" if key_prefix else f"s3://{bucket}/{first_key}"
        log.info("Benchmarking on %s", first_url)
        handler = handler_from_s3(CedaFileHandler, first_url, fs=fs)
    else:
        local_dir = Path(input_prefix)
        ceda_files = sorted(local_dir.glob(CEDA_GLOB))
        if not ceda_files:
            raise FileNotFoundError(f"No CEDA NetCDF files found under {local_dir}")
        log.info("Benchmarking on %s", ceda_files[0].name)
        handler = CedaFileHandler.from_path(ceda_files[0])

    lis_grid = load_lis_grid(ns.lis_path)
    ds = handler.get_dataset()
    source_area = handler.get_area_def()

    lats = ds["lat"].values if "lat" in ds else source_area.get_lonlats()[1][:, 0]
    lons = ds["lon"].values if "lon" in ds else source_area.get_lonlats()[0][0, :]
    source_grid = xr.Dataset(
        coords={"lat": np.sort(np.unique(lats)), "lon": np.sort(np.unique(lons))}
    )

    # Assign 1-D lat/lon coords to y/x dims if not already present, then swap
    if "lat" not in ds.coords:
        ds = ds.assign_coords(
            lat=("y", np.sort(np.unique(lats))),
            lon=("x", np.sort(np.unique(lons))),
        )
    # Prepare dataset with lat/lon as dim names for xESMF
    ds_regrid = ds.swap_dims({"y": "lat", "x": "lon"})
    source_shape = (len(source_grid["lat"]), len(source_grid["lon"]))

    results: list[BenchmarkResult] = []

    for method in ("nearest_s2d", "bilinear"):
        weights_path = ns.weights_dir / f"ceda-lis-weights-{method}.nc"
        ns.weights_dir.mkdir(parents=True, exist_ok=True)

        if ns.no_cache:
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
                    label=f"CEDA {method} [weights gen]",
                    source_shape=source_shape,
                    elapsed_s=weights_elapsed,
                    rss_delta_mib=weights_rss,
                    notes="weights generation only",
                )
            )
        else:
            compute_weights(source_grid, lis_grid, weights_path, method=method)

        regridder = load_regridder(source_grid, lis_grid, weights_path, method=method)

        elapsed, rss_delta = _time_call(regridder, ds_regrid)
        results.append(
            BenchmarkResult(
                label=f"CEDA {method}",
                source_shape=source_shape,
                elapsed_s=elapsed,
                rss_delta_mib=rss_delta,
            )
        )
        log.info("%s: %.2f s, %.1f MiB", method, elapsed, rss_delta)

    print(render_report(results, title="CEDA Regrid Benchmark"))

    if ns.output_dir is not None:
        ts = datetime.now(tz=timezone.utc)
        ns.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = ns.output_dir / f"ceda_benchmark_{ts.strftime('%Y%m%dT%H%M%SZ')}.txt"
        report_path.write_text(render_report(results, title="CEDA Regrid Benchmark", timestamp=ts))
        log.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()

