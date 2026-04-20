#!/usr/bin/env python
"""Benchmark VIIRS CGF Snow Cover regridding: compare pyresample methods."""

import sys
import argparse
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from join_scratch.datasets import ViirsFileHandler
from join_scratch.regrid import regrid
from join_scratch.utils import _time_call, BenchmarkResult, render_report
from lis_grid import build_lis_area_definition
from s3_utils import make_store, make_fs, handler_from_s3
from pyresample.geometry import SwathDefinition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

VIIRS_GLOB = "**/*.h5"
METHODS = ["nearest", "bilinear", "ewa", "bucket_avg"]
S3_BUCKET = "airborne-smce-prod-user-bucket"
S3_JOIN_PREFIX = "JOIN"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark VIIRS CGF Snow Cover regridding methods."
    )
    parser.add_argument(
        "--lis-path",
        default=f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/lis_input_NMP_1000m_missouri.nc",
        help="Local path or s3:// URI to the LIS NetCDF file.",
    )
    parser.add_argument(
        "--input-prefix",
        default=f"s3://{S3_BUCKET}/{S3_JOIN_PREFIX}/VIIRS",
        help="S3 URI prefix (s3://bucket/prefix) or local directory containing VIIRS HDF5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write timestamped report file (default: print to stdout only)",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=20,
        help="Maximum number of VIIRS tiles to load per date (default: 20). "
             "Use 0 for no limit (loads all tiles, may require large memory).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help=(
            "VIIRS uses no weights files; this flag clears the pyresample resampler "
            "cache between runs so each method is benchmarked cold. "
            "Has no effect on the output report format."
        ),
    )
    ns = parser.parse_args()

    lis_area = build_lis_area_definition(ns.lis_path)

    # Resolve file(s) to benchmark — group by date, take first date
    input_prefix = str(ns.input_prefix)
    if input_prefix.startswith("s3://"):
        without_scheme = input_prefix[len("s3://"):]
        bucket, _, key_prefix = without_scheme.partition("/")
        store = make_store(bucket, prefix=key_prefix)
        fs = make_fs()
        all_keys = sorted(
            obj["path"] for page in store.list(None) for obj in page
            if obj["path"].endswith(".h5")
        )
        if not all_keys:
            raise FileNotFoundError(f"No VIIRS HDF5 files found at {input_prefix}")
        # Group by date key (second dot-separated component of filename)
        date_groups: dict[str, list[str]] = defaultdict(list)
        for key in all_keys:
            fname = key.split("/")[-1]
            parts = fname.split(".")
            date_key = parts[1] if len(parts) > 1 else fname
            date_groups[date_key].append(key)
        date_key, group_keys = next(iter(sorted(date_groups.items())))
        if ns.max_tiles and len(group_keys) > ns.max_tiles:
            group_keys = group_keys[: ns.max_tiles]
        log.info("Benchmarking on date %s (%d tile(s))", date_key, len(group_keys))
        handlers = [
            handler_from_s3(
                ViirsFileHandler,
                f"s3://{bucket}/{key_prefix}/{k}" if key_prefix else f"s3://{bucket}/{k}",
                fs=fs,
            )
            for k in group_keys
        ]
    else:
        local_dir = Path(input_prefix)
        viirs_files = sorted(local_dir.glob(VIIRS_GLOB))
        if not viirs_files:
            raise FileNotFoundError(f"No VIIRS HDF5 files found under {local_dir}")
        date_groups_local: dict[str, list[Path]] = defaultdict(list)
        for p in viirs_files:
            parts = p.stem.split(".")
            date_key = parts[1] if len(parts) > 1 else p.stem
            date_groups_local[date_key].append(p)
        date_key, local_paths = next(iter(sorted(date_groups_local.items())))
        if ns.max_tiles and len(local_paths) > ns.max_tiles:
            local_paths = local_paths[: ns.max_tiles]
        log.info("Benchmarking on date %s (%d tile(s))", date_key, len(local_paths))
        handlers = [ViirsFileHandler.from_path(p) for p in local_paths]

    # Load and composite tiles
    all_data, all_lons, all_lats = [], [], []
    for handler in handlers:
        da = handler.get_dataset()
        swath_def = da.attrs["area"]
        all_data.append(da.values)
        all_lons.append(swath_def.lons.values)
        all_lats.append(swath_def.lats.values)

    composite_data = np.concatenate(all_data, axis=0)
    composite_lons = np.concatenate(all_lons, axis=0)
    composite_lats = np.concatenate(all_lats, axis=0)

    lons_da = xr.DataArray(composite_lons, dims=["y", "x"])
    lats_da = xr.DataArray(composite_lats, dims=["y", "x"])
    source_def = SwathDefinition(lons=lons_da, lats=lats_da)
    composite_da = xr.DataArray(composite_data, dims=["y", "x"])
    source_shape = composite_da.shape

    results: list[BenchmarkResult] = []
    for method in METHODS:
        if ns.no_cache:
            # Clear pyresample's resampler cache so the method is benchmarked cold
            try:
                from pyresample.kd_tree import KDTreeResampler
                KDTreeResampler.cache = {}
            except (ImportError, AttributeError):
                pass
            import gc
            gc.collect()
        elapsed, rss_delta = _time_call(
            regrid, composite_da, source_def, lis_area, method=method
        )
        results.append(
            BenchmarkResult(
                label=f"VIIRS {method}",
                source_shape=source_shape,
                elapsed_s=elapsed,
                rss_delta_mib=rss_delta,
            )
        )
        log.info("%s: %.2f s, %.1f MiB", method, elapsed, rss_delta)

    print(render_report(results, title="VIIRS Regrid Benchmark"))

    if ns.output_dir is not None:
        ts = datetime.now(tz=timezone.utc)
        ns.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = ns.output_dir / f"viirs_benchmark_{ts.strftime('%Y%m%dT%H%M%SZ')}.txt"
        report_path.write_text(render_report(results, title="VIIRS Regrid Benchmark", timestamp=ts))
        log.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()

