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
from pyresample.geometry import SwathDefinition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

VIIRS_GLOB = "**/*.h5"
METHODS = ["nearest", "bilinear", "ewa", "bucket_avg"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark VIIRS CGF Snow Cover regridding methods."
    )
    parser.add_argument("--lis-path", required=True, type=Path)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
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
            "VIIRS uses no weights files; this flag clears the pyresample resampler "
            "cache between runs so each method is benchmarked cold. "
            "Has no effect on the output report format."
        ),
    )
    ns = parser.parse_args()

    viirs_files = sorted(ns.input_dir.glob(VIIRS_GLOB))
    if not viirs_files:
        raise FileNotFoundError(f"No VIIRS HDF5 files found under {ns.input_dir}")

    lis_area = build_lis_area_definition(ns.lis_path)

    # Group by date and take the first date only
    date_groups: dict[str, list[Path]] = defaultdict(list)
    for p in viirs_files:
        parts = p.stem.split(".")
        date_key = parts[1] if len(parts) > 1 else p.stem
        date_groups[date_key].append(p)

    date_key, paths = next(iter(sorted(date_groups.items())))
    log.info("Benchmarking on date %s (%d tile(s))", date_key, len(paths))

    # Load and composite tiles
    all_data, all_lons, all_lats = [], [], []
    for path in paths:
        handler = ViirsFileHandler.from_path(path)
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
