#!/usr/bin/env python
"""Benchmark regridding approaches for CEDA ESA CCI SWE -> LIS grid.

Compares wall-clock time and RSS memory for:
  - xesmf_bilinear       : xESMF bilinear (weight generation; weights cached to file)
  - nearest (cold/warm)  : satpy KDTreeResampler nearest-neighbour
  - bilinear (cold/warm) : satpy BilinearResampler
  - ewa                  : satpy DaskEWAResampler (Elliptical Weighted Averaging)
  - bucket_avg           : satpy BucketAvg (scatter-add average)

Writes a plain-text report to _reports/ceda_regrid_benchmark.txt.
"""

import argparse
import gc
import logging
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import xarray as xr

from join_scratch.amsr2.amsr2_regrid import build_lis_area_definition, load_lis_grid
from join_scratch.ceda.ceda_regrid import (
    CEDA_GLOB,
    SATPY_CACHE,
    WEIGHTS_PATH,
    build_ceda_swath_definition,
    compute_weights,
    load_ceda,
    load_regridder,
    regrid_bilinear,
    regrid_bucket_avg,
    regrid_ewa,
    regrid_nearest,
)
from join_scratch.storage import (
    add_storage_args,
    storage_config_from_namespace,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = ROOT / "_reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rss_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / 1024 / 1024
    return usage / 1024


@dataclass
class BenchmarkResult:
    label: str
    source_shape: tuple[int, int]
    elapsed_s: float
    rss_delta_mib: float
    notes: str = ""


def _time_call(fn, *args, **kwargs) -> tuple[float, float]:
    gc.collect()
    rss_before = _rss_mib()
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0, _rss_mib() - rss_before


# ---------------------------------------------------------------------------
# Individual benchmark runners
# ---------------------------------------------------------------------------


def bench_xesmf(ceda_ds: xr.Dataset, lis_grid: xr.Dataset) -> BenchmarkResult:
    source_grid = ceda_ds[["lat", "lon"]]
    ny, nx = int(source_grid.sizes["lat"]), int(source_grid.sizes["lon"])
    log.info("Benchmarking xesmf_bilinear (source %d x %d) …", ny, nx)

    elapsed, rss_delta = _time_call(
        compute_weights, source_grid, lis_grid, WEIGHTS_PATH
    )

    regridder = load_regridder(source_grid, lis_grid, WEIGHTS_PATH)
    nnz = int(regridder.weights.data.nnz)

    log.info("xesmf_bilinear: %.2f s | RSS +%.1f MiB | nnz=%d", elapsed, rss_delta, nnz)
    return BenchmarkResult(
        label="xesmf_bilinear",
        source_shape=(ny, nx),
        elapsed_s=elapsed,
        rss_delta_mib=rss_delta,
        notes=f"nnz weights={nnz:,d}",
    )


def _bench_satpy(
    label: str,
    fn,
    ceda_ds: xr.Dataset,
    source_def,
    target_def,
    cache_dir: Path,
    has_cache: bool,
) -> BenchmarkResult:
    ny, nx = int(ceda_ds.sizes["lat"]), int(ceda_ds.sizes["lon"])
    cache_tag = "warm" if has_cache else "cold"
    log.info("Benchmarking %s (%s cache, source %d x %d) …", label, cache_tag, ny, nx)
    elapsed, rss_delta = _time_call(fn, ceda_ds, source_def, target_def, cache_dir)
    log.info("%s (%s): %.2f s | RSS +%.1f MiB", label, cache_tag, elapsed, rss_delta)
    return BenchmarkResult(
        label=f"{label} ({cache_tag})",
        source_shape=(ny, nx),
        elapsed_s=elapsed,
        rss_delta_mib=rss_delta,
        notes=f"{cache_tag} cache",
    )


def bench_ewa(ceda_ds: xr.Dataset, source_def, target_def) -> BenchmarkResult:
    ny, nx = int(ceda_ds.sizes["lat"]), int(ceda_ds.sizes["lon"])
    log.info("Benchmarking ewa (source %d x %d) …", ny, nx)
    elapsed, rss_delta = _time_call(regrid_ewa, ceda_ds, source_def, target_def)
    log.info("ewa: %.2f s | RSS +%.1f MiB", elapsed, rss_delta)
    return BenchmarkResult(
        label="ewa",
        source_shape=(ny, nx),
        elapsed_s=elapsed,
        rss_delta_mib=rss_delta,
        notes="no caching",
    )


def bench_bucket_avg(ceda_ds: xr.Dataset, source_def, target_def) -> BenchmarkResult:
    ny, nx = int(ceda_ds.sizes["lat"]), int(ceda_ds.sizes["lon"])
    log.info("Benchmarking bucket_avg (source %d x %d) …", ny, nx)
    elapsed, rss_delta = _time_call(regrid_bucket_avg, ceda_ds, source_def, target_def)
    log.info("bucket_avg: %.2f s | RSS +%.1f MiB", elapsed, rss_delta)
    return BenchmarkResult(
        label="bucket_avg",
        source_shape=(ny, nx),
        elapsed_s=elapsed,
        rss_delta_mib=rss_delta,
        notes="no caching",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_report(results: list[BenchmarkResult]) -> str:
    col_w = (28, 18, 10, 16, 40)
    header = (
        f"{'Method':<{col_w[0]}} {'Source shape':<{col_w[1]}} "
        f"{'Time (s)':>{col_w[2]}} {'RSS delta (MiB)':>{col_w[3]}} "
        f"{'Notes':<{col_w[4]}}"
    )
    sep = "  ".join("-" * w for w in col_w)

    lines = [
        "CEDA Regridding Benchmark",
        "=========================",
        "",
        "Algorithms compared (all via satpy unless noted):",
        "  xesmf_bilinear       : xESMF bilinear (ESMF weight generation only)",
        "  nearest (cold/warm)  : satpy KDTreeResampler nearest-neighbour",
        "  bilinear (cold/warm) : satpy BilinearResampler",
        "  ewa                  : satpy DaskEWAResampler (Elliptical Weighted Averaging; no caching)",
        "  bucket_avg           : satpy BucketAvg (scatter-add average; no caching)",
        "",
        "cold = cache computed and written to disk on this run",
        "warm = cache reloaded from disk (no recomputation)",
        "",
        "Caching notes:",
        "  nearest/bilinear write zarr files to _data/ceda/satpy-cache/",
        "  EWA and BucketAvg have no precomputed index structures to cache",
        "  xESMF writes a NetCDF weights file to _data/ceda/ceda-lis-weights.nc",
        "",
        header,
        sep,
    ]

    for r in results:
        shape_str = f"{r.source_shape[0]} x {r.source_shape[1]}"
        lines.append(
            f"{r.label:<{col_w[0]}} {shape_str:<{col_w[1]}} "
            f"{r.elapsed_s:>{col_w[2]}.2f} {r.rss_delta_mib:>{col_w[3]}.1f} "
            f"{r.notes:<{col_w[4]}}"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CEDA ESA CCI SWE regridding approaches."
    )
    add_storage_args(parser)
    ns = parser.parse_args()
    storage = storage_config_from_namespace(ns)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ceda_files = storage.glob(CEDA_GLOB)
    if not ceda_files:
        raise FileNotFoundError(f"No CEDA files found in {storage.storage_location}")

    log.info("Loading data …")
    lis_grid = load_lis_grid(storage)
    ceda_ds = load_ceda(ceda_files[0], storage)
    lis_area = build_lis_area_definition(storage)
    ceda_swath = build_ceda_swath_definition(ceda_ds)

    bench_cache = SATPY_CACHE.parent / "satpy-cache-bench"
    if bench_cache.exists():
        shutil.rmtree(bench_cache)
    bench_cache.mkdir(parents=True)

    try:
        results = [bench_xesmf(ceda_ds, lis_grid)]

        for label, fn in [
            ("nearest", regrid_nearest),
            ("bilinear", regrid_bilinear),
        ]:
            results.append(
                _bench_satpy(
                    label,
                    fn,
                    ceda_ds,
                    ceda_swath,
                    lis_area,
                    bench_cache,
                    has_cache=False,
                )
            )
            results.append(
                _bench_satpy(
                    label,
                    fn,
                    ceda_ds,
                    ceda_swath,
                    lis_area,
                    bench_cache,
                    has_cache=True,
                )
            )

        results.append(bench_ewa(ceda_ds, ceda_swath, lis_area))
        results.append(bench_bucket_avg(ceda_ds, ceda_swath, lis_area))
    finally:
        shutil.rmtree(bench_cache, ignore_errors=True)

    report = render_report(results)
    print(report)

    report_path = REPORTS_DIR / "ceda_regrid_benchmark.txt"
    report_path.write_text(report)
    log.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
