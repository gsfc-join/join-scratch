#!/usr/bin/env python
"""Benchmark regridding approaches for VIIRS -> LIS grid.

Compares wall-clock time and RSS memory for the four satpy methods:
  - nearest        : satpy KDTreeResampler nearest-neighbour
  - bilinear       : satpy BilinearResampler
  - ewa            : satpy DaskEWAResampler (Elliptical Weighted Averaging)
  - bucket_avg     : satpy BucketAvg (scatter-add average)

For nearest and bilinear, timing is reported for both the cold run (cache
written to disk) and a warm run (cache reloaded from disk).

xESMF is not used for VIIRS because the sinusoidal-tile source grid is not a
regular lon/lat grid — xESMF requires a rectilinear source.

Writes a plain-text report to _reports/viirs_regrid_benchmark.txt.
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

import re

from join_scratch.amsr2.amsr2_regrid import build_lis_area_definition
from join_scratch.storage import (
    StorageConfig,
    add_storage_args,
    storage_config_from_namespace,
)
from join_scratch.viirs.viirs_regrid import (
    SATPY_CACHE,
    VIIRS_GLOB,
    _lis_lonlat_bounds,
    build_viirs_domain_mapping,
    build_viirs_swath_definition,
    filter_tiles_by_domain,
    load_viirs_tiles,
    regrid_bilinear,
    regrid_bucket_avg,
    regrid_ewa,
    regrid_nearest,
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
    """Return current process RSS memory in MiB."""
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


def _bench_satpy(
    label: str,
    fn,
    tile: dict,
    source_def,
    target_def,
    cache_dir: Path,
    has_cache: bool,
) -> BenchmarkResult:
    ny, nx = tile["data"].shape
    cache_tag = "warm" if has_cache else "cold"
    log.info("Benchmarking %s (%s cache, source %d x %d) …", label, cache_tag, ny, nx)
    elapsed, rss_delta = _time_call(fn, tile, source_def, target_def, cache_dir)
    log.info("%s (%s): %.2f s | RSS +%.1f MiB", label, cache_tag, elapsed, rss_delta)
    return BenchmarkResult(
        label=f"{label} ({cache_tag})",
        source_shape=(ny, nx),
        elapsed_s=elapsed,
        rss_delta_mib=rss_delta,
        notes=f"{cache_tag} cache",
    )


def bench_ewa(tile: dict, source_def, target_def) -> BenchmarkResult:
    ny, nx = tile["data"].shape
    log.info("Benchmarking ewa (source %d x %d) …", ny, nx)
    elapsed, rss_delta = _time_call(regrid_ewa, tile, source_def, target_def)
    log.info("ewa: %.2f s | RSS +%.1f MiB", elapsed, rss_delta)
    return BenchmarkResult(
        label="ewa",
        source_shape=(ny, nx),
        elapsed_s=elapsed,
        rss_delta_mib=rss_delta,
        notes="no caching",
    )


def bench_bucket_avg(tile: dict, source_def, target_def) -> BenchmarkResult:
    ny, nx = tile["data"].shape
    log.info("Benchmarking bucket_avg (source %d x %d) …", ny, nx)
    elapsed, rss_delta = _time_call(regrid_bucket_avg, tile, source_def, target_def)
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


def render_report(
    results: list[BenchmarkResult],
    mapping_elapsed_s: float,
    mapping_shape: tuple[int, int],
    mapping_n_valid: int,
    date_key: str,
    n_tiles: int,
) -> str:
    col_w = (28, 18, 10, 16, 40)
    header = (
        f"{'Method':<{col_w[0]}} {'Source shape':<{col_w[1]}} "
        f"{'Time (s)':>{col_w[2]}} {'RSS delta (MiB)':>{col_w[3]}} "
        f"{'Notes':<{col_w[4]}}"
    )
    sep = "  ".join("-" * w for w in col_w)

    lines = [
        "VIIRS Regridding Benchmark",
        "==========================",
        "",
        f"Date:        {date_key}",
        f"Tiles loaded: {n_tiles}",
        "",
        "Algorithms compared (all via satpy):",
        "  nearest (cold/warm)  : satpy KDTreeResampler nearest-neighbour",
        "  bilinear (cold/warm) : satpy BilinearResampler",
        "  ewa                  : satpy DaskEWAResampler (Elliptical Weighted Averaging; no caching)",
        "  bucket_avg           : satpy BucketAvg (scatter-add average; no caching)",
        "",
        "Note: xESMF is not applicable because the VIIRS sinusoidal tile",
        "source grid is not a rectilinear lon/lat grid.",
        "",
        "cold = cache computed and written to disk on this run",
        "warm = cache reloaded from disk (no recomputation)",
        "",
        "Caching notes:",
        "  nearest/bilinear write zarr files to _data/viirs/satpy-cache/",
        "  EWA and BucketAvg have no precomputed index structures to cache",
        "",
        "--- Domain mapping (build_viirs_domain_mapping) ---",
        f"  Grid shape : {mapping_shape[0]} rows × {mapping_shape[1]} cols",
        f"  Valid pixels: {mapping_n_valid:,}",
        f"  Elapsed    : {mapping_elapsed_s:.3f} s",
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
    parser = argparse.ArgumentParser(description="Benchmark VIIRS regridding methods.")
    add_storage_args(parser)
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "Process only this date key (e.g. 'A2019001').  "
            "Defaults to the first date found in the glob results."
        ),
    )
    args = parser.parse_args()
    storage = storage_config_from_namespace(args)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Build LIS area and domain mapping (timed) ---
    lis_area = build_lis_area_definition(storage)
    lon_min, lat_min, lon_max, lat_max = _lis_lonlat_bounds(lis_area)

    log.info("Building VIIRS domain mapping …")
    t0 = time.perf_counter()
    mapping = build_viirs_domain_mapping((lon_min, lat_min, lon_max, lat_max))
    mapping_elapsed_s = time.perf_counter() - t0
    mapping_shape = (mapping.dims["y"], mapping.dims["x"])
    mapping_n_valid = int((mapping["tile"] != "").values.sum())
    log.info(
        "Domain mapping built in %.3f s (%d×%d grid, %d valid pixels)",
        mapping_elapsed_s,
        mapping_shape[0],
        mapping_shape[1],
        mapping_n_valid,
    )

    # --- Glob and filter to LIS domain ---
    viirs_files = storage.glob(VIIRS_GLOB)
    if not viirs_files:
        raise FileNotFoundError(f"No VIIRS files found with glob {VIIRS_GLOB!r}")

    log.info("Found %d VIIRS file(s); filtering to LIS domain …", len(viirs_files))
    viirs_files = filter_tiles_by_domain(viirs_files, lis_area)
    if not viirs_files:
        raise FileNotFoundError("No VIIRS tiles overlap the LIS domain.")

    # --- Group by date, pick one day ---
    _hv_date_re = re.compile(r"\.(A\d{7})\.")
    date_groups: dict[str, list[str]] = {}
    for path in viirs_files:
        m = _hv_date_re.search(path)
        if m:
            date_groups.setdefault(m.group(1), []).append(path)

    if args.date:
        if args.date not in date_groups:
            raise ValueError(
                f"Date {args.date!r} not found. Available: {sorted(date_groups)}"
            )
        date_key = args.date
    else:
        date_key = sorted(date_groups)[0]

    day_files = date_groups[date_key]
    log.info("Benchmarking date %s (%d tile(s)) …", date_key, len(day_files))

    # --- Load tiles and build swath ---
    tile = load_viirs_tiles(day_files, storage)
    source_def = build_viirs_swath_definition(tile)

    # Temporary cache dir — always start cold for the benchmark
    bench_cache = SATPY_CACHE.parent / "satpy-cache-bench"
    if bench_cache.exists():
        shutil.rmtree(bench_cache)
    bench_cache.mkdir(parents=True)

    try:
        results = []

        for label, fn in [
            ("nearest", regrid_nearest),
            ("bilinear", regrid_bilinear),
        ]:
            results.append(
                _bench_satpy(
                    label, fn, tile, source_def, lis_area, bench_cache, has_cache=False
                )
            )
            results.append(
                _bench_satpy(
                    label, fn, tile, source_def, lis_area, bench_cache, has_cache=True
                )
            )

        results.append(bench_ewa(tile, source_def, lis_area))
        results.append(bench_bucket_avg(tile, source_def, lis_area))
    finally:
        shutil.rmtree(bench_cache, ignore_errors=True)

    report = render_report(
        results,
        mapping_elapsed_s=mapping_elapsed_s,
        mapping_shape=mapping_shape,
        mapping_n_valid=mapping_n_valid,
        date_key=date_key,
        n_tiles=len(day_files),
    )
    print(report)

    report_path = REPORTS_DIR / "viirs_regrid_benchmark.txt"
    report_path.write_text(report)
    log.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
