#!/usr/bin/env python
"""Benchmark weight-generation speed and memory: global AMSR2 grid vs subsetted.

Compares two strategies for building xESMF bilinear weights:
  - global:   pass the full 1800x3600 AMSR2 equirectangular grid
  - subsetted: pre-clip AMSR2 to the LIS bounding box (+ 1° buffer)

Measures wall-clock time and peak RSS memory for each strategy and writes
plain-text reports to _reports/ in the project root.
"""

import gc
import logging
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf

from join_scratch.amsr2_regrid import (
    DATA_RAW,
    LIS_PATH,
    _AMSR2_LAT,
    _AMSR2_LON,
    AMSR2_GLOB,
    load_amsr2,
    load_lis_grid,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "_reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    """Return current RSS memory usage in MiB."""
    # getrusage returns bytes on Linux, kilobytes on macOS
    import sys

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / 1024 / 1024  # bytes -> MiB
    return usage / 1024  # KB -> MiB


def _subset_grid(ds: xr.Dataset, lis_grid: xr.Dataset) -> xr.Dataset:
    """Clip *ds* to the LIS bounding box with a 1° buffer."""
    lat_min = float(np.floor(lis_grid["lat"].min()))
    lat_max = float(np.ceil(lis_grid["lat"].max()))
    lon_min = float(np.floor(lis_grid["lon"].min()))
    lon_max = float(np.ceil(lis_grid["lon"].max()))
    return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))


# ---------------------------------------------------------------------------
# Benchmark dataclass & runner
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    label: str
    source_shape: tuple[int, int]
    elapsed_s: float
    peak_rss_mib: float
    nnz_weights: int


def run_benchmark(
    label: str,
    source_grid: xr.Dataset,
    target_grid: xr.Dataset,
) -> BenchmarkResult:
    """Build a fresh xESMF regridder and record time + peak memory."""
    log.info("Running benchmark: %s (source shape %s)", label, dict(source_grid.sizes))

    gc.collect()
    rss_before = _rss_mb()

    t0 = time.perf_counter()
    regridder = xesmf.Regridder(
        source_grid,
        target_grid,
        method="bilinear",
        periodic=True,
    )
    elapsed = time.perf_counter() - t0

    rss_after = _rss_mb()
    peak_rss = rss_after - rss_before

    # Number of non-zero weights in the sparse weight matrix
    nnz = int(regridder.weights.data.nnz)

    ny = int(source_grid.sizes["lat"])
    nx = int(source_grid.sizes["lon"])

    log.info(
        "%s: %.2f s | RSS delta %.1f MiB | nnz weights %d",
        label,
        elapsed,
        peak_rss,
        nnz,
    )

    return BenchmarkResult(
        label=label,
        source_shape=(ny, nx),
        elapsed_s=elapsed,
        peak_rss_mib=peak_rss,
        nnz_weights=nnz,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_report(results: list[BenchmarkResult]) -> str:
    lines = [
        "AMSR2 Weight-Generation Benchmark",
        "==================================",
        "",
        f"{'Label':<12} {'Source shape':<20} {'Time (s)':>10} {'RSS delta (MiB)':>16} {'NNZ weights':>12}",
        f"{'-' * 12} {'-' * 20} {'-' * 10} {'-' * 16} {'-' * 12}",
    ]
    for r in results:
        shape_str = f"{r.source_shape[0]} x {r.source_shape[1]}"
        lines.append(
            f"{r.label:<12} {shape_str:<20} {r.elapsed_s:>10.2f} {r.peak_rss_mib:>16.1f} {r.nnz_weights:>12d}"
        )

    if len(results) == 2:
        a, b = results[0], results[1]
        lines += [
            "",
            "Comparison (global vs subsetted)",
            "---------------------------------",
            f"  Time speedup:        {a.elapsed_s / b.elapsed_s:.2f}x  ({a.elapsed_s:.2f}s -> {b.elapsed_s:.2f}s)",
            f"  RSS delta reduction: {a.peak_rss_mib:.1f} MiB -> {b.peak_rss_mib:.1f} MiB",
            f"  NNZ weights:         {a.nnz_weights:,d} vs {b.nnz_weights:,d}  (identical = {a.nnz_weights == b.nnz_weights})",
        ]

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lis_grid = load_lis_grid(LIS_PATH)

    amsr2_files = sorted(DATA_RAW.glob(AMSR2_GLOB))
    if not amsr2_files:
        raise FileNotFoundError(f"No AMSR2 files found under {DATA_RAW}")

    # Load the first AMSR2 file (global)
    amsr2_ds = load_amsr2(amsr2_files[0])
    global_grid = amsr2_ds[["lat", "lon"]]
    subsetted_grid = _subset_grid(global_grid, lis_grid)

    results = [
        run_benchmark("global", global_grid, lis_grid),
        run_benchmark("subsetted", subsetted_grid, lis_grid),
    ]

    report = render_report(results)
    print(report)

    report_path = REPORTS_DIR / "amsr2_regrid_benchmark.txt"
    report_path.write_text(report)
    log.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
