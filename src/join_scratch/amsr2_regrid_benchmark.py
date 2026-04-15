#!/usr/bin/env python
"""Benchmark regridding approaches for AMSR2 -> LIS grid.

Compares weight-generation (or equivalent setup) time and memory for:
  - xesmf_bilinear:  xESMF bilinear (global AMSR2 source grid)
  - kd_nearest:      pyresample kd_tree nearest-neighbour
  - kd_gauss:        pyresample kd_tree Gaussian-weighted
  - pr_bilinear:     pyresample XArrayBilinearResampler

Notes on excluded algorithms:
  - EWA (Elliptical Weighted Averaging): requires SwathDefinition as source
    (hard isinstance check in DaskEWAResampler.__init__) and AreaDefinition
    as target. Our source is a regular lat/lon grid (not a scan-based swath)
    and our target is an AreaDefinition, so only one constraint would be met.
    EWA is designed for scan-line satellite swath data and is not appropriate
    here.
  - pyresample bilinear with SwathDefinition target: requires AreaDefinition
    as target (needs get_proj_coords()); we use AreaDefinition, so this works.

Measures wall-clock time and peak RSS memory for each strategy and writes
plain-text reports to _reports/ in the project root.
"""

import gc
import logging
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf

from join_scratch.amsr2_regrid import (
    AMSR2,
    AMSR2_GLOB,
    DATA_RAW,
    LIS_PATH,
    build_amsr2_swath_definition,
    build_lis_area_definition,
    load_amsr2,
    load_lis_grid,
    regrid_bilinear_pyresample,
    regrid_kd_gauss,
    regrid_kd_nearest,
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


def _rss_mib() -> float:
    """Return current process RSS memory in MiB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / 1024 / 1024  # macOS: bytes -> MiB
    return usage / 1024  # Linux: KB -> MiB


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    label: str
    source_shape: tuple[int, int]
    elapsed_s: float
    rss_delta_mib: float
    notes: str = ""


# ---------------------------------------------------------------------------
# Individual benchmark runners
# ---------------------------------------------------------------------------


def bench_xesmf(
    amsr2_ds: xr.Dataset,
    lis_grid: xr.Dataset,
) -> BenchmarkResult:
    """xESMF bilinear: measure weight computation time + memory."""
    source_grid = amsr2_ds[["lat", "lon"]]
    ny, nx = int(source_grid.sizes["lat"]), int(source_grid.sizes["lon"])
    log.info("Benchmarking xesmf_bilinear (source %d x %d) …", ny, nx)

    gc.collect()
    rss_before = _rss_mib()
    t0 = time.perf_counter()

    regridder = xesmf.Regridder(
        source_grid,
        lis_grid,
        method="bilinear",
        periodic=True,
    )

    elapsed = time.perf_counter() - t0
    rss_delta = _rss_mib() - rss_before
    nnz = int(regridder.weights.data.nnz)

    log.info("xesmf_bilinear: %.2f s | RSS +%.1f MiB | nnz=%d", elapsed, rss_delta, nnz)
    return BenchmarkResult(
        label="xesmf_bilinear",
        source_shape=(ny, nx),
        elapsed_s=elapsed,
        rss_delta_mib=rss_delta,
        notes=f"nnz weights={nnz:,d}",
    )


def bench_kd_nearest(
    amsr2_ds: xr.Dataset,
    source_def,
    target_def,
) -> BenchmarkResult:
    """pyresample kd_tree nearest-neighbour: measure full resample time + memory."""
    ny, nx = int(amsr2_ds.sizes["lat"]), int(amsr2_ds.sizes["lon"])
    log.info("Benchmarking kd_nearest (source %d x %d) …", ny, nx)

    gc.collect()
    rss_before = _rss_mib()
    t0 = time.perf_counter()

    regrid_kd_nearest(amsr2_ds, source_def, target_def)

    elapsed = time.perf_counter() - t0
    rss_delta = _rss_mib() - rss_before

    log.info("kd_nearest: %.2f s | RSS +%.1f MiB", elapsed, rss_delta)
    return BenchmarkResult(
        label="kd_nearest",
        source_shape=(ny, nx),
        elapsed_s=elapsed,
        rss_delta_mib=rss_delta,
        notes=f"radius={AMSR2.kd_radius_m / 1000:.0f} km",
    )


def bench_kd_gauss(
    amsr2_ds: xr.Dataset,
    source_def,
    target_def,
) -> BenchmarkResult:
    """pyresample kd_tree Gaussian: measure full resample time + memory."""
    ny, nx = int(amsr2_ds.sizes["lat"]), int(amsr2_ds.sizes["lon"])
    log.info("Benchmarking kd_gauss (source %d x %d) …", ny, nx)

    gc.collect()
    rss_before = _rss_mib()
    t0 = time.perf_counter()

    regrid_kd_gauss(amsr2_ds, source_def, target_def)

    elapsed = time.perf_counter() - t0
    rss_delta = _rss_mib() - rss_before

    log.info("kd_gauss: %.2f s | RSS +%.1f MiB", elapsed, rss_delta)
    return BenchmarkResult(
        label="kd_gauss",
        source_shape=(ny, nx),
        elapsed_s=elapsed,
        rss_delta_mib=rss_delta,
        notes=f"radius={AMSR2.kd_radius_m / 1000:.0f} km sigma={AMSR2.gauss_sigma_m / 1000:.0f} km",
    )


def bench_pr_bilinear(
    amsr2_ds: xr.Dataset,
    source_def,
    target_def,
) -> BenchmarkResult:
    """pyresample XArrayBilinearResampler: measure full resample time + memory."""
    ny, nx = int(amsr2_ds.sizes["lat"]), int(amsr2_ds.sizes["lon"])
    log.info("Benchmarking pr_bilinear (source %d x %d) …", ny, nx)

    gc.collect()
    rss_before = _rss_mib()
    t0 = time.perf_counter()

    regrid_bilinear_pyresample(amsr2_ds, source_def, target_def)

    elapsed = time.perf_counter() - t0
    rss_delta = _rss_mib() - rss_before

    log.info("pr_bilinear: %.2f s | RSS +%.1f MiB", elapsed, rss_delta)
    return BenchmarkResult(
        label="pr_bilinear",
        source_shape=(ny, nx),
        elapsed_s=elapsed,
        rss_delta_mib=rss_delta,
        notes=f"radius={AMSR2.kd_radius_m / 1000:.0f} km",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_report(results: list[BenchmarkResult]) -> str:
    col_w = (20, 18, 10, 16, 40)
    header = (
        f"{'Method':<{col_w[0]}} {'Source shape':<{col_w[1]}} "
        f"{'Time (s)':>{col_w[2]}} {'RSS delta (MiB)':>{col_w[3]}} "
        f"{'Notes':<{col_w[4]}}"
    )
    sep = "  ".join("-" * w for w in col_w)

    lines = [
        "AMSR2 Regridding Benchmark",
        "==========================",
        "",
        "Algorithms compared:",
        "  xesmf_bilinear : xESMF bilinear (ESMF weight generation only)",
        "  kd_nearest     : pyresample kd_tree nearest-neighbour (full resample)",
        "  kd_gauss       : pyresample kd_tree Gaussian-weighted (full resample)",
        "  pr_bilinear    : pyresample XArrayBilinearResampler (full resample)",
        "",
        "Excluded algorithms:",
        "  EWA            : requires SwathDefinition source (scan-line swath data only)",
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
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    amsr2_files = sorted(DATA_RAW.glob(AMSR2_GLOB))
    if not amsr2_files:
        raise FileNotFoundError(f"No AMSR2 files found under {DATA_RAW}")

    log.info("Loading data …")
    lis_grid = load_lis_grid(LIS_PATH)
    amsr2_ds = load_amsr2(amsr2_files[0])

    lis_area = build_lis_area_definition(LIS_PATH)
    amsr2_swath = build_amsr2_swath_definition(amsr2_ds)

    results = [
        bench_xesmf(amsr2_ds, lis_grid),
        bench_kd_nearest(amsr2_ds, amsr2_swath, lis_area),
        bench_kd_gauss(amsr2_ds, amsr2_swath, lis_area),
        bench_pr_bilinear(amsr2_ds, amsr2_swath, lis_area),
    ]

    report = render_report(results)
    print(report)

    report_path = REPORTS_DIR / "amsr2_regrid_benchmark.txt"
    report_path.write_text(report)
    log.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
