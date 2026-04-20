#!/usr/bin/env python
"""Benchmark AMSR2 regridding: compare nearest vs bilinear methods."""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from join_scratch.datasets import Amsr2FileHandler
from join_scratch.regrid.regular_to_regular import compute_weights, load_regridder
from join_scratch.utils import _time_call, BenchmarkResult, render_report
from lis_grid import load_lis_grid

import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

AMSR2_GLOB = "**/*.h5"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AMSR2 regridding methods.")
    parser.add_argument("--lis-path", required=True, type=Path)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--weights-dir", type=Path, default=Path("_data/amsr2"))
    ns = parser.parse_args()

    amsr2_files = sorted(ns.input_dir.glob(AMSR2_GLOB))
    if not amsr2_files:
        raise FileNotFoundError(f"No AMSR2 HDF5 files found under {ns.input_dir}")
    path = amsr2_files[0]
    log.info("Benchmarking on %s", path.name)

    lis_grid = load_lis_grid(ns.lis_path)
    handler = Amsr2FileHandler.from_path(path)

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


if __name__ == "__main__":
    main()
