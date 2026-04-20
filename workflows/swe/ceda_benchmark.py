#!/usr/bin/env python
"""Benchmark CEDA SWE regridding: compare nearest vs bilinear methods."""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from join_scratch.datasets import CedaFileHandler
from join_scratch.regrid.regular_to_regular import compute_weights, load_regridder
from join_scratch.utils import _time_call, BenchmarkResult, render_report
from lis_grid import load_lis_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

CEDA_GLOB = "**/*.nc"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CEDA SWE regridding methods."
    )
    parser.add_argument("--lis-path", required=True, type=Path)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--weights-dir", type=Path, default=Path("_data/ceda"))
    ns = parser.parse_args()

    ceda_files = sorted(ns.input_dir.glob(CEDA_GLOB))
    if not ceda_files:
        raise FileNotFoundError(f"No CEDA NetCDF files found under {ns.input_dir}")
    path = ceda_files[0]
    log.info("Benchmarking on %s", path.name)

    lis_grid = load_lis_grid(ns.lis_path)
    handler = CedaFileHandler.from_path(path)
    ds = handler.get_dataset()
    source_area = handler.get_area_def()

    # Build source grid Dataset for xESMF weight computation
    import xarray as xr
    import numpy as np

    lats = ds["lat"].values if "lat" in ds else source_area.get_lonlats()[1][:, 0]
    lons = ds["lon"].values if "lon" in ds else source_area.get_lonlats()[0][0, :]
    source_grid = xr.Dataset(
        coords={"lat": np.sort(np.unique(lats)), "lon": np.sort(np.unique(lons))}
    )

    # Prepare dataset with lat/lon as dim names for xESMF
    ds_regrid = ds.swap_dims({"y": "lat", "x": "lon"})
    source_shape = (len(source_grid["lat"]), len(source_grid["lon"]))

    results: list[BenchmarkResult] = []

    for method in ("nearest_s2d", "bilinear"):
        weights_path = ns.weights_dir / f"ceda-lis-weights-{method}.nc"
        ns.weights_dir.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
