#!/usr/bin/env python
"""Benchmark ICESat-2 ATL06 fetch + regridding to the LIS model grid.

Measures two sequential stages:
  1. SlideRule fetch  – query SlideRule for ATL06 data over the LIS domain.
  2. Regrid           – grid the resulting 1-D point cloud onto the LIS pixel grid
                        using the pixel-binning mean method.

Use --no-cache to force a fresh SlideRule download (ignoring any local Parquet
cache) and include the fetch timing in the benchmark report.  Without
--no-cache the fetch stage is skipped if a valid cache file exists, and only
the regrid stage is benchmarked.
"""

import sys
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import geopandas as gpd
import xarray as xr

from join_scratch.datasets import Icesat2FileHandler
from join_scratch.regrid import regrid
from join_scratch.utils import _time_call, BenchmarkResult, render_report
from lis_grid import build_lis_area_definition, load_lis_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# Default temporal search window (one week of early-2019 data)
T0_DEFAULT = "2019-01-01T00:00:00Z"
T1_DEFAULT = "2019-01-07T23:59:59Z"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ICESat-2 ATL06 fetch + regridding to the LIS model grid."
    )
    parser.add_argument("--lis-path", required=True, type=Path)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("_data/icesat2/atl06_benchmark.parquet"),
        help="Path to the cached ATL06 Parquet file (default: _data/icesat2/atl06_benchmark.parquet).",
    )
    parser.add_argument(
        "--t0",
        default=T0_DEFAULT,
        help=f"Start time for SlideRule query (ISO 8601, default: {T0_DEFAULT}).",
    )
    parser.add_argument(
        "--t1",
        default=T1_DEFAULT,
        help=f"End time for SlideRule query (ISO 8601, default: {T1_DEFAULT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write timestamped report file (default: print to stdout only).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help=(
            "Force re-download from SlideRule even if a local Parquet cache exists, "
            "and include the fetch timing in the report."
        ),
    )
    ns = parser.parse_args()

    lis_grid = load_lis_grid(ns.lis_path)
    lis_area = build_lis_area_definition(ns.lis_path)

    # Build LIS bounding polygon for SlideRule query
    lat = lis_grid["lat"].values
    lon = lis_grid["lon"].values
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    polygon = [
        {"lon": lon_min, "lat": lat_min},
        {"lon": lon_max, "lat": lat_min},
        {"lon": lon_max, "lat": lat_max},
        {"lon": lon_min, "lat": lat_max},
        {"lon": lon_min, "lat": lat_min},
    ]

    parms: dict = {
        "poly": polygon,
        "t0": ns.t0,
        "t1": ns.t1,
    }

    ns.cache_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []

    # ── Stage 1: SlideRule fetch ──────────────────────────────────────────────
    if ns.no_cache:
        if ns.cache_path.exists():
            ns.cache_path.unlink()
            log.info("--no-cache: removed existing Parquet cache %s", ns.cache_path)

        log.info("Initialising SlideRule client …")
        from sliderule import sliderule as _sliderule
        _sliderule.init()

        cache_path = ns.cache_path

        def _fetch_and_cache() -> None:
            gdf: gpd.GeoDataFrame = _sliderule.run("atl06x", parms)
            log.info("SlideRule returned %d ATL06 observations", len(gdf))
            gdf.to_parquet(cache_path)

        fetch_elapsed, fetch_rss = _time_call(_fetch_and_cache)
        log.info("SlideRule fetch: %.2f s, %.1f MiB", fetch_elapsed, fetch_rss)

        try:
            gdf_for_shape = gpd.read_parquet(ns.cache_path)
            n_obs = len(gdf_for_shape)
        except Exception:
            n_obs = 0

        results.append(
            BenchmarkResult(
                label="ICESat-2 SlideRule fetch",
                source_shape=(n_obs, 1),
                elapsed_s=fetch_elapsed,
                rss_delta_mib=fetch_rss,
                notes=f"t0={ns.t0[:10]}, t1={ns.t1[:10]}",
            )
        )
    else:
        if not ns.cache_path.exists():
            raise FileNotFoundError(
                f"No cached ATL06 data at {ns.cache_path}. "
                "Run with --no-cache to fetch from SlideRule first."
            )
        log.info("Using cached ATL06 data from %s", ns.cache_path)

    # ── Stage 2: Regrid ───────────────────────────────────────────────────────
    handler = Icesat2FileHandler.from_path(ns.cache_path)
    da = handler.get_dataset()
    source_area = da.attrs["area"]
    n_obs = int(len(da))
    source_shape = (n_obs, 1)

    regrid_elapsed, regrid_rss = _time_call(regrid, da, source_area, lis_area, method="mean")
    log.info("Regrid: %.2f s, %.1f MiB", regrid_elapsed, regrid_rss)

    results.append(
        BenchmarkResult(
            label="ICESat-2 regrid (mean)",
            source_shape=source_shape,
            elapsed_s=regrid_elapsed,
            rss_delta_mib=regrid_rss,
            notes=f"{n_obs} ATL06 points → LIS 1km grid",
        )
    )

    print(render_report(results, title="ICESat-2 ATL06 Benchmark"))

    if ns.output_dir is not None:
        ts = datetime.now(tz=timezone.utc)
        ns.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = ns.output_dir / f"icesat2_benchmark_{ts.strftime('%Y%m%dT%H%M%SZ')}.txt"
        report_path.write_text(render_report(results, title="ICESat-2 ATL06 Benchmark", timestamp=ts))
        log.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
