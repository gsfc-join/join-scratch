#!/usr/bin/env python
"""Full SWE workflow: regrid all input datasets to the LIS grid and combine.

Regrids each of the following datasets onto the LIS 1 km Lambert Conformal grid
and writes all outputs into a single NetCDF file with informatively named variables:

  Dataset              Variable(s)           Default method    Output variable(s)
  ─────────────────────────────────────────────────────────────────────────────────
  AMSR2 snow depth     Geophysical Data[0]   bilinear          amsr2_snow_depth_mean
                       Geophysical Data[1]   bilinear          amsr2_snow_depth_uncertainty
  CEDA ESA CCI SWE     swe                   bilinear          ceda_swe
                       swe_std               bilinear          ceda_swe_std
  VIIRS CGF snow cover CGF_NDSI_Snow_Cover   nearest           viirs_cgf_ndsi_snow_cover
  ICESat-2 ATL06       h_li                  mean              icesat2_h_li

For AMSR2 ``inner`` dimension (phony_dim_2): index 0 = ascending-pass mean value,
index 1 = descending-pass uncertainty, per dataset documentation.

Paths may be local file-system paths or ``s3://`` URIs — detection is automatic.

Usage example (local)
─────────────────────
    python full-workflow.py \\
        --lis-path /data/lis_input_NMP_1000m_missouri.nc \\
        --amsr2-dir /data/amsr2 \\
        --ceda-dir  /data/ceda \\
        --viirs-dir /data/viirs \\
        --icesat2-parquet /data/icesat2/atl06.parquet \\
        --output-path /data/swe_combined.nc

Usage example (S3)
──────────────────
    python full-workflow.py \\
        --lis-path s3://my-bucket/lis_input_NMP_1000m_missouri.nc \\
        --amsr2-dir s3://my-bucket/amsr2 \\
        --ceda-dir  s3://my-bucket/ceda \\
        --viirs-dir s3://my-bucket/viirs \\
        --icesat2-parquet s3://my-bucket/icesat2/atl06.parquet \\
        --output-path /data/swe_combined.nc
"""

import sys
import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from join_scratch.datasets import Amsr2FileHandler, CedaFileHandler, Icesat2FileHandler, ViirsFileHandler
from join_scratch.regrid import regrid
from join_scratch.regrid.regular_to_regular import compute_weights, load_regridder
from lis_grid import build_lis_area_definition, load_lis_grid
from s3_utils import _is_s3, make_fs, make_store, list_s3, handler_from_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

AMSR2_GLOB = "**/*.h5"
CEDA_GLOB = "**/*.nc"
VIIRS_GLOB = "**/*.h5"


# ── path helpers ──────────────────────────────────────────────────────────────

def _list_files(dir_path: str, suffix: str, fs=None) -> list[str]:
    """List files under *dir_path* matching *suffix*, local or S3."""
    if _is_s3(dir_path):
        # Parse bucket and prefix from s3://bucket/prefix
        without_scheme = dir_path[len("s3://"):]
        bucket, _, prefix = without_scheme.partition("/")
        store = make_store(bucket, prefix=prefix)
        keys = list_s3(store)
        prefix_slash = prefix.rstrip("/") + "/" if prefix else ""
        urls = [f"s3://{bucket}/{prefix_slash}{k}" for k in keys if k.endswith(suffix)]
        return sorted(urls)
    else:
        return [str(p) for p in sorted(Path(dir_path).glob(f"**/*{suffix}"))]


def _path_name(path_str: str) -> str:
    """Return a short display name for a path (last component)."""
    return path_str.rstrip("/").split("/")[-1]


def _ensure_local_dir(path: str) -> Path:
    """Ensure a local directory exists and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── helpers ───────────────────────────────────────────────────────────────────

def _regrid_amsr2(
    input_dir: str,
    lis_grid: xr.Dataset,
    method: str,
    weights_dir: str,
    overwrite_weights: bool = False,
    fs=None,
) -> dict[str, xr.DataArray]:
    """Regrid the first AMSR2 file found and return {var_name: DataArray}."""
    files = _list_files(input_dir, ".h5", fs=fs)
    if not files:
        log.warning("No AMSR2 files found under %s — skipping", input_dir)
        return {}
    path = files[0]
    log.info("AMSR2: using %s", _path_name(path))

    if _is_s3(path):
        handler = handler_from_s3(Amsr2FileHandler, path, fs=fs)
    else:
        handler = Amsr2FileHandler.from_path(path)

    lat_asc = np.sort(handler._lat)
    lon_asc = np.sort(handler._lon)
    source_grid = xr.Dataset(coords={"lat": lat_asc, "lon": lon_asc})

    weights_local_dir = _ensure_local_dir(weights_dir)
    weights_path = weights_local_dir / f"amsr2-lis-weights-{method}.nc"
    compute_weights(source_grid, lis_grid, weights_path, method=method, overwrite=overwrite_weights)
    regridder = load_regridder(source_grid, lis_grid, weights_path, method=method)

    da = handler.get_dataset().rename({"y": "lat", "x": "lon"}).sortby("lat")

    # AMSR2 inner dim: index 0 = mean, index 1 = uncertainty
    da_mean  = da.isel(inner=0).drop_vars("inner", errors="ignore")
    da_unc   = da.isel(inner=1).drop_vars("inner", errors="ignore")

    ds_mean = da_mean.to_dataset(name="Geophysical Data")
    ds_unc  = da_unc.to_dataset(name="Geophysical Data")

    log.info("AMSR2: regridding mean (inner=0) …")
    rg_mean = regridder(ds_mean)["Geophysical Data"]

    log.info("AMSR2: regridding uncertainty (inner=1) …")
    rg_unc  = regridder(ds_unc)["Geophysical Data"]

    def _da(arr, long_name, units):
        return xr.DataArray(
            arr.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={"long_name": long_name, "units": units, "source": _path_name(path)},
        )

    return {
        "amsr2_snow_depth_mean": _da(
            rg_mean,
            "AMSR2 snow depth (ascending-pass mean)",
            "mm",
        ),
        "amsr2_snow_depth_uncertainty": _da(
            rg_unc,
            "AMSR2 snow depth uncertainty (descending-pass value)",
            "mm",
        ),
    }


def _regrid_ceda(
    input_dir: str,
    lis_grid: xr.Dataset,
    method: str,
    weights_dir: str,
    overwrite_weights: bool = False,
    fs=None,
) -> dict[str, xr.DataArray]:
    """Regrid the first CEDA file found and return {var_name: DataArray}."""
    files = _list_files(input_dir, ".nc", fs=fs)
    if not files:
        log.warning("No CEDA files found under %s — skipping", input_dir)
        return {}
    path = files[0]
    log.info("CEDA: using %s", _path_name(path))

    if _is_s3(path):
        handler = handler_from_s3(CedaFileHandler, path, fs=fs)
    else:
        handler = CedaFileHandler.from_path(path)

    ds = handler.get_dataset()

    lat_vals = ds["lat"].values if "lat" in ds else ds["y"].values
    lon_vals = ds["lon"].values if "lon" in ds else ds["x"].values
    if lat_vals.ndim == 2:
        lat_vals = lat_vals[:, 0]
    if lon_vals.ndim == 2:
        lon_vals = lon_vals[0, :]
    source_grid = xr.Dataset(
        coords={"lat": np.sort(np.unique(lat_vals)), "lon": np.sort(np.unique(lon_vals))}
    )

    weights_local_dir = _ensure_local_dir(weights_dir)
    weights_path = weights_local_dir / f"ceda-lis-weights-{method}.nc"
    compute_weights(source_grid, lis_grid, weights_path, method=method, overwrite=overwrite_weights)
    regridder = load_regridder(source_grid, lis_grid, weights_path, method=method)

    # Restore lat/lon as dim names for xESMF
    ds_xesmf = ds.swap_dims({"y": "lat", "x": "lon"})

    log.info("CEDA: regridding swe and swe_std …")
    rg = regridder(ds_xesmf)

    def _da(arr, long_name, units):
        return xr.DataArray(
            arr.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={"long_name": long_name, "units": units, "source": _path_name(path)},
        )

    return {
        "ceda_swe": _da(rg["swe"], "CEDA ESA CCI snow water equivalent", "mm"),
        "ceda_swe_std": _da(rg["swe_std"], "CEDA ESA CCI SWE standard deviation", "mm"),
    }


def _viirs_tile_bbox(h: int, v: int) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) for a MODIS/VIIRS h/v tile.

    The MODIS sinusoidal tile grid divides the globe into 36 × 18 tiles.
    Each tile spans exactly 10° of latitude and ~10° equivalent in the
    sinusoidal projection.  Tile (h=0, v=0) is the top-left (north-west) tile.
    """
    lat_max = 90.0 - v * 10.0
    lat_min = lat_max - 10.0
    # Longitude extent depends on latitude; use the wider of top/bottom edge
    # sin-projection x-extent per tile is (2*pi*R/36) metres, but for bbox
    # purposes we just use the geographic span at each latitude edge.
    import math
    def _lon_half_width(lat_deg):
        lat_r = math.radians(abs(lat_deg))
        cos_lat = math.cos(lat_r) if lat_r < math.pi / 2 else 1e-9
        return 10.0 / cos_lat  # degrees longitude for 10° equivalent arc
    hw = max(_lon_half_width(lat_min), _lon_half_width(lat_max))
    lon_centre = -180.0 + (h + 0.5) * (360.0 / 36.0)
    return lon_centre - hw, lat_min, lon_centre + hw, lat_max


def _viirs_tile_overlaps_area(h: int, v: int, area) -> bool:
    """Return True if the VIIRS tile (h, v) may overlap *area*'s lat/lon bbox."""
    lon_min, lat_min, lon_max, lat_max = _viirs_tile_bbox(h, v)
    # Get the area's geographic bounding box from its 4 corner lon/lats
    try:
        import math as _math
        corners = area.outer_boundary_corners  # list of Coordinate objects (lon/lat in radians)
        # Coordinate objects have .lon and .lat attributes in radians
        corner_lons = [_math.degrees(c.lon) for c in corners]
        corner_lats = [_math.degrees(c.lat) for c in corners]
        a_lon_min, a_lon_max = min(corner_lons), max(corner_lons)
        a_lat_min, a_lat_max = min(corner_lats), max(corner_lats)
    except Exception:
        return True  # Can't determine — include tile to be safe
    return not (lon_max < a_lon_min or lon_min > a_lon_max or
                lat_max < a_lat_min or lat_min > a_lat_max)


def _regrid_viirs(
    input_dir: str,
    lis_area,
    method: str,
    fs=None,
    max_tiles: int | None = None,
) -> dict[str, xr.DataArray]:
    """Regrid the first VIIRS date group found and return {var_name: DataArray}."""
    from pyresample.geometry import SwathDefinition

    files = _list_files(input_dir, ".h5", fs=fs)
    if not files:
        log.warning("No VIIRS files found under %s — skipping", input_dir)
        return {}

    date_groups: dict[str, list[str]] = defaultdict(list)
    for p in files:
        stem = _path_name(p).rsplit(".", 1)[0] if "." in _path_name(p) else _path_name(p)
        parts = stem.split(".")
        date_key = parts[1] if len(parts) > 1 else stem
        date_groups[date_key].append(p)

    date_key, paths = next(iter(sorted(date_groups.items())))

    # Filter to tiles that spatially overlap the LIS domain
    filtered = []
    skipped = 0
    for p in paths:
        fname = _path_name(p)
        m = __import__("re").search(r"\.h(\d{2})v(\d{2})\.", fname)
        if m is not None:
            h, v = int(m.group(1)), int(m.group(2))
            if not _viirs_tile_overlaps_area(h, v, lis_area):
                skipped += 1
                continue
        filtered.append(p)
    if skipped:
        log.info("VIIRS: skipped %d tile(s) outside LIS domain bbox", skipped)
    paths = filtered
    if not paths:
        log.warning("VIIRS: no tiles overlap the LIS domain — skipping")
        return {}
    if max_tiles is not None and len(paths) > max_tiles:
        raise RuntimeError(
            f"VIIRS: {len(paths)} tile(s) would be loaded but --max-viirs-tiles={max_tiles}. "
            "Aborting to prevent OOM. Either the spatial filter is not working correctly "
            "or the domain is unusually large. Increase --max-viirs-tiles only if expected."
        )
    log.info("VIIRS: using date %s (%d tile(s))", date_key, len(paths))

    all_data, all_lons, all_lats = [], [], []
    for path in paths:
        if _is_s3(path):
            handler = handler_from_s3(ViirsFileHandler, path, fs=fs)
        else:
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

    log.info("VIIRS: regridding with method=%s …", method)
    rg = regrid(composite_da, source_def, lis_area, method=method)

    return {
        "viirs_cgf_ndsi_snow_cover": xr.DataArray(
            rg.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "VIIRS CGF NDSI snow cover",
                "units": "1",
                "source": f"date_key={date_key}",
            },
        )
    }


def _regrid_icesat2(
    parquet_path: str,
    lis_area,
    lis_grid: xr.Dataset,
    fs=None,
) -> dict[str, xr.DataArray]:
    """Regrid ICESat-2 ATL06 point cloud and return {var_name: DataArray}.

    For S3 URIs, the path is passed directly to geopandas.read_parquet which
    delegates to pyarrow's native S3 support (no FSFile wrapper needed).
    For local paths, existence is checked before proceeding.
    """
    if not _is_s3(parquet_path) and not Path(parquet_path).exists():
        log.warning("ICESat-2 Parquet not found at %s — skipping", parquet_path)
        return {}
    # Icesat2FileHandler.get_dataset calls geopandas.read_parquet(str(self.filename))
    # which supports s3:// URIs natively via pyarrow, so from_path without fs works.
    handler = Icesat2FileHandler.from_path(parquet_path)

    log.info("ICESat-2: loading from %s", parquet_path)
    da = handler.get_dataset()
    source_area = da.attrs["area"]
    log.info("ICESat-2: regridding %d observations using mean …", len(da))
    rg = regrid(da, source_area, lis_area, method="mean")
    return {
        "icesat2_h_li": xr.DataArray(
            rg.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={
                "long_name": "ICESat-2 ATL06 land-ice surface height (mean per LIS pixel)",
                "units": "meters",
                "source": _path_name(parquet_path),
            },
        )
    }


def _write_output(ds_out: xr.Dataset, output_path: str, encoding: dict, fs=None) -> None:
    """Write *ds_out* to *output_path*, which may be a local path or S3 URI."""
    if _is_s3(output_path):
        if fs is None:
            fs = make_fs()
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            ds_out.to_netcdf(tmp_path, engine="h5netcdf", encoding=encoding)
            with open(tmp_path, "rb") as f:
                data = f.read()
            with fs.open(output_path, "wb") as fout:
                fout.write(data)
        finally:
            os.unlink(tmp_path)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(output_path, engine="h5netcdf", encoding=encoding)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regrid all SWE input datasets to the LIS grid and combine into "
            "a single NetCDF file.  All paths may be local or s3:// URIs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lis-path", required=True,
                        help="Path to the LIS input NetCDF file (local or s3://).")
    parser.add_argument("--amsr2-dir", default=None,
                        help="Directory containing AMSR2 HDF5 files (local or s3://).")
    parser.add_argument("--ceda-dir", default=None,
                        help="Directory containing CEDA ESA CCI SWE NetCDF files (local or s3://).")
    parser.add_argument("--viirs-dir", default=None,
                        help="Directory containing VIIRS CGF snow cover HDF5 files (local or s3://).")
    parser.add_argument("--icesat2-parquet", default=None,
                        help="Path to the cached ICESat-2 ATL06 Parquet file (local or s3://).")
    parser.add_argument("--weights-dir", default="_data/weights",
                        help="Local directory for xESMF weights files.")
    parser.add_argument("--output-path", default="_data/swe_combined.nc",
                        help="Output combined NetCDF path (local or s3://).")
    # Per-dataset method overrides
    parser.add_argument("--amsr2-method", default="bilinear",
                        choices=["bilinear", "nearest_s2d", "conservative"],
                        help="xESMF regridding method for AMSR2.")
    parser.add_argument("--ceda-method", default="bilinear",
                        choices=["bilinear", "nearest_s2d", "conservative"],
                        help="xESMF regridding method for CEDA.")
    parser.add_argument("--viirs-method", default="nearest",
                        choices=["nearest", "bilinear", "ewa", "bucket_avg"],
                        help="pyresample regridding method for VIIRS.")
    parser.add_argument("--overwrite-weights", action="store_true", default=False,
                        help="Always recompute xESMF weights even if cached files exist.")
    parser.add_argument("--max-viirs-tiles", type=int, default=None, metavar="N",
                        help="Abort before loading if the filtered VIIRS tile count exceeds N.")
    ns = parser.parse_args()

    # Build a shared fsspec store if any S3 paths are present
    any_s3 = any(
        _is_s3(str(p))
        for p in [ns.lis_path, ns.amsr2_dir, ns.ceda_dir, ns.viirs_dir, ns.icesat2_parquet]
        if p is not None
    )
    fs = make_fs() if any_s3 else None

    lis_grid = load_lis_grid(ns.lis_path, fs=fs)
    lis_area = build_lis_area_definition(ns.lis_path, fs=fs, cache_dir=ns.weights_dir, overwrite=ns.overwrite_weights)

    data_vars: dict[str, xr.DataArray] = {}

    if ns.amsr2_dir is not None:
        data_vars.update(_regrid_amsr2(ns.amsr2_dir, lis_grid, ns.amsr2_method, ns.weights_dir, overwrite_weights=ns.overwrite_weights, fs=fs))

    if ns.ceda_dir is not None:
        data_vars.update(_regrid_ceda(ns.ceda_dir, lis_grid, ns.ceda_method, ns.weights_dir, overwrite_weights=ns.overwrite_weights, fs=fs))

    if ns.viirs_dir is not None:
        data_vars.update(_regrid_viirs(ns.viirs_dir, lis_area, ns.viirs_method, fs=fs, max_tiles=ns.max_viirs_tiles))

    if ns.icesat2_parquet is not None:
        data_vars.update(_regrid_icesat2(ns.icesat2_parquet, lis_area, lis_grid, fs=fs))

    if not data_vars:
        log.error(
            "No input directories provided or no data found. "
            "Pass at least one of --amsr2-dir, --ceda-dir, --viirs-dir, --icesat2-parquet."
        )
        raise SystemExit(1)

    # Build combined Dataset with shared LIS lat/lon coordinates
    ds_out = xr.Dataset(
        data_vars,
        coords={
            "lat": lis_grid["lat"],
            "lon": lis_grid["lon"],
        },
        attrs={
            "description": (
                "Combined SWE and snow-cover observations regridded to the "
                "LIS 1 km Lambert Conformal grid (Missouri/NMP domain)."
            ),
            "conventions": "CF-1.8",
        },
    )

    encoding = {
        var: {"dtype": "float32", "_FillValue": np.float32("nan")}
        for var in data_vars
    }

    log.info("Writing combined output to %s …", ns.output_path)
    _write_output(ds_out, ns.output_path, encoding, fs=fs)
    log.info("Done: %s", ns.output_path)

    log.info("Variables written:")
    for var in data_vars:
        shape = data_vars[var].shape
        log.info("  %-45s %s", var, shape)


if __name__ == "__main__":
    main()
