#!/usr/bin/env python
"""AWS Batch entrypoint: generate ESMF regridding weights.

Workflow
--------
1. Optionally check S3 for a cached weights file and exit early (cache hit).
2. Build or download the source SCRIP grid.
3. Build or download the destination SCRIP grid.
4. Run ESMF_RegridWeightGen.
5. Upload the output weights file to S3.

The source grid geometry is provided by the source module (``sources/<source>.py``
from the icechunk package), which exports ``SRC_GRID_SPEC``.  This can be
overridden at runtime via ``SRC_GRID_URI`` or ``SRC_GRID_SPEC`` env vars.

Configuration — environment variables
--------------------------------------
Required:
  SOURCE            Data source identifier (e.g. ``amsr2``).
                    Used to look up the default source grid spec from
                    the icechunk sources package (``sources/<source>.py``).
  WEIGHTS_URI       s3:// URI where the output weights file will be uploaded.

  DST_GRID          Destination grid — one of three forms (detected automatically):
                      s3://bucket/path/to/lis_input.nc   — LIS domain NetCDF on S3
                      s3://bucket/path/to/grid.nc        — pre-built SCRIP file on S3
                      '{"nlat":360,"nlon":720,...}'      — JSON bounding box for a
                                                           regular equirectangular grid.

Optional:
  SRC_GRID_URI      s3:// URI of a pre-built source SCRIP grid.
                    Overrides the source module SRC_GRID_SPEC.
  SRC_GRID_SPEC     JSON object describing a regular lat/lon source grid.
                    Overrides the source module SRC_GRID_SPEC.
                    Ignored when SRC_GRID_URI is set.
  METHOD            Regridding method passed to ESMF_RegridWeightGen.
                    One of: bilinear (default), conserve, conserve2nd,
                    patch, nearest_stod, nearest_dtos.
  EXTRA_ARGS        Space-separated extra flags for ESMF_RegridWeightGen
                    (default: "--no_log --ignore_unmapped").
  FORCE_REGENERATE  Set to "true" / "1" / "yes" to skip the S3 cache check
                    and always run ESMF_RegridWeightGen.  Default: false.
  AWS_REGION        AWS region (default: us-west-2).
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import boto3
import numpy as np
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── source module loader ───────────────────────────────────────────────────────

def _load_source_module(source: str):
    """Import sources/<source>.py — shared with the icechunk populate job.

    sources/ is copied into /app/sources/ in both the join-esmf-regrid and
    join-icechunk images.  Since both scripts run from /app, sources/ is a
    natural sibling and importable without any sys.path manipulation.
    """
    try:
        return importlib.import_module(f"sources.{source}")
    except ModuleNotFoundError:
        log.error(
            "Unknown source %r — no module sources/%s.py found.",
            source, source,
        )
        sys.exit(1)


# ── S3 utilities ───────────────────────────────────────────────────────────────

def _s3_client():
    """Return a boto3 S3 client using the ECS task IAM role (no keys needed)."""
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-2"))


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    without_scheme = uri[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")


def s3_download(uri: str, dest: Path) -> None:
    """Download an S3 object to *dest*."""
    bucket, key = _parse_s3_uri(uri)
    log.info("S3 download  s3://%s/%s  →  %s", bucket, key, dest)
    try:
        _s3_client().download_file(bucket, key, str(dest))
    except ClientError as exc:
        log.error("S3 download failed: %s", exc)
        sys.exit(1)
    log.info("Downloaded %s (%d bytes)", dest.name, dest.stat().st_size)


def s3_upload(src: Path, uri: str) -> None:
    """Upload *src* to an S3 URI."""
    bucket, key = _parse_s3_uri(uri)
    log.info("S3 upload    %s  →  s3://%s/%s", src, bucket, key)
    try:
        _s3_client().upload_file(str(src), bucket, key)
    except ClientError as exc:
        log.error("S3 upload failed: %s", exc)
        sys.exit(1)
    log.info("Uploaded %s (%d bytes)", src.name, src.stat().st_size)


def s3_object_exists(uri: str) -> bool:
    """Return True if the S3 object exists."""
    bucket, key = _parse_s3_uri(uri)
    try:
        _s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


# ── SCRIP grid writing ─────────────────────────────────────────────────────────

def write_scrip_grid(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    path: Path,
    title: str = "grid",
    mask2d: np.ndarray | None = None,
) -> None:
    """Write a 2-D curvilinear grid in SCRIP NetCDF format.

    Parameters
    ----------
    lat2d, lon2d:
        2-D arrays of cell-centre coordinates, shape (ny, nx).
    path:
        Output NetCDF file path.
    title:
        Value for the global ``title`` attribute.
    mask2d:
        Optional 2-D integer mask (1 = active, 0 = masked), shape (ny, nx).
        If None, all cells are marked active.
    """
    import netCDF4 as nc  # noqa: N813

    ny, nx = lat2d.shape
    ncells = ny * nx
    lat_flat = lat2d.flatten()
    lon_flat = lon2d.flatten()

    dlat = np.gradient(lat2d, axis=0)
    dlon = np.gradient(lon2d, axis=1)

    # Corner order: SW, SE, NE, NW
    lat_corners = np.stack([
        lat_flat - dlat.flatten() / 2,
        lat_flat - dlat.flatten() / 2,
        lat_flat + dlat.flatten() / 2,
        lat_flat + dlat.flatten() / 2,
    ], axis=0)
    lon_corners = np.stack([
        lon_flat - dlon.flatten() / 2,
        lon_flat + dlon.flatten() / 2,
        lon_flat + dlon.flatten() / 2,
        lon_flat - dlon.flatten() / 2,
    ], axis=0)

    with nc.Dataset(path, "w") as ds:
        ds.title = title
        ds.createDimension("grid_size", ncells)
        ds.createDimension("grid_corners", 4)
        ds.createDimension("grid_rank", 2)

        gd = ds.createVariable("grid_dims", "i4", ("grid_rank",))
        gd[:] = [nx, ny]

        gcl = ds.createVariable("grid_center_lat", "f8", ("grid_size",), fill_value=False)
        gcl.units = "degrees"
        gcl[:] = lat_flat

        gco = ds.createVariable("grid_center_lon", "f8", ("grid_size",), fill_value=False)
        gco.units = "degrees"
        gco[:] = lon_flat

        gcrl = ds.createVariable("grid_corner_lat", "f8", ("grid_size", "grid_corners"), fill_value=False)
        gcrl.units = "degrees"
        gcrl[:] = lat_corners.T

        gcro = ds.createVariable("grid_corner_lon", "f8", ("grid_size", "grid_corners"), fill_value=False)
        gcro.units = "degrees"
        gcro[:] = lon_corners.T

        gim = ds.createVariable("grid_imask", "i4", ("grid_size",))
        gim.units = "unitless"
        gim[:] = (
            mask2d.flatten().astype("i4")
            if mask2d is not None
            else np.ones(ncells, dtype="i4")
        )

    log.info("Wrote SCRIP grid '%s' to %s (%d cells)", title, path, ncells)


def build_equirect_scrip(path: Path, spec: dict) -> None:
    """Write a regular lat/lon SCRIP grid from a spec dict.

    Parameters
    ----------
    path:
        Output SCRIP NetCDF path.
    spec:
        Dict with keys: nlat, nlon, lat_min, lat_max, lon_min, lon_max, title.
        All keys are required — no silent defaults.
    """
    nlat    = int(spec["nlat"])
    nlon    = int(spec["nlon"])
    lat_max = float(spec["lat_max"])
    lat_min = float(spec["lat_min"])
    lon_min = float(spec["lon_min"])
    lon_max = float(spec["lon_max"])
    title   = str(spec.get("title", f"equirectangular {nlat}x{nlon}"))

    lats = np.linspace(lat_max, lat_min, nlat)
    lons = np.linspace(lon_min, lon_max, nlon)
    lon2d, lat2d = np.meshgrid(lons, lats)
    write_scrip_grid(lat2d, lon2d, path, title=title)
    log.info("Generated custom equirectangular SCRIP grid: %s (%dx%d)", title, nlat, nlon)


def build_lis_scrip(lis_path: str, path: Path, tmp_dir: Path) -> None:
    """Download the LIS NetCDF from S3 and write a destination SCRIP grid.

    Parameters
    ----------
    lis_path:
        ``s3://`` URI or local path to the LIS input NetCDF file.
    path:
        Output path for the destination SCRIP NetCDF.
    tmp_dir:
        Scratch directory (used if lis_path is on S3).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from lis_grid import load_lis_grid

    if _is_s3(lis_path):
        local_lis = tmp_dir / "lis_input.nc"
        s3_download(lis_path, local_lis)
        lis_ds = load_lis_grid(local_lis)
    else:
        lis_ds = load_lis_grid(lis_path)

    dst_lat2d = lis_ds["lat"].values  # (ny, nx), north-first
    dst_lon2d = lis_ds["lon"].values
    write_scrip_grid(dst_lat2d, dst_lon2d, path, title="LIS 1km LCC")


# ── main ───────────────────────────────────────────────────────────────────────

def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        log.error("Required environment variable %s is not set.", name)
        sys.exit(1)
    return val


def _parse_dst_grid(value: str) -> tuple[str, dict | None]:
    """Parse DST_GRID into (kind, parsed) where kind is 'lis', 'scrip', or 'spec'.

    Detection logic:
      - Starts with '{' → JSON bounding box spec
      - Ends with .nc / .nc4 / .netcdf → S3 URI; distinguish LIS vs SCRIP by
        checking for the MAP_PROJECTION global attribute (present in LIS files).
        If attribute is absent, assume pre-built SCRIP.
    """
    value = value.strip()
    if value.startswith("{"):
        try:
            spec = json.loads(value)
            if not isinstance(spec, dict):
                raise ValueError("DST_GRID JSON must be an object")
            return "spec", spec
        except (json.JSONDecodeError, ValueError) as exc:
            log.error("Invalid DST_GRID JSON: %s", exc)
            sys.exit(1)

    if not _is_s3(value):
        log.error("DST_GRID must be an s3:// URI or a JSON object, got: %r", value)
        sys.exit(1)

    # Peek at the NetCDF global attributes to tell LIS from SCRIP.
    try:
        import netCDF4 as nc4  # noqa: N813
        import tempfile as _tmp
        local = Path(_tmp.mktemp(suffix=".nc"))
        s3_download(value, local)
        with nc4.Dataset(local) as ds:
            is_lis = hasattr(ds, "MAP_PROJECTION")
        local.unlink(missing_ok=True)
        return ("lis" if is_lis else "scrip"), None
    except Exception as exc:
        log.error("Could not inspect DST_GRID file %s: %s", value, exc)
        sys.exit(1)


def main() -> None:
    source      = _require_env("SOURCE")
    weights_uri = _require_env("WEIGHTS_URI")
    dst_grid    = _require_env("DST_GRID")

    src_grid_uri   = os.environ.get("SRC_GRID_URI", "").strip()
    src_grid_spec  = os.environ.get("SRC_GRID_SPEC", "").strip()
    method         = os.environ.get("METHOD", "bilinear")
    extra_args_str = os.environ.get("EXTRA_ARGS", "--no_log --ignore_unmapped")
    extra_args     = extra_args_str.split() if extra_args_str.strip() else []
    force_regen    = os.environ.get("FORCE_REGENERATE", "").lower() in ("1", "true", "yes")

    log.info("SOURCE           : %s", source)
    log.info("DST_GRID         : %s", dst_grid)
    log.info("SRC_GRID_URI     : %s", src_grid_uri or "(not set)")
    log.info("SRC_GRID_SPEC    : %s", src_grid_spec or "(not set — using source module default)")
    log.info("WEIGHTS_URI      : %s", weights_uri)
    log.info("METHOD           : %s", method)
    log.info("EXTRA_ARGS       : %s", extra_args)
    log.info("FORCE_REGENERATE : %s", force_regen)

    # ── load source module ────────────────────────────────────────────────────
    src_mod = _load_source_module(source)
    if not hasattr(src_mod, "SRC_GRID_SPEC"):
        log.error("sources/%s.py does not export SRC_GRID_SPEC.", source)
        sys.exit(1)

    # ── S3 cache check ────────────────────────────────────────────────────────
    if not force_regen and _is_s3(weights_uri) and s3_object_exists(weights_uri):
        log.info(
            "Weights already exist at %s — skipping generation. "
            "Set FORCE_REGENERATE=true to override.",
            weights_uri,
        )
        return

    # ── parse DST_GRID ────────────────────────────────────────────────────────
    dst_kind, dst_spec = _parse_dst_grid(dst_grid)
    log.info("DST_GRID kind: %s", dst_kind)

    # ── resolve source SCRIP grid spec ────────────────────────────────────────
    # Priority: SRC_GRID_URI > SRC_GRID_SPEC env var > source module SRC_GRID_SPEC
    parsed_src_spec: dict | None = None
    if not src_grid_uri:
        if src_grid_spec:
            try:
                parsed_src_spec = json.loads(src_grid_spec)
                if not isinstance(parsed_src_spec, dict):
                    raise ValueError("SRC_GRID_SPEC must be a JSON object")
                log.info("Using SRC_GRID_SPEC from env var.")
            except (json.JSONDecodeError, ValueError) as exc:
                log.error("Invalid SRC_GRID_SPEC: %s", exc)
                sys.exit(1)
        else:
            parsed_src_spec = src_mod.SRC_GRID_SPEC
            log.info("Using SRC_GRID_SPEC from sources/%s.py: %s", source, parsed_src_spec)

    with tempfile.TemporaryDirectory(prefix="esmf_regrid_") as tmp_str:
        tmp = Path(tmp_str)

        # ── 1. Source SCRIP grid ──────────────────────────────────────────────
        src_grid_path = tmp / "src_grid.nc"
        if src_grid_uri:
            s3_download(src_grid_uri, src_grid_path)
        else:
            log.info("Generating source SCRIP grid …")
            build_equirect_scrip(src_grid_path, parsed_src_spec)

        # ── 2. Destination SCRIP grid ─────────────────────────────────────────
        dst_grid_path = tmp / "dst_grid.nc"
        if dst_kind == "scrip":
            s3_download(dst_grid, dst_grid_path)
        elif dst_kind == "spec":
            log.info("Generating destination SCRIP grid from DST_GRID JSON spec …")
            build_equirect_scrip(dst_grid_path, dst_spec)
        else:  # lis
            log.info("Building LIS destination SCRIP grid from %s …", dst_grid)
            build_lis_scrip(dst_grid, dst_grid_path, tmp)

        # ── 3. Run ESMF_RegridWeightGen ───────────────────────────────────────
        weights_local = tmp / "weights.nc"
        cmd = [
            "ESMF_RegridWeightGen",
            "--source",      str(src_grid_path),
            "--destination", str(dst_grid_path),
            "--weight",      str(weights_local),
            "--method",      method,
            "--src_type",    "SCRIP",
            "--dst_type",    "SCRIP",
        ] + extra_args

        log.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp))

        if result.stdout.strip():
            log.info("ESMF stdout:\n%s", result.stdout)
        if result.stderr.strip():
            log.warning("ESMF stderr:\n%s", result.stderr)

        if result.returncode != 0:
            log.error("ESMF_RegridWeightGen failed (exit %d)", result.returncode)
            sys.exit(result.returncode)

        log.info(
            "ESMF_RegridWeightGen complete. Weights: %s (%d bytes)",
            weights_local.name,
            weights_local.stat().st_size,
        )

        # ── 4. Upload weights to S3 ───────────────────────────────────────────
        if _is_s3(weights_uri):
            s3_upload(weights_local, weights_uri)
        else:
            import shutil
            out = Path(weights_uri)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(weights_local, out)
            log.info("Weights saved to %s", out)

    log.info("Done.")


if __name__ == "__main__":
    main()
