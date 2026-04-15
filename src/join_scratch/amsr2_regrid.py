#!/usr/bin/env python
"""Regrid local AMSR2 snow depth files to the LIS input grid using xESMF."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr
import xesmf
from pyresample.bilinear import XArrayBilinearResampler
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample import kd_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "_data-raw"
DATA_OUT = ROOT / "_data" / "amsr2"
LIS_PATH = DATA_RAW / "lis_input_NMP_1000m_missouri.nc"
AMSR2_GLOB = "JOIN/AMSR2/**/*.h5"
WEIGHTS_PATH = DATA_OUT / "amsr2-lis-weights.nc"


# ---------------------------------------------------------------------------
# AMSR2 constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Amsr2Constants:
    """Constants describing the AMSR2 equirectangular grid and regrid parameters.

    The AMSR2 L3 monthly product uses a fixed global 0.1° equirectangular grid
    with no explicit coordinate variables in the HDF5 file; all values here are
    derived from the product specification.

    Pyresample radius of influence is set slightly larger than the AMSR2 pixel
    diagonal (~11 km at mid-latitudes) to avoid gaps in the target grid.
    """

    n_lat: int = 1800
    n_lon: int = 3600
    lat: np.ndarray = field(default_factory=lambda: np.linspace(89.95, -89.95, 1800))
    lon: np.ndarray = field(default_factory=lambda: np.linspace(-179.95, 179.95, 3600))
    kd_radius_m: float = 15_000.0
    gauss_sigma_m: float = 10_000.0


AMSR2 = Amsr2Constants()


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------


def load_lis_grid(path: Path) -> xr.Dataset:
    """Load the LIS input file and return a dataset with lat/lon/lat_b/lon_b."""
    log.info("Loading LIS grid from %s", path)
    ds = xr.open_dataset(path, engine="h5netcdf")
    return ds[["lat", "lon", "lat_b", "lon_b"]]


def build_lis_area_definition(path: Path) -> AreaDefinition:
    """Construct a pyresample AreaDefinition for the LIS Lambert Conformal grid.

    All parameters are read from the global attributes of the LIS input file:
      - MAP_PROJECTION: LAMBERT CONFORMAL
      - SOUTH_WEST_CORNER_LAT/LON (pixel centres)
      - TRUELAT1/TRUELAT2, STANDARD_LON
      - DX/DY (km), grid shape from the lat/lon variable dimensions

    AreaDefinition stores rows top-to-bottom (row 0 = northernmost); the
    area_extent (x_ll, y_ll, x_ur, y_ur) reconciles this with LIS's
    south-first row ordering automatically.
    """
    ds = xr.open_dataset(path, engine="h5netcdf")
    attrs = ds.attrs

    sw_lat = float(attrs["SOUTH_WEST_CORNER_LAT"])
    sw_lon = float(attrs["SOUTH_WEST_CORNER_LON"])
    dx_m = float(attrs["DX"]) * 1000.0
    dy_m = float(attrs["DY"]) * 1000.0
    truelat1 = float(attrs["TRUELAT1"])
    truelat2 = float(attrs["TRUELAT2"])
    standard_lon = float(attrs["STANDARD_LON"])
    ny, nx = ds["lat"].shape

    crs = pyproj.CRS.from_dict(
        {
            "proj": "lcc",
            "lat_1": truelat1,
            "lat_2": truelat2,
            "lon_0": standard_lon,
            "lat_0": (truelat1 + truelat2) / 2.0,
            "datum": "WGS84",
            "units": "m",
        }
    )

    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_sw, y_sw = transformer.transform(sw_lon, sw_lat)

    x_min = x_sw - dx_m / 2
    y_min = y_sw - dy_m / 2
    x_max = x_min + nx * dx_m
    y_max = y_min + ny * dy_m

    return AreaDefinition(
        "lis_lcc",
        "LIS Lambert Conformal 1 km",
        "lis_lcc",
        crs.to_dict(),
        nx,
        ny,
        (x_min, y_min, x_max, y_max),
    )


def build_amsr2_swath_definition(ds: xr.Dataset) -> SwathDefinition:
    """Build a pyresample SwathDefinition from an AMSR2 xarray Dataset.

    The dataset must already have 1-D 'lat' and 'lon' coordinates assigned;
    they are broadcast to 2-D meshgrids as required by pyresample.
    """
    lons_2d, lats_2d = np.meshgrid(ds["lon"].values, ds["lat"].values)
    return SwathDefinition(lons=lons_2d, lats=lats_2d)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_amsr2(path: Path) -> xr.Dataset:
    """Load an AMSR2 HDF5 file, assign coordinates, and mask invalid values.

    Returns the full global dataset ready for regridding.
    """
    log.info("Loading AMSR2 file %s", path)
    ds = (
        xr.open_dataset(path, engine="h5netcdf", phony_dims="sort")
        .rename_dims({"phony_dim_0": "lat", "phony_dim_1": "lon"})
        .assign_coords(lat=AMSR2.lat, lon=AMSR2.lon)
        .sortby(["lat", "lon"])
    )

    # Mask fill values (valid data is >= 0)
    ds = ds.where(ds["Geophysical Data"] >= 0)
    return ds


# ---------------------------------------------------------------------------
# xESMF regridding
# ---------------------------------------------------------------------------


def _weights_are_valid(weights_path: Path, n_in: int, n_out: int) -> bool:
    """Return True if the cached weight file matches the expected grid sizes."""
    try:
        w = xr.open_dataset(weights_path)
        max_col = int(w["col"].max())
        max_row = int(w["row"].max())
        return max_col <= n_in and max_row <= n_out
    except Exception:
        return False


def get_regridder(
    source_grid: xr.Dataset,
    target_grid: xr.Dataset,
    weights_path: Path,
) -> xesmf.Regridder:
    """Build or load an xESMF bilinear regridder.

    If *weights_path* already exists and is consistent with the source/target
    grid sizes, the weights are reused; otherwise they are recomputed and saved.
    """
    if weights_path.exists():
        # Infer expected sizes from the grids
        from xesmf.frontend import _get_lon_lat

        lon_in, lat_in = _get_lon_lat(source_grid)
        lon_out, lat_out = _get_lon_lat(target_grid)
        n_in = int(np.asarray(lat_in).size)
        n_out = int(np.asarray(lat_out).size)
        if _weights_are_valid(weights_path, n_in, n_out):
            log.info("Reusing existing xESMF weights from %s", weights_path)
            return xesmf.Regridder(
                source_grid,
                target_grid,
                method="bilinear",
                periodic=True,
                weights=str(weights_path),
                reuse_weights=True,
            )
        log.warning(
            "Cached weights at %s are incompatible with current grids "
            "(n_in=%d, n_out=%d); recomputing.",
            weights_path,
            n_in,
            n_out,
        )
        weights_path.unlink()

    log.info("Computing xESMF bilinear weights …")
    regridder = xesmf.Regridder(
        source_grid,
        target_grid,
        method="bilinear",
        periodic=True,
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    regridder.to_netcdf(str(weights_path))
    log.info("xESMF weights saved to %s", weights_path)
    return regridder


# ---------------------------------------------------------------------------
# pyresample regridding
# ---------------------------------------------------------------------------


def regrid_kd_nearest(
    ds: xr.Dataset,
    source_def: SwathDefinition,
    target_def: AreaDefinition,
) -> np.ndarray:
    """Regrid using pyresample kd_tree nearest-neighbour resampling.

    Returns a float32 array of shape (NY, NX, n_inner).
    """
    data = ds["Geophysical Data"].values  # (nlat, nlon, n_inner)
    slices = [
        kd_tree.resample_nearest(
            source_def,
            data[:, :, i],
            target_def,
            radius_of_influence=AMSR2.kd_radius_m,
            fill_value=np.nan,
        )
        for i in range(data.shape[2])
    ]
    return np.stack(slices, axis=-1)


def regrid_kd_gauss(
    ds: xr.Dataset,
    source_def: SwathDefinition,
    target_def: AreaDefinition,
) -> np.ndarray:
    """Regrid using pyresample kd_tree Gaussian-weighted resampling.

    Returns a float32 array of shape (NY, NX, n_inner).
    """
    data = ds["Geophysical Data"].values
    slices = [
        kd_tree.resample_gauss(
            source_def,
            data[:, :, i],
            target_def,
            radius_of_influence=AMSR2.kd_radius_m,
            sigmas=AMSR2.gauss_sigma_m,
            fill_value=np.nan,
        )
        for i in range(data.shape[2])
    ]
    return np.stack(slices, axis=-1)


def regrid_bilinear_pyresample(
    ds: xr.Dataset,
    source_def: SwathDefinition,
    target_def: AreaDefinition,
) -> np.ndarray:
    """Regrid using pyresample XArrayBilinearResampler.

    Returns a float32 array of shape (NY, NX, n_inner).
    """
    data = ds["Geophysical Data"].values
    slices = []
    for i in range(data.shape[2]):
        da = xr.DataArray(data[:, :, i].astype(np.float32), dims=["y", "x"])
        resampler = XArrayBilinearResampler(
            source_def,
            target_def,
            radius_of_influence=AMSR2.kd_radius_m,
        )
        slices.append(resampler.resample(da).values)
    return np.stack(slices, axis=-1)


# ---------------------------------------------------------------------------
# File-level regridding (xESMF path, writes NetCDF)
# ---------------------------------------------------------------------------


def regrid_file(
    amsr2_path: Path,
    regridder: xesmf.Regridder,
    out_dir: Path,
) -> Path:
    """Regrid a single AMSR2 file with xESMF and write the result to *out_dir*.

    Returns the path of the written output file.
    """
    ds = load_amsr2(amsr2_path)

    log.info("Regridding %s …", amsr2_path.name)
    regridded = regridder(ds)

    out_path = out_dir / (amsr2_path.stem + ".nc")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Writing output to %s", out_path)
    regridded.to_netcdf(out_path, engine="h5netcdf")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    amsr2_files = sorted(DATA_RAW.glob(AMSR2_GLOB))
    if not amsr2_files:
        raise FileNotFoundError(
            f"No AMSR2 files found matching '{AMSR2_GLOB}' under {DATA_RAW}"
        )
    log.info("Found %d AMSR2 file(s)", len(amsr2_files))

    lis_grid = load_lis_grid(LIS_PATH)

    # Build/load the regridder using the first file's grid (all files share
    # the same equirectangular 0.1° grid, so one weight set covers all).
    source_ds = load_amsr2(amsr2_files[0])
    source_grid = source_ds[["lat", "lon"]]
    regridder = get_regridder(source_grid, lis_grid, WEIGHTS_PATH)

    for amsr2_path in amsr2_files:
        out_path = regrid_file(amsr2_path, regridder, DATA_OUT)
        log.info("Done: %s", out_path)

    log.info("All files regridded successfully.")


if __name__ == "__main__":
    main()
