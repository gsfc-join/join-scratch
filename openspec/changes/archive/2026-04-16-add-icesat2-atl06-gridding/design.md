## Context

The JOIN project grids satellite snow observations to the LIS model grid. Three datasets currently follow the same pattern: read from storage (local or S3), regrid to the LIS Lambert Conformal grid, write NetCDF output. Each dataset lives in `src/join_scratch/<dataset>/` and uses `StorageConfig` for I/O abstraction.

ICESat-2 ATL06 differs from the other datasets in a key way: data is not pulled from a file-based archive but retrieved on demand from the SlideRule Earth cloud processing service. The retrieval is bounded by spatial AOI and temporal window rather than file paths.

## Goals / Non-Goals

**Goals:**
- Add `src/join_scratch/icesat2/` module following the same structural conventions as `amsr2`, `viirs`, and `ceda`
- Retrieve ICESat-2 ATL06 snow height data for the LIS domain bounding box and the first week of January 2019 via SlideRule
- Cache the raw SlideRule response locally as a Parquet file; add a `--force-download` (or `--no-cache`) argparse flag to overwrite
- Grid the point data to the LIS pixel grid: mean of all ATL06 `h_mean` values within each pixel; NaN if no observations
- Output a NetCDF file with the same grid dimensions and coordinate variables as the other regridded outputs
- Support `--storage` / `--storage-location` arguments identical to other modules (used to read the LIS input grid)

**Non-Goals:**
- Multi-temporal output (one output file for the week, not per-day files)
- Retrieving other ATL06 variables beyond `h_mean`
- Uncertainty propagation or quality filtering beyond SlideRule's defaults

## Decisions

### SlideRule for data retrieval
SlideRule's `icesat2.atl06p()` API handles spatial subsetting, ATL06 processing parameters, and returns a GeoDataFrame with point-level snow height data. This avoids managing raw `.h5` files and handles the complex ATL06 processing chain.

**Alternative**: Download raw ATL06 HDF5 files from NSIDC and process locally. Rejected — much more complex, requires `icepyx` or manual download scripts, and SlideRule already does this correctly in the cloud.

### Local Parquet cache
The SlideRule response (a GeoDataFrame) is serialized to Parquet using `geopandas.to_parquet()`. Parquet is compact, preserves geometry, and loads quickly with `geopandas.read_parquet()`. Cache path is `_data/icesat2/atl06_cache_<date_range>.parquet`. A `--force-download` flag bypasses cache.

**Alternative**: Cache as NetCDF or CSV. Parquet is preferred because it preserves the GeoDataFrame schema including geometry and dtypes without additional parsing.

### Gridding strategy: numpy bincount / scipy stats
Since ICESat-2 points are irregularly spaced, we map each point to its nearest LIS pixel using the LIS lat/lon grid, then compute the mean per pixel using `scipy.stats.binned_statistic_2d` or a manual `numpy` bincount approach. This avoids the overhead of pyresample/xESMF for a simple point-to-pixel mean.

The LIS grid is a Lambert Conformal projection. Points (lat/lon) must be projected to LIS CRS coordinates, then mapped to pixel indices using the pixel spacing and origin.

**Alternative**: Use pyresample `BucketAvg` (as used in AMSR2). This would work but requires constructing a SwathDefinition from irregular points, which adds complexity with no benefit for a simple mean.

### Output format
Output is a single NetCDF file `atl06_gridded_<date_range>.nc` with dimensions `(north_south, east_west)` matching the LIS grid, containing one variable `h_mean` (float32, fill_value=NaN). Coordinate variables `lat` and `lon` are copied from the LIS grid dataset.

## Risks / Trade-offs

- [Risk: SlideRule service availability] → Mitigated by local Parquet cache; once retrieved, no network dependency
- [Risk: Sparse output misleads users] → Document clearly in variable attributes that NaN = no observations
- [Risk: LIS grid projection math errors] → Validate by spot-checking that projected point indices fall within expected pixel bounds; compare a few points against known coordinates
- [Trade-off: Weekly mean vs. daily granularity] → Weekly aggregation was specified; finer temporal resolution is a future extension

## Implementation Notes (Amendments)

The following deviations from the original design were discovered during implementation:

- **SlideRule endpoint**: The design referenced `icesat2.atl06p` (legacy p-series). The implementation uses `sliderule.run("atl06x", parms)` — the recommended modern x-series endpoint. The `atl06x` endpoint returns standard ATL06 product columns directly, with the height variable named **`h_li`** (not `h_mean` as initially assumed). The output NetCDF variable was updated accordingly.
- **`build_lis_lcc_crs()` instead of `build_lis_area_definition()`**: The design proposed reusing or importing `build_lis_area_definition()` from `amsr2`. The implementation instead provides a standalone `build_lis_lcc_crs()` that returns a `(pyproj.CRS, grid_info dict)` tuple, avoiding a pyresample `AreaDefinition` dependency for a simple point-binning operation.
- **Gridding via `numpy` bincount**: Confirmed as designed — `np.add.at` used instead of pyresample/xESMF.
- **`_FillValue` in attrs**: xarray raised a `ValueError` when `_FillValue` appeared in both `attrs` and `encoding`. Removed from `attrs`; encoding dict alone controls the fill value.
- **`cartopy` added as dependency**: The visualization script (`atl06_visualize.py`) required `cartopy` for coastlines and borders. Added via `pixi add cartopy`.
- **Visualization script added**: `src/join_scratch/icesat2/atl06_visualize.py` and `viz_icesat2` pixi task were added beyond the original task scope.

## Migration Plan

No changes to existing modules or interfaces. This is a purely additive change. The new module can be run independently:

```
python -m join_scratch.icesat2.atl06_regrid --storage local
```

Add `sliderule` and `geopandas` to project dependencies in `pyproject.toml`.
