## 1. Module Scaffolding

- [x] 1.1 Create `src/join_scratch/icesat2/` directory with `__init__.py`
- [x] 1.2 Create `src/join_scratch/icesat2/atl06_regrid.py` with module skeleton (imports, constants, logging, argparse)

## 2. SlideRule Retrieval & Caching

- [x] 2.1 Implement `retrieve_atl06()` function that queries SlideRule `icesat2.atl06p` for the LIS domain bounding box and the first week of January 2019
- [x] 2.2 Implement `load_or_fetch_atl06()` function that loads from Parquet cache if it exists, otherwise calls `retrieve_atl06()` and saves to cache
- [x] 2.3 Add `--force-download` argparse flag to bypass and overwrite the cache

## 3. Gridding

- [x] 3.1 Implement `build_lis_area_definition()` (reuse or import from `amsr2`) to get LIS CRS and pixel grid parameters
- [x] 3.2 Implement `grid_atl06()` function: project ATL06 points to LIS CRS, map to pixel indices, compute per-pixel mean of `h_mean`, return float32 array with NaN for empty pixels

## 4. Output

- [x] 4.1 Implement `write_output()` function: wrap gridded array in an `xr.Dataset` with `lat`/`lon` coordinate variables copied from the LIS grid, write as NetCDF with `h_mean` variable and NaN fill value

## 5. CLI & Integration

- [x] 5.1 Implement `main()` with `add_storage_args` / `storage_config_from_namespace` following the same pattern as other modules
- [x] 5.2 Wire up all steps in `main()`: load LIS grid, load/fetch ATL06, grid, write output
- [x] 5.3 Add `icesat2` entry point to `pyproject.toml` (or verify module is runnable via `python -m join_scratch.icesat2.atl06_regrid`)
