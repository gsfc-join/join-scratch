## 1. Shared Infrastructure

- [x] 1.1 Add `obstore` to project dependencies (`pyproject.toml` or `pixi.toml`); verify `import obstore.fsspec` works in the environment
- [x] 1.2 Create `src/join_scratch/utils/` subpackage: `__init__.py` and `benchmark.py` consolidating `BenchmarkResult`, `_rss_mib()`, `_time_call()`, and `render_report()` from the three benchmark files
- [x] 1.3 Create `src/join_scratch/regrid/` subpackage stubs: `__init__.py`, `regular_to_regular.py`, `swath_to_regular.py`, `sparse_to_regular.py`
- [x] 1.4 Create `src/join_scratch/datasets/` subpackage stub: `__init__.py`, `base.py`

## 2. Dataset Base Class

- [x] 2.1 Implement `datasets/base.py`: subclass (or wrap) `satpy.readers.core.file_handlers.BaseFileHandler`; add `from_path(path, fs=None)` classmethod convention; document the three-argument constructor requirement
- [x] 2.2 Verify that `from satpy.readers.core.file_handlers import BaseFileHandler` works in the project environment; if not, define a compatible ABC with the same interface and document the deviation

## 3. AMSR2 Dataset Handler

- [x] 3.1 Create `src/join_scratch/datasets/amsr2.py` with `Amsr2FileHandler`: implement `get_area_def()` returning the 0.1° equirectangular `AreaDefinition`; `get_dataset()` loading HDF5 and masking fill values; `from_path(path, fs=None)` classmethod
- [x] 3.2 Export `Amsr2FileHandler` from `datasets/__init__.py`
- [x] 3.3 Delete `src/join_scratch/amsr2/` directory entirely

## 4. CEDA Dataset Handler

- [x] 4.1 Create `src/join_scratch/datasets/ceda.py` with `CedaFileHandler`: `get_area_def()` returning 0.1° equirectangular `AreaDefinition`; `get_dataset()` loading NetCDF and masking flags; `from_path(path, fs=None)`
- [x] 4.2 Export `CedaFileHandler` from `datasets/__init__.py`
- [x] 4.3 Delete `src/join_scratch/ceda/` directory entirely

## 5. VIIRS Dataset Handler

- [x] 5.1 Create `src/join_scratch/datasets/viirs/` directory: `__init__.py` and `handler.py` with `ViirsFileHandler`; one instance = one HDF5 tile; `get_dataset()` computes per-pixel sinusoidal lon/lat and sets `attrs["area"]` to `SwathDefinition`; `from_path(path, fs=None)`
- [x] 5.2 Export `ViirsFileHandler` from `datasets/__init__.py`
- [x] 5.3 Delete `src/join_scratch/viirs/` directory entirely

## 6. ICESat-2 Dataset Handler

- [x] 6.1 Create `src/join_scratch/datasets/icesat2.py` with `Icesat2FileHandler`: reads cached parquet file using `geopandas.read_parquet`; `get_dataset()` returns 1-D `xr.DataArray` with `attrs["area"]` set to `SwathDefinition` from point lat/lon; `from_path(path, fs=None)`; no SlideRule imports
- [x] 6.2 Export `Icesat2FileHandler` from `datasets/__init__.py`
- [x] 6.3 Delete `src/join_scratch/icesat2/` directory entirely

## 7. Regrid Dispatch Layer

- [x] 7.1 Implement `regrid/regular_to_regular.py`: routes `AreaDef → AreaDef` by `method` and optional `backend`; `method="nearest"` via pyresample; `method="bilinear"` via xESMF (default) or pyresample; `method="conservative"` via xESMF only; move `compute_weights()` and `load_regridder()` here (removing duplicate from `ceda_regrid.py`)
- [x] 7.2 Implement `regrid/swath_to_regular.py`: routes 2-D `SwathDef → AreaDef` by `method`; `nearest`, `bilinear`, `ewa`, `bucket_avg` all via pyresample; structure to allow future xESMF locstream backend
- [x] 7.3 Implement `regrid/sparse_to_regular.py`: routes 1-D `SwathDef → AreaDef`; `method="mean"` via CRS-projection + pixel-binning; migrated from `icesat2/atl06_regrid.py:grid_atl06()`
- [x] 7.4 Implement `regrid/__init__.py` dispatch: `regrid(source_handler, target_area, method, backend=None, **kwargs)`; route on `(type(source_area), type(target_area))`; raise `NotImplementedError` for unsupported pairs; raise `ValueError` for unknown backends with a message listing valid options; document default backends per method in module docstring

## 8. FSFile / obstore Migration

- [x] 8.1 Update all workflow scripts and any remaining usages of `StorageConfig` to use `FSFile(path, fs=ObstoreFileSystem(...))` for S3 files and plain `pathlib.Path` for local files
- [x] 8.2 Delete `src/join_scratch/storage.py`

## 9. SWE Workflow Scripts

- [x] 9.1 Create `workflows/swe/` directory
- [x] 9.2 Create `workflows/swe/lis_grid.py`: move `LIS_RELPATH`, `load_lis_grid()`, and `build_lis_area_definition()` here from `amsr2_regrid.py`
- [x] 9.3 Create `workflows/swe/visualize_utils.py` with the `_lis_boundary_lonlat` helper (consolidated from three visualize files)
- [x] 9.4 Create `workflows/swe/fetch_atl06.py`: SlideRule retrieval + parquet caching (extracted from `icesat2/atl06_regrid.py`); inline the old `retrieve_atl06` function
- [x] 9.5 Create `workflows/swe/amsr2_regrid.py`, `amsr2_visualize.py`, `amsr2_benchmark.py`: CLI/vis/bench scripts importing from library
- [x] 9.6 Create `workflows/swe/ceda_regrid.py`, `ceda_visualize.py`, `ceda_benchmark.py`
- [x] 9.7 Create `workflows/swe/viirs_regrid.py`, `viirs_visualize.py`, `viirs_benchmark.py` (tile compositing and domain-mapping logic moves here; fix the double `build_lis_area_definition` call)
- [x] 9.8 Create `workflows/swe/icesat2_regrid.py`, `icesat2_visualize.py` (inline `write_output`; `_lis_extent` helper into `visualize_utils.py` or inline)
- [x] 9.9 Move or archive `src/join_scratch/_scratch/swe_naive.py` to `workflows/swe/_scratch/`

## 10. Cleanup

- [x] 10.1 Remove unused import `import io` from `src/join_scratch/storage.py` (already removed with storage.py in 8.2; verify no other occurrence)
- [x] 10.2 Verify `_SIN_R` is defined exactly once in `datasets/viirs/handler.py` (not also inlined as `R = 6371007.181`)
- [x] 10.3 Confirm no occurrence of `"lis_input_NMP_1000m_missouri.nc"` in any `src/join_scratch/` module
- [x] 10.4 Run import smoke test: `python -c "from join_scratch.datasets import Amsr2FileHandler, CedaFileHandler, ViirsFileHandler, Icesat2FileHandler; from join_scratch.regrid import regrid"` and confirm no errors
