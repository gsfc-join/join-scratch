## Why

The current `src/join_scratch` library is organized by instrument, which has led to significant code duplication (triplicated benchmark boilerplate, duplicated `compute_weights`/`load_regridder`, a shared helper function living in the wrong module) and tightly couples use-case logic with library logic. Refactoring around a common dataset interface modeled on satpy's `BaseFileHandler` pattern will make it straightforward to add new instruments, eliminate redundancy, and clearly separate the SWE workflow from the general-purpose library.

## What Changes

- Introduce a flat `src/join_scratch/datasets/` subpackage where each dataset is a module (`amsr2.py`, `ceda.py`, `viirs/`, `icesat2.py`) implementing a `BaseFileHandler`-style interface aligned with satpy's conventions.
- Use satpy/pyresample types (`AreaDefinition`, `SwathDefinition`) throughout; all deviations from satpy conventions are justified and documented.
- ICESat-2 ATL06 is modeled as 1-D sparse data with a `SwathDefinition`, following satpy's existing pattern for point/lidar data (e.g., MTG LI, CALIOP). The SlideRule API retrieval step produces cached parquet files that the handler reads.
- For remote reads, use satpy's `FSFile` wrapper with the `obstore.fsspec` backend for S3 access, in place of the current `s3fs`-based `StorageConfig`.
- Define a regridding dispatch layer in `src/join_scratch/regrid/` that accepts satpy-compatible datasets and routes to the appropriate algorithm.
- Move all LIS-specific grid code (LIS Lambert Conformal `AreaDefinition`, `load_lis_grid`, `LIS_RELPATH`) to `workflows/swe/` — the LIS grid is specific to the SWE use case, not the general library. The library instead provides a generic utility for constructing projected `AreaDefinition` objects from parameter dicts.
- Move all SWE use-case scripts (CLI entry points, benchmarks, visualizations) to `workflows/swe/`.
- Consolidate benchmark infrastructure into `src/join_scratch/utils/benchmark.py`.
- Remove unused imports, eliminate duplicate constant definitions, fix the double `build_lis_area_definition` call in `viirs_regrid.py`.

## Capabilities

### New Capabilities

- `dataset-interface`: A `BaseFileHandler`-derived abstract class in `src/join_scratch/datasets/base.py` establishing the common dataset interface for all instruments.
- `datasets`: Flat `datasets/` subpackage with one module per instrument implementing the dataset interface.
- `regrid-dispatch`: Unified regridding API routing (input dataset, output area, method) to the correct algorithm.
- `shared-utils`: Shared `utils/benchmark.py` consolidating benchmark infrastructure from all three benchmark files.
- `swe-workflow`: Top-level `workflows/swe/` directory containing SWE use-case scripts and the LIS grid helpers.

### Modified Capabilities

- `amsr2-dataset`: AMSR2 module refactored to `datasets/amsr2.py` implementing `BaseFileHandler`; CLI/workflow code removed to `workflows/swe/`.
- `ceda-dataset`: CEDA module refactored to `datasets/ceda.py` implementing `BaseFileHandler`; CLI/workflow code removed to `workflows/swe/`.
- `viirs-dataset`: VIIRS module refactored to `datasets/viirs/` (directory, due to tile complexity) implementing `BaseFileHandler`; CLI/workflow code removed to `workflows/swe/`.
- `icesat2-dataset`: ICESat-2 ATL06 module refactored to `datasets/icesat2.py` implementing `BaseFileHandler` over cached parquet files; SlideRule retrieval step separated as a fetch utility; CLI/workflow code removed to `workflows/swe/`.

## Impact

- **`src/join_scratch/`**: Major restructure — new `datasets/`, `regrid/`, `utils/` subpackages; old instrument subpackages removed.
- **CLI entry points**: Moved from instrument modules to `workflows/swe/`; existing invocations will need path updates.
- **No external API changes**: The library is not published as a package; no version bump required.
- **Dependencies**: `obstore` added for S3 access; `s3fs` usage replaced.
