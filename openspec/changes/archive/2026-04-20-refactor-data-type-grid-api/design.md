## Context

`src/join_scratch` currently organizes code by instrument: `amsr2/`, `ceda/`, `viirs/`, `icesat2/`. This has produced significant duplication and tightly couples SWE use-case workflow scripts with the general-purpose regridding library.

Three distinct input data geometries appear across the four instruments:
- **Regular (equirectangular) grid**: AMSR2 (0.1° global) and CEDA ESA CCI (0.1° global)
- **Satellite swath / sinusoidal tile**: VIIRS CGF Snow Cover (MODIS Sinusoidal 375 m tiles)
- **Point cloud / sparse observations**: ICESat-2 ATL06 (SlideRule-retrieved track points)

The LIS domain (Lambert Conformal, 1 km, Missouri) is the sole output grid in the current codebase and is specific to the SWE use case.

## Goals / Non-Goals

**Goals:**
- Implement a `BaseFileHandler`-style dataset interface for all four instruments, aligned with satpy's conventions.
- Use pyresample's `AreaDefinition` and `SwathDefinition` types throughout, following satpy's established patterns.
- Restructure into a flat `src/join_scratch/datasets/` subpackage (one module per instrument, or a subdirectory for VIIRS due to tile complexity).
- Move all LIS-specific grid code to `workflows/swe/` (it is SWE-workflow-specific, not a general library concern).
- Implement a regridding dispatch layer in `src/join_scratch/regrid/`.
- Use satpy's `FSFile` wrapper with the `obstore.fsspec` backend for S3 reads.
- Move SWE workflow scripts to `workflows/swe/`.
- Consolidate benchmark infrastructure and remove duplication.

**Non-Goals:**
- Plugging the file handlers into the full satpy reader YAML infrastructure (no `.yaml` reader config files).
- Adding new instruments or output grids.
- Changing the underlying regridding algorithms.
- Publishing the library as a package.
- Fixing pre-existing LSP/type errors unrelated to this refactor.

## Decisions

### D1: Follow satpy's `BaseFileHandler` API (partial adoption, without YAML infrastructure)

**Decision**: Each dataset module implements the `BaseFileHandler` interface: `__init__(filename, filename_info, filetype_info)`, `get_dataset(dataset_id, ds_info) -> xr.DataArray | None`, and `get_area_def(dsid) -> AreaDefinition | None`. The YAML-driven reader registration and `Scene` integration are not adopted — our library has its own simpler dispatch.

**Rationale**: Adopting the `BaseFileHandler` call signatures gives us compatibility with satpy's conventions without requiring the full reader plugin infrastructure. The three-argument constructor is slightly awkward when called directly, so we will add convenience factory classmethods (e.g., `Amsr2FileHandler.from_path(path, fs=None)`).

**Deviation from satpy**: No YAML reader configs; no `Scene` integration. `get_area_def` is called by our dispatch layer, not by satpy internals. Justified because this is a research library, not a satpy plugin.

### D2: ICESat-2 modeled as 1-D sparse data with `SwathDefinition`

**Decision**: ICESat-2 ATL06 is modeled following satpy's existing pattern for point/event data (MTG Lightning Imager, NUCAPS soundings): 1-D `xr.DataArray` with a `SwathDefinition` built from the point lat/lon coordinates. The SlideRule API retrieval step produces a cached parquet file; `Icesat2FileHandler` reads that parquet file (not the SlideRule API directly). A separate fetch utility in `workflows/swe/` handles the retrieval step.

**Rationale**: ICESat-2 points are irregular observations at specific lat/lon locations — exactly the use case satpy's `SwathDefinition` + 1-D array pattern is designed for. The MTG LI reader is the closest analogue; it returns 1-D arrays of lightning event locations with `SwathDefinition`. Keeping retrieval separate from reading mirrors satpy's separation of download and read.

**Deviation from satpy**: Parquet is not a file format satpy currently handles natively (satpy reads HDF5/NetCDF/GRIB). The `Icesat2FileHandler.get_dataset()` uses `geopandas.read_parquet` + `dask` to load the cached data. Justified because SlideRule's output is tabular point data, not raster/swath raster. No satpy reader exists for ICESat-2 ATL06 point output.

### D3: Flat `datasets/` subpackage with a directory only for VIIRS

**Decision**:
```
src/join_scratch/datasets/
├── __init__.py
├── base.py        # Thin wrapper/mixin on top of BaseFileHandler
├── amsr2.py       # Amsr2FileHandler
├── ceda.py        # CedaFileHandler
├── icesat2.py     # Icesat2FileHandler
└── viirs/         # Directory: tile loading is multi-file and complex
    ├── __init__.py
    └── handler.py  # ViirsFileHandler
```

VIIRS is a directory because its tile-compositing logic (sinusoidal projection, tile filtering, multi-HDF5 composition) is significantly more complex and may warrant additional helper modules alongside the main handler.

**Rationale**: One file per simple dataset is the minimal approach and avoids the previous one-directory-per-instrument layout that caused duplication. VIIRS is the natural exception because of its tile architecture.

**Alternative considered**: All four as flat `.py` files — feasible but would make `viirs.py` very large and harder to navigate.

### D4: `FSFile` with `obstore.fsspec` for S3 reads

**Decision**: Replace the existing `StorageConfig` / `s3fs` pattern with satpy's `FSFile` wrapper carrying an `obstore.fsspec` filesystem. File handlers receive `FSFile` objects (or plain `Path` objects for local files) as their `filename` argument. The `StorageConfig` dataclass is removed or deprecated in favor of this approach.

**Rationale**: `obstore` provides better performance for S3 reads through Rust-based async I/O. Satpy's `FSFile` is already an `os.PathLike` that is accepted by `xr.open_dataset`, `h5py.File`, etc. Using it directly avoids a custom abstraction layer. Satpy accepts any fsspec-compatible filesystem in `FSFile`, so `ObstoreFileSystem` can be passed as the `fs` argument.

**Note on satpy integration**: `obstore.fsspec` provides an `ObstoreFileSystem` implementing `fsspec.spec.AbstractFileSystem`. Satpy's `FSFile(path, fs=ObstoreFileSystem(...))` should work for any file handler using `open_dataset(self.filename, ...)` rather than `xr.open_dataset(str(self.filename), ...)`.

### D5: LIS grid lives in `workflows/swe/`

**Decision**: The `load_lis_grid`, `build_lis_area_definition`, and `LIS_RELPATH` helpers move to `workflows/swe/lis_grid.py`. The library provides no LIS-specific module. Generic grid construction utilities (e.g., building an `AreaDefinition` from a CF-convention projection dict) remain in the library if needed for multiple datasets.

**Rationale**: The LIS Lambert Conformal 1 km Missouri grid is only used in the SWE workflow. It is a specific parameterization of a regular projected grid, not a library abstraction. Moving it to `workflows/swe/` enforces the dependency direction (workflow knows about LIS; library does not).

### D6: Regrid dispatch routes by grid-type pair; method selects algorithm and backend

**Decision**: The dispatch key is the pair `(type(source_area), type(target_area))`. Within each pair, the `method` argument selects the algorithm, and the algorithm may be implemented by more than one backend. The module names reflect the grid-type pair, not the backend library.

| Grid-type pair | Available methods (→ backend) |
|---|---|
| `AreaDef → AreaDef` | `nearest` → pyresample; `bilinear` → xESMF or pyresample; `conservative` → xESMF only |
| `SwathDef 2-D → AreaDef` | `nearest` → pyresample; `bilinear` → pyresample EWA; `ewa` → pyresample; `bucket_avg` → pyresample; `bilinear_xesmf` → xESMF locstream (future) |
| `SwathDef 1-D → AreaDef` | `mean` → pixel binning (manual); extensible |

When multiple backends implement the same method (e.g., `bilinear` for `AreaDef → AreaDef`), the caller can pass an optional `backend=` argument to select explicitly, or the dispatch layer picks a default. This is intentional: different backends have different performance profiles and memory characteristics, which is critical for benchmarking and for scaling workflows from laptop-sized jobs to distributed/large-domain runs.

**Rationale**: Conflating "grid-type pair" with "backend library" (e.g., `regular_to_regular` == xESMF) would prevent comparing algorithms on the same data and would make it impossible to scale by swapping backends. The grid-type pair is a fixed structural property of the data; the backend choice is a runtime decision.

**Alternative considered**: Separate top-level functions per backend (e.g., `regrid_xesmf`, `regrid_pyresample`) — discards the dispatch structure and forces callers to know implementation details.

### D7: New directory layout for `src/join_scratch/`

```
src/join_scratch/
├── __init__.py
├── datasets/
│   ├── __init__.py
│   ├── base.py              # BaseFileHandler wrapper/mixin
│   ├── amsr2.py             # Amsr2FileHandler
│   ├── ceda.py              # CedaFileHandler
│   ├── icesat2.py           # Icesat2FileHandler (reads cached parquet)
│   └── viirs/
│       ├── __init__.py
│       └── handler.py       # ViirsFileHandler
├── regrid/
│   ├── __init__.py          # regrid(source, target_area, method, backend=None, **kwargs)
│   ├── regular_to_regular.py  # AreaDef→AreaDef: xESMF and pyresample backends
│   ├── swath_to_regular.py    # SwathDef 2-D→AreaDef: pyresample backends; xESMF locstream optional
│   └── sparse_to_regular.py   # SwathDef 1-D→AreaDef: pixel binning
└── utils/
    ├── __init__.py
    └── benchmark.py         # BenchmarkResult, _rss_mib, _time_call, render_report
```

`storage.py` is removed (replaced by `FSFile` + `obstore.fsspec`).

### D7: Inline single-call-site private helpers

`retrieve_atl06` (called only inside `load_or_fetch_atl06`) and `write_output` (called only in `main()`) are inlined into their respective callers when migrated to the workflow scripts.

## Risks / Trade-offs

- **Risk: `obstore.fsspec` API compatibility** — `obstore` is a newer library; the `ObstoreFileSystem` API may differ between versions. → Mitigation: Pin `obstore` version; document usage clearly.
- **Risk: `BaseFileHandler` constructor is awkward to call directly** — `(filename, filename_info, filetype_info)` is designed for satpy's internal use. → Mitigation: Add `from_path(path, fs=None)` factory classmethods that construct the `filename_info`/`filetype_info` dicts from sensible defaults.
- **Risk: ICESat-2 parquet handler deviates from satpy conventions** — No satpy precedent for parquet reads. → Mitigation: Document clearly; the deviation is justified by the data format.
- **Risk: Behavioral regression in regridding output** → Mitigation: No algorithm code is rewritten; benchmark and visualization scripts can be re-run after migration.
- **Risk: VIIRS tile compositing is complex** — The sinusoidal projection and multi-tile loading require careful handling of the `ViirsFileHandler` interface. → Mitigation: Each tile is one `ViirsFileHandler` instance; compositing is done at the workflow level by loading multiple handlers.

## Migration Plan

1. Create new subpackage stubs (`datasets/`, `regrid/`, `utils/`) with `__init__.py` files.
2. Implement `utils/benchmark.py` from the three duplicated sources.
3. Implement `datasets/base.py` with the `BaseFileHandler` wrapper.
4. Implement dataset handlers one at a time: AMSR2 → CEDA → VIIRS → ICESat-2.
5. Implement `regrid/` dispatch layer.
6. Create `workflows/swe/` with LIS grid helpers and migrated workflow scripts.
7. Delete old instrument subpackages.
8. Clean up unused imports and duplicate constants.

**Rollback**: All changes are on a single Git branch. Old instrument modules are deleted only after new structure is verified.

## Open Questions

- Should `base.py` subclass `satpy.readers.core.file_handlers.BaseFileHandler` directly (adding satpy as a required dependency even for non-satpy use) or define an independent ABC with the same interface? Given satpy is already a dependency (used for regridding), direct subclassing is likely fine but should be confirmed.
- Should `StorageConfig` be kept as a thin compatibility shim during transition, or removed immediately?
