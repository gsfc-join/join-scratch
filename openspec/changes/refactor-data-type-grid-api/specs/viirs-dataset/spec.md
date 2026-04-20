## ADDED Requirements

### Requirement: ViirsFileHandler in datasets/viirs/
`src/join_scratch/datasets/viirs/handler.py` SHALL implement `ViirsFileHandler`. Each instance corresponds to one HDF5 tile file. `get_dataset(dsid, ds_info)` SHALL return a 2-D `xr.DataArray` with `attrs["area"]` set to a `SwathDefinition` built from per-pixel sinusoidal lon/lat. `from_path(path, fs=None)` SHALL be provided. Tile compositing (loading multiple tiles for a domain) is the responsibility of the workflow layer.

#### Scenario: ViirsFileHandler area is SwathDefinition
- **WHEN** `ViirsFileHandler.from_path(...).get_dataset(dsid, ds_info)` is called
- **THEN** `result.attrs["area"]` is a `SwathDefinition`

### Requirement: VIIRS double area-definition call fixed
Within any single regrid operation, `build_lis_area_definition` (or equivalent) SHALL be called exactly once. The bug where it was called twice in `viirs_regrid.py`'s `main()` SHALL not be replicated.

#### Scenario: LIS area loaded once per operation
- **WHEN** reviewing the VIIRS SWE workflow script
- **THEN** the LIS area definition is constructed exactly once per invocation

### Requirement: VIIRS old modules removed
`viirs/viirs_regrid.py`, `viirs/viirs_visualize.py`, and `viirs/viirs_regrid_benchmark.py` SHALL be deleted. The `viirs/` subdirectory under `src/join_scratch/` SHALL be removed.

#### Scenario: No viirs subdirectory in library src
- **WHEN** listing `src/join_scratch/`
- **THEN** no `viirs/` subdirectory is present (it is now under `datasets/viirs/`)
