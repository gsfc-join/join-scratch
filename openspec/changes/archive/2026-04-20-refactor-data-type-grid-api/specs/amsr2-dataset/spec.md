## ADDED Requirements

### Requirement: Amsr2FileHandler in datasets/amsr2.py
`src/join_scratch/datasets/amsr2.py` SHALL implement `Amsr2FileHandler` subclassing the base handler. `get_area_def(dsid)` SHALL return an `AreaDefinition` for the 0.1° equirectangular global grid. `get_dataset(dsid, ds_info)` SHALL return an `xr.DataArray` with fill values masked. `from_path(path, fs=None)` classmethod SHALL be provided.

#### Scenario: Amsr2FileHandler area definition is AreaDefinition
- **WHEN** `Amsr2FileHandler.from_path(...).get_area_def(dsid)` is called
- **THEN** the return value is an `AreaDefinition` instance

#### Scenario: Grid constants defined once
- **WHEN** reviewing `datasets/amsr2.py`
- **THEN** AMSR2 grid constants (1800×3600, 0.1°, radius_of_influence) appear only once

### Requirement: AMSR2 old modules removed
`amsr2/amsr2_regrid.py`, `amsr2/amsr2_visualize.py`, and `amsr2/amsr2_regrid_benchmark.py` SHALL be deleted. The `amsr2/` subdirectory SHALL be removed entirely.

#### Scenario: No amsr2 subdirectory in library
- **WHEN** listing `src/join_scratch/`
- **THEN** no `amsr2/` subdirectory is present
