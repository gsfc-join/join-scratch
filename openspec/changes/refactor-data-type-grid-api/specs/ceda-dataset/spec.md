## ADDED Requirements

### Requirement: CedaFileHandler in datasets/ceda.py
`src/join_scratch/datasets/ceda.py` SHALL implement `CedaFileHandler`. `get_area_def(dsid)` SHALL return an `AreaDefinition` for the 0.1° equirectangular grid. `get_dataset(dsid, ds_info)` SHALL mask flag values and return an `xr.DataArray`. `from_path(path, fs=None)` SHALL be provided.

#### Scenario: CedaFileHandler area definition is AreaDefinition
- **WHEN** `CedaFileHandler.from_path(...).get_area_def(dsid)` is called
- **THEN** the return value is an `AreaDefinition`

### Requirement: No duplicate xESMF helpers in ceda
`datasets/ceda.py` SHALL NOT define `compute_weights` or `load_regridder`. These SHALL be imported from `join_scratch.regrid.regular_to_regular`.

#### Scenario: No xESMF helpers in ceda module
- **WHEN** reviewing `src/join_scratch/datasets/ceda.py`
- **THEN** neither `compute_weights` nor `load_regridder` is defined there

### Requirement: CEDA old modules removed
`ceda/ceda_regrid.py`, `ceda/ceda_visualize.py`, and `ceda/ceda_regrid_benchmark.py` SHALL be deleted. The `ceda/` subdirectory SHALL be removed entirely.

#### Scenario: No ceda subdirectory in library
- **WHEN** listing `src/join_scratch/`
- **THEN** no `ceda/` subdirectory is present
