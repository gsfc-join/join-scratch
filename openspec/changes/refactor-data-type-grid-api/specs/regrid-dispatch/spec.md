## ADDED Requirements

### Requirement: Regridding dispatch function
`src/join_scratch/regrid/__init__.py` SHALL provide `regrid(source_handler, target_area, method, backend=None, **kwargs)`. Dispatch routes on `(type(source_area), type(target_area))`. The `method` argument selects the algorithm; `backend` optionally selects among multiple implementations of the same method. If `backend` is `None`, the dispatch layer picks a documented default for that method.

#### Scenario: AreaDefinition source dispatches to regular_to_regular
- **WHEN** the source handler returns an `AreaDefinition` from `get_area_def()` and the target is an `AreaDefinition`
- **THEN** `regrid()` routes to `regular_to_regular`

#### Scenario: 2-D SwathDefinition source dispatches to swath_to_regular
- **WHEN** the source data has a 2-D `SwathDefinition` area
- **THEN** `regrid()` routes to `swath_to_regular`

#### Scenario: 1-D SwathDefinition source dispatches to sparse_to_regular
- **WHEN** the source data has a 1-D `SwathDefinition` area
- **THEN** `regrid()` routes to `sparse_to_regular`

#### Scenario: Unsupported combination raises error
- **WHEN** source and target types form an unsupported pair
- **THEN** `regrid()` raises `NotImplementedError` with a message listing supported pairs

### Requirement: Multiple backends per method
Where more than one backend implements the same method, both SHALL be callable via the `backend` argument. The default backend for each method SHALL be documented in the module docstring.

#### Scenario: bilinear with explicit xESMF backend
- **WHEN** `regrid(source, target, method="bilinear", backend="xesmf")` is called for an `AreaDef → AreaDef` pair
- **THEN** the xESMF bilinear regridder is used

#### Scenario: bilinear with explicit pyresample backend
- **WHEN** `regrid(source, target, method="bilinear", backend="pyresample")` is called for an `AreaDef → AreaDef` pair
- **THEN** the pyresample bilinear regridder is used

#### Scenario: Unknown backend raises ValueError
- **WHEN** `regrid(source, target, method="bilinear", backend="invalid")` is called
- **THEN** a `ValueError` is raised listing valid backends for that method/pair

### Requirement: regular_to_regular module
`src/join_scratch/regrid/regular_to_regular.py` SHALL implement `AreaDefinition → AreaDefinition` regridding with the following method/backend matrix:
- `method="nearest"` → pyresample
- `method="bilinear"`, `backend="xesmf"` (default) or `backend="pyresample"` → respective library
- `method="conservative"` → xESMF only

xESMF helpers (`compute_weights`, `load_regridder`) SHALL be consolidated here from the current duplicate definitions in `amsr2_regrid.py` and `ceda_regrid.py`.

#### Scenario: conservative method only available via xESMF
- **WHEN** `regrid(source, target, method="conservative", backend="pyresample")` is called
- **THEN** a `ValueError` is raised indicating conservative regridding requires xESMF

### Requirement: swath_to_regular module
`src/join_scratch/regrid/swath_to_regular.py` SHALL implement `SwathDefinition (2-D) → AreaDefinition` regridding with the following methods, all via pyresample/satpy:
- `method="nearest"`, `method="bilinear"`, `method="ewa"`, `method="bucket_avg"`

The module SHALL be designed so that additional backends (e.g., xESMF locstream) can be added in the future without changing the dispatch interface.

#### Scenario: EWA method routes to pyresample
- **WHEN** `regrid(source, target, method="ewa")` is called for a 2-D swath source
- **THEN** pyresample's EWA resampler is used

### Requirement: sparse_to_regular module
`src/join_scratch/regrid/sparse_to_regular.py` SHALL implement `SwathDefinition (1-D) → AreaDefinition` regridding via CRS-projection + pixel-binning (mean aggregation). Additional aggregation methods (e.g., median, count) MAY be added later.

#### Scenario: Point cloud binned to grid
- **WHEN** 1-D point data with `SwathDefinition` is passed to `sparse_to_regular` with `method="mean"`
- **THEN** each output pixel contains the mean of all source points in that pixel; empty pixels are NaN
