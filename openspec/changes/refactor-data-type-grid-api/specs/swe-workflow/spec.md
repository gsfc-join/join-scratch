## ADDED Requirements

### Requirement: workflows/swe directory
A top-level `workflows/swe/` directory SHALL contain: LIS grid helpers (`lis_grid.py`), CLI entry points for each dataset, benchmark scripts, visualization scripts, and a `visualize_utils.py` with the `_lis_boundary_lonlat` helper.

#### Scenario: LIS grid helpers in workflows/swe
- **WHEN** searching for `build_lis_area_definition`
- **THEN** it is found only in `workflows/swe/lis_grid.py`, not in any `src/join_scratch/` module

### Requirement: Visualization helper consolidated
`_lis_boundary_lonlat` SHALL be defined once in `workflows/swe/visualize_utils.py` and imported by all visualization scripts.

#### Scenario: Single definition of _lis_boundary_lonlat
- **WHEN** searching the entire codebase for `def _lis_boundary_lonlat`
- **THEN** exactly one definition is found

### Requirement: Workflow scripts import from library
All scripts in `workflows/swe/` SHALL import dataset handlers, regridding, and utilities from `join_scratch`. Workflow scripts SHALL NOT contain duplicated logic already present in the library.

#### Scenario: Workflow script has no inline regrid logic
- **WHEN** reviewing any `workflows/swe/*_regrid.py` script
- **THEN** it contains only argument parsing, FSFile construction, handler instantiation, and calls to library functions

### Requirement: SlideRule fetch utility in workflows/swe
A `workflows/swe/fetch_atl06.py` script (or equivalent) SHALL handle the SlideRule API retrieval step for ICESat-2, producing cached parquet output. This is separate from `Icesat2FileHandler` which only reads already-cached parquet.

#### Scenario: Fetch utility separated from reader
- **WHEN** reviewing `src/join_scratch/datasets/icesat2.py`
- **THEN** it contains no direct calls to `sliderule.run` or SlideRule API functions
