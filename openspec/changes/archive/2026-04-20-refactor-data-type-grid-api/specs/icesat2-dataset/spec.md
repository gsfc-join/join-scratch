## ADDED Requirements

### Requirement: Icesat2FileHandler in datasets/icesat2.py
`src/join_scratch/datasets/icesat2.py` SHALL implement `Icesat2FileHandler` that reads a cached parquet file (produced by the SlideRule fetch step). `get_dataset(dsid, ds_info)` SHALL return a 1-D `xr.DataArray` with `attrs["area"]` set to a `SwathDefinition` built from the point lat/lon columns. `from_path(path, fs=None)` SHALL be provided. Following satpy's pattern for point data (MTG Lightning Imager, NUCAPS), data is represented as 1-D arrays with a `SwathDefinition`.

#### Scenario: Icesat2FileHandler returns 1-D data with SwathDefinition
- **WHEN** `Icesat2FileHandler.from_path("atl06_cache.parquet").get_dataset(dsid, ds_info)` is called
- **THEN** the result is a 1-D `xr.DataArray` and `result.attrs["area"]` is a `SwathDefinition`

### Requirement: Icesat2FileHandler does not call SlideRule
`datasets/icesat2.py` SHALL NOT import or call any SlideRule functions. It only reads parquet files.

#### Scenario: No sliderule imports in icesat2 handler
- **WHEN** reviewing `src/join_scratch/datasets/icesat2.py`
- **THEN** there is no import of `sliderule` or any `sliderule.*` module

### Requirement: ICESat-2 old modules removed
`icesat2/atl06_regrid.py` and `icesat2/atl06_visualize.py` SHALL be deleted. The `icesat2/` subdirectory under `src/join_scratch/` SHALL be removed.

#### Scenario: No icesat2 subdirectory in library src
- **WHEN** listing `src/join_scratch/`
- **THEN** no `icesat2/` subdirectory is present (the handler is now `datasets/icesat2.py`)
