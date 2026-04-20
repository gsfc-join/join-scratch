## ADDED Requirements

### Requirement: Dataset base interface
`src/join_scratch/datasets/base.py` SHALL define a base class (subclassing or wrapping `satpy.readers.core.file_handlers.BaseFileHandler`) establishing the common interface: `__init__(filename, filename_info, filetype_info)`, `get_dataset(dataset_id, ds_info) -> xr.DataArray | None`, `get_area_def(dsid) -> AreaDefinition | None`, `start_time` property, `end_time` property.

#### Scenario: Handler contract enforced
- **WHEN** a class subclasses the base and fails to implement `get_dataset`
- **THEN** instantiating that class raises `TypeError`

#### Scenario: FSFile accepted as filename
- **WHEN** a handler is instantiated with a `satpy.readers.core.remote.FSFile` as `filename`
- **THEN** the handler opens the file without error using `open_dataset(self.filename, ...)` semantics

### Requirement: Factory classmethod on all handlers
Every dataset handler SHALL provide a `from_path(path, fs=None)` classmethod that constructs the handler with sensible default `filename_info` and `filetype_info` dicts, so callers are not required to pass satpy's YAML-derived dicts manually.

#### Scenario: Handler created without YAML dicts
- **WHEN** `Amsr2FileHandler.from_path("/path/to/file.h5")` is called
- **THEN** a valid handler is returned that can call `get_dataset`

### Requirement: Datasets use AreaDefinition and SwathDefinition
All handlers SHALL use `pyresample.geometry.AreaDefinition` for regular-grid sources (AMSR2, CEDA) and `pyresample.geometry.SwathDefinition` for swath/point sources (VIIRS, ICESat-2), consistent with satpy conventions.

#### Scenario: Regular grid handler returns AreaDefinition
- **WHEN** `handler.get_area_def(dsid)` is called on `Amsr2FileHandler` or `CedaFileHandler`
- **THEN** the return value is an instance of `pyresample.geometry.AreaDefinition`

#### Scenario: Point/swath handler area set on DataArray
- **WHEN** `handler.get_dataset(dsid, ds_info)` is called on `ViirsFileHandler` or `Icesat2FileHandler`
- **THEN** the returned `xr.DataArray` has `attrs["area"]` set to a `SwathDefinition`

### Requirement: Public exports from datasets subpackage
`src/join_scratch/datasets/__init__.py` SHALL export `Amsr2FileHandler`, `CedaFileHandler`, `ViirsFileHandler`, and `Icesat2FileHandler`.

#### Scenario: Importable from datasets subpackage
- **WHEN** `from join_scratch.datasets import Amsr2FileHandler, CedaFileHandler, ViirsFileHandler, Icesat2FileHandler`
- **THEN** all four names import without error
