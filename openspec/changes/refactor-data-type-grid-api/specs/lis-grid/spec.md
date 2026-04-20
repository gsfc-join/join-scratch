## ADDED Requirements

### Requirement: FSFile with obstore.fsspec for S3
Dataset handlers SHALL accept `satpy.readers.core.remote.FSFile` objects (with an `obstore.fsspec.ObstoreFileSystem` as the `fs` argument) as their `filename` argument for S3-hosted files. The `StorageConfig` / `s3fs` abstraction SHALL be removed.

#### Scenario: Handler opens S3 file via FSFile + obstore
- **WHEN** a handler is instantiated with `FSFile("s3://bucket/path/file.h5", fs=ObstoreFileSystem(store=...))`
- **THEN** the handler opens the file without error

#### Scenario: Handler also accepts local Path
- **WHEN** a handler is instantiated with a plain `pathlib.Path`
- **THEN** the handler opens the file without error

### Requirement: StorageConfig removed
The `src/join_scratch/storage.py` module and `StorageConfig` class SHALL be removed. All usages in the codebase SHALL be replaced with `FSFile` + `obstore.fsspec` for S3 or plain `Path` for local files.

#### Scenario: storage.py does not exist post-refactor
- **WHEN** checking the library source tree after the refactor
- **THEN** `src/join_scratch/storage.py` is absent

### Requirement: obstore dependency declared
`obstore` SHALL be added to the project's declared dependencies (e.g., `pyproject.toml` or `pixi.toml`) alongside `satpy`.

#### Scenario: obstore importable in project environment
- **WHEN** `import obstore.fsspec` is run in the project environment
- **THEN** it imports without error
