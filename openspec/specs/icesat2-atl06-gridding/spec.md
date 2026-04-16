## ADDED Requirements

### Requirement: Retrieve ICESat-2 ATL06 data via SlideRule
The system SHALL retrieve ICESat-2 ATL06 snow height data using the SlideRule Earth Python client (`sliderule.icesat2.atl06p`), bounded by the LIS domain bounding box and a configurable temporal window (default: 2019-01-01 to 2019-01-07).

#### Scenario: Successful retrieval
- **WHEN** the module is run with network access and no local cache exists
- **THEN** SlideRule returns a GeoDataFrame of ATL06 point observations with `h_mean` values covering the LIS domain for the specified time window

#### Scenario: Retrieval with temporal bounds
- **WHEN** the default time window (first week of January 2019) is used
- **THEN** only observations with acquisition time in [2019-01-01, 2019-01-07] are returned

### Requirement: Cache SlideRule response locally
The system SHALL serialize the SlideRule GeoDataFrame to a local Parquet file after retrieval. On subsequent runs, the system SHALL load from the cache instead of querying SlideRule, unless `--force-download` is specified.

#### Scenario: Cache hit
- **WHEN** a valid Parquet cache file exists and `--force-download` is not set
- **THEN** the module loads the GeoDataFrame from the Parquet file without contacting SlideRule

#### Scenario: Cache miss
- **WHEN** no cache file exists
- **THEN** the module retrieves data from SlideRule and writes the result to the Parquet cache

#### Scenario: Force download overrides cache
- **WHEN** `--force-download` is passed as a CLI argument and a cache file exists
- **THEN** the module re-queries SlideRule and overwrites the existing cache file

### Requirement: Grid ATL06 point data to the LIS model grid
The system SHALL project ICESat-2 ATL06 point observations (lat/lon) to the LIS Lambert Conformal grid CRS and compute the mean `h_mean` value for all points falling within each LIS pixel.

#### Scenario: Pixel with one or more observations
- **WHEN** one or more ATL06 points fall within a LIS pixel
- **THEN** the output grid cell value SHALL be the arithmetic mean of all `h_mean` values in that pixel

#### Scenario: Pixel with no observations
- **WHEN** no ATL06 points fall within a LIS pixel
- **THEN** the output grid cell value SHALL be NaN (fill value)

#### Scenario: Sparse output
- **WHEN** the gridded output is produced for the first week of January 2019
- **THEN** the majority of grid cells SHALL be NaN, reflecting ICESat-2's narrow-beam sampling

### Requirement: Output NetCDF with LIS grid structure
The system SHALL write the gridded ATL06 snow height data to a NetCDF file with dimensions `(north_south, east_west)` and coordinate variables `lat` and `lon` matching those of the LIS input grid.

#### Scenario: Output dimensions match LIS grid
- **WHEN** the output NetCDF is opened
- **THEN** the `north_south` and `east_west` dimension sizes SHALL equal those of the LIS input grid's `lat` variable

#### Scenario: Output variable present
- **WHEN** the output NetCDF is opened
- **THEN** a variable named `h_mean` of type float32 SHALL be present with a `_FillValue` of NaN

### Requirement: Storage argument compatibility
The module SHALL accept the same `--storage` and `--storage-location` CLI arguments as the `amsr2`, `viirs`, and `ceda` modules, using `StorageConfig` from `join_scratch.storage` to read the LIS input grid file.

#### Scenario: Local storage default
- **WHEN** the module is run without storage arguments
- **THEN** the LIS grid is read from the local `_data-raw/` directory

#### Scenario: S3 storage
- **WHEN** `--storage s3` is passed
- **THEN** the LIS grid is read from the default S3 location using `StorageConfig`
