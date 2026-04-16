## Why

The JOIN project currently grids three datasets (AMSR2, VIIRS, CEDA) to the LIS model grid but lacks ICESat-2 ATL06 snow height observations. Adding ICESat-2 ATL06 snow-on snow height data provides a direct, independent observational dataset for validating and constraining LIS snowpack simulations.

## What Changes

- Add a new `icesat2` module under `src/join_scratch/icesat2/` following the same structure as `amsr2`, `viirs`, and `ceda`
- Use the SlideRule Earth Python client to retrieve ICESat-2 ATL06 data on demand
- Cache retrieved SlideRule data locally to avoid repeated remote queries
- Grid ICESat-2 ATL06 snow height values to the LIS model grid by averaging all observations falling within each pixel (sparse output with NA for pixels with no observations)
- Support the same storage argument pattern (local/remote) as other datasets
- Initial temporal window: first week of January 2019

## Capabilities

### New Capabilities

- `icesat2-atl06-gridding`: Retrieve ICESat-2 ATL06 snow height data via SlideRule, cache locally, and grid to the LIS model grid with mean aggregation and NA fill for empty pixels

### Modified Capabilities

## Impact

- New Python dependency: `sliderule` (SlideRule Earth Python client)
- New module: `src/join_scratch/icesat2/`
- Output NetCDF files follow the same grid and structure as AMSR2, VIIRS, and CEDA outputs but are spatially sparse
- No changes to existing modules or interfaces
