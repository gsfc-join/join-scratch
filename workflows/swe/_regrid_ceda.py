def _regrid_ceda(
    input_dir: str,
    lis_grid: xr.Dataset,
    method: str,
    weights_dir: str,    
    current_date: pd.Timestamp,
    overwrite_weights: bool = False,    
    fs=None,
) -> dict[str, xr.DataArray]:
    """Regrid the file found and return {var_name: DataArray}."""
    # Format the date to match how it appears in your filenames
    # Example format: "20190101"
    date_str = current_date.strftime("%Y%m%d") 
    
    all_files = _list_files(input_dir, ".nc", fs=fs)
    if not all_files:
        log.warning("No CEDA files found under %s — skipping", input_dir)
        return {}

    files = [f for f in all_files if date_str in f]
    
    path = files[0]
    log.info("CEDA: using %s", _path_name(path))

    if _is_s3(path):
        handler = handler_from_s3(CedaFileHandler, path, fs=fs)
    else:
        handler = CedaFileHandler.from_path(path)

    ds = handler.get_dataset()

    lat_vals = ds["lat"].values if "lat" in ds else ds["y"].values
    lon_vals = ds["lon"].values if "lon" in ds else ds["x"].values
    if lat_vals.ndim == 2:
        lat_vals = lat_vals[:, 0]
    if lon_vals.ndim == 2:
        lon_vals = lon_vals[0, :]
    source_grid = xr.Dataset(
        coords={"lat": np.sort(np.unique(lat_vals)), "lon": np.sort(np.unique(lon_vals))}
    )

    weights_local_dir = _ensure_local_dir(weights_dir)
    weights_path = weights_local_dir / f"ceda-lis-weights-{method}.nc"
    compute_weights(source_grid, lis_grid, weights_path, method=method, overwrite=overwrite_weights)
    regridder = load_regridder(source_grid, lis_grid, weights_path, method=method)

    # Restore lat/lon as dim names for xESMF
    ds_xesmf = ds.swap_dims({"y": "lat", "x": "lon"})

    log.info("CEDA: regridding swe and swe_std …")
    rg = regridder(ds_xesmf)

    def _da(arr, long_name, units):
        return xr.DataArray(
            arr.values.astype(np.float32),
            dims=["north_south", "east_west"],
            attrs={"long_name": long_name, "units": units, "source": _path_name(path)},
        )

    return {
        "ceda_swe": _da(rg["swe"], "CEDA ESA CCI snow water equivalent", "mm"),
        "ceda_swe_std": _da(rg["swe_std"], "CEDA ESA CCI SWE standard deviation", "mm"),
    }
