"""Regrid source descriptor: GCOM-W1 AMSR2 L3 Snow Depth.

This module only provides the source grid specification for ESMF weight
generation.  It has no dependency on icechunk or any other heavy library.

Interface
---------
    SRC_GRID_SPEC : dict   — equirectangular grid parameters for build_equirect_scrip()
"""

SRC_GRID_SPEC = {
    "nlat":    1800,
    "nlon":    3600,
    "lat_max":  89.95,
    "lat_min": -89.95,
    "lon_min":   0.05,
    "lon_max": 359.95,
    "title":   "AMSR2 0.1deg global equirectangular",
}
