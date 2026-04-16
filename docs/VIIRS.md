# VIIRS Sinusoidal Tile Grid

## Grid overview

VIIRS (and MODIS) distribute gridded products on the **MODIS Sinusoidal Tile Grid**,
a fixed global tiling of the sinusoidal (SIN) projection.

- **Projection**: Sinusoidal, `+proj=sinu +R=6371007.181 +nadgrids=@null +wktext`
- **Global grid**: 36 columns × 18 rows of tiles, denoted `hHHvVV`
  - `h` runs 00–35 (west to east)
  - `v` runs 00–17 (north to south, so `v00` = 80°–90°N, `v17` = 80°–90°S)
- **Tile size**: Each tile spans exactly 10° of latitude and a fixed SIN-projected
  width corresponding to 10° of arc along the equator
- **Pixel resolution**: 375 m for `VJ110A1F` (VIIRS/JPSS-1 CGF Snow Cover), giving
  3000 × 3000 pixels per tile
- **Coordinates**: `XDim` and `YDim` arrays in each file are SIN projection metres

### Tile size in projection units

```
R = 6371007.181 m  (sphere radius used by MODIS/VIIRS SIN grid)

tile_width_x  = 2 * pi * R / 36  ≈  1,111,950 m
tile_height_y = pi * R / 18       ≈  1,111,950 m
```

Both dimensions are equal (≈ 1,111,950 m ≈ 10° of arc on the sphere).

### Mapping tile indices to SIN coordinates

For a tile `(h, v)`, the south-west corner in SIN metres is:

```
x_min = (h - 18)       * tile_width_x
y_min = (9 - v - 1)    * tile_height_y   # y increases northward

x_max = x_min + tile_width_x
y_max = y_min + tile_height_y
```

### Converting SIN coordinates to geographic (lon/lat)

The sinusoidal projection encodes latitude directly in `y`:

```
lat = y / R          (radians → degrees)
lon = x / (R * cos(lat))   (radians → degrees)
```

This is applied with pyproj in `viirs_regrid.py`:

```python
SIN_CRS = "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"
transformer = pyproj.Transformer.from_crs(SIN_CRS, "EPSG:4326", always_xy=True)
lon, lat = transformer.transform(x_metres, y_metres)
```

## Identifying tiles that overlap the LIS Missouri domain

The LIS domain covers approximately **35°–50°N, 94°–108°W**.

Using the formulas above, the four corner longitudes of each candidate tile were
computed by evaluating lon at both the western and eastern SIN x-edges and at both
the southern and northern y-edges (latitude affects the lon conversion).  Tiles whose
bounding box intersected the LIS domain were identified as:

| Tile   | Lat range    | Lon range (approx)   | Overlap with LIS domain |
|--------|-------------|----------------------|------------------------|
| h09v04 | 40°–50°N    | 140°W – 104°W        | partial (western edge) |
| h10v04 | 40°–50°N    | 125°W –  91°W        | **core**               |
| h10v05 | 30°–40°N    | 104°W –  81°W        | southern edge          |
| h11v04 | 40°–50°N    | 109°W –  78°W        | **core**               |

The lon ranges are narrower than 10° because at 40°–50°N the sinusoidal projection
compresses horizontal distances by `cos(lat)` ≈ 0.74–0.77.

Bounding coordinates were confirmed from the HDF5 root attributes
(`WestBoundingCoord`, `EastBoundingCoord`, `SouthBoundingCoord`, `NorthBoundingCoord`)
after downloading the files.

## Sample data in this repository

`_data-raw/JOIN/VIIRS/VJ110A1F/2019/001/` contains tiles for **2019-01-01** (day 001):

- `VJ110A1F.A2019001.h09v04.*` — 6.1 M valid pixels
- `VJ110A1F.A2019001.h10v04.*` — 9.0 M valid pixels
- `VJ110A1F.A2019001.h10v05.*` — 8.7 M valid pixels
- `VJ110A1F.A2019001.h11v04.*` — 8.3 M valid pixels

The original two sample tiles (`h00v08`, `h00v09`) cover the equatorial Pacific
(180°W–170°W, 10°S–10°N) and have no overlap with the LIS domain; they are
retained for reference but produce all-NaN output when regridded to the LIS grid.
