# SWE Workflow API Design

High-level design for the Snow Water Equivalent (SWE) multi-dataset processing API.
This document covers architecture, data flow, and open questions. It is a design artifact —
no infrastructure is implemented here.

---

## Overview

The API accepts a spatial bounding box and a single date. It triggers a parallel pipeline
that virtualizes and regrids four satellite/model datasets onto a common 1 km Lambert
Conformal grid (the LIS NMP/Missouri domain), then merges them into a single output NetCDF.

**Output:** One `swe_combined_{date}.nc` per execution, written to S3. No temporal stacking.

---

## Datasets

| Dataset | Format | Source | Virtualization | Regrid Method |
|---|---|---|---|---|
| AMSR2 | HDF5 (0.1° global equirectangular) | JAXA G-Portal (HTTPS, anonymous) | IceChunk virtual store on S3 | ESMF bilinear → scipy sparse |
| CEDA ESA CCI SWE | NetCDF (0.1° global equirectangular) | CEDA Archive (HTTPS, `data.ceda.ac.uk`)| IceChunk virtual store on S3 | ESMF bilinear → scipy sparse |
| VIIRS CGF Snow Cover | HDF5 (MODIS sinusoidal tiles ~500m) | NSIDC DAAC, AWS S3 `us-west-2` protected bucket (`nsidc-cumulus-prod-protected`) — requires Earthdata Login + temporary AWS credentials from `/s3credentials` endpoint | IceChunk virtual store on S3 (tiled sinusoidal mosaic, 36×18 global grid) | ESMF bilinear (4×4 sub-tile parallel) → scipy sparse |
| ICESat-2 ATL06 | GeoParquet (point cloud) | SlideRule (HTTP API) | None — fetched on demand | pyproj CRS projection → NumPy pixel-binning mean (no weights) |

**Target grid:** LIS 1 km Lambert Conformal Conic, NMP/Missouri domain
(`JOIN/lis_input_NMP_1000m_missouri.nc` on S3).

---

## Entry Point

```
API Gateway (HTTP POST)
  │
  │  Request body:
  │  {
  │    "date": "2019-01-15",
  │    "bbox": {                    ← optional, defaults to full LIS domain
  │      "lat_min": ...,
  │      "lat_max": ...,
  │      "lon_min": ...,
  │      "lon_max": ...
  │    },
  │    "datasets": ["amsr2", "ceda", "viirs", "icesat2"]  ← optional, defaults to all
  │  }
  │
  ▼
Lambda: Parse + Dispatch
  - Validates inputs
  - Defaults datasets to all four if not specified
  - Fans out one StartExecution per requested dataset SFN (async, non-blocking)
  - Writes execution state to DynamoDB keyed by execution_id
  - Returns execution_id to caller immediately
```

The `bbox` is used by VIIRS tile filtering only. The ICESat-2 SlideRule query uses the
full LIS domain polygon derived from the LIS grid file, not the caller's `bbox`.
AMSR2 and CEDA cover the full LIS domain natively and do not use the bbox.

Caller polls `GET /status/{execution_id}` to check progress and retrieve the S3 output
URI when complete.

---

## Components

Three layers: an orchestration layer that manages execution state, dataset workflows
that define the per-dataset pipeline, and shared microservices that do the actual work.

### Orchestration Layer

3 Lambdas. Manage the lifecycle of an API call across all datasets.

| Service | Type | Responsibility |
|---|---|---|
| `api-gateway-handler` | Lambda | Validates request, resolves defaults, generates `execution_id`, writes initial DynamoDB record, fans out dataset SFN executions async, returns `execution_id` to caller |
| `collector` | Lambda | Triggered on each dataset SFN completion via `SendTaskSuccess`. Updates DynamoDB (pending → completed). When all datasets complete, invokes `stack-outputs`. |
| `status-check` | Lambda | Reads DynamoDB execution record for a given `execution_id`. Returns current status, which datasets are pending/completed, and the output S3 URI when done. |

### Dataset Workflows

4 Standard Step Functions. Each owns the full pipeline for one dataset — cache checks,
virtualization, weight generation, and regridding — assembled from shared microservices.
These are workflow definitions, not independently deployed services.

| Workflow | Composes |
|---|---|
| `amsr2-pipeline` | `preflight-check` → parallel(`icechunk-writer`, `esmf-weight-gen`) → `sparse-regrid` |
| `ceda-pipeline` | `preflight-check` → parallel(`icechunk-writer`, `esmf-weight-gen`) → `sparse-regrid` |
| `viirs-pipeline` | `preflight-check` → parallel(`icechunk-writer`, Map:`esmf-weight-gen`×16 → `merge-viirs-weights`) → `sparse-regrid` |
| `icesat2-pipeline` | `preflight-check` → `sliderule-fetch` → `point-cloud-regrid` |

### Shared Microservices

8 Lambda functions and Batch/Fargate job definitions. Stateless, parameterized, and
dataset-agnostic. Deployed once, invoked by multiple dataset workflows.

| Service | Type | Responsibility |
|---|---|---|
| `preflight-check` | Lambda | Given a list of S3 URIs, returns an exists/missing flag for each. Gates all downstream compute. |
| `icechunk-writer` | Lambda | Opens or creates an IceChunk repository, writes virtual chunk references, commits. Parameterized for source type (HTTPS vs S3) and virtualization strategy. |
| `esmf-weight-gen` | Batch/Fargate | Builds SCRIP source and target grids, runs `ESMF_RegridWeightGen`, uploads weights file to S3. Parameterized for source grid spec. |
| `merge-viirs-weights` | Lambda | Reads 16 per-tile VIIRS weight `.nc4` files from S3, merges via `merge_esmf_weights_2d()`, uploads merged `.nc4`. Used only by `viirs-pipeline`. |
| `sparse-regrid` | Batch/Fargate | Loads weights `.nc4` as scipy CSR matrix, applies NaN-safe sparse matrix multiply (`W @ src`), writes regridded NetCDF to S3. No ESMF binary dependency. |
| `point-cloud-regrid` | Batch/Fargate | Projects 1-D lon/lat point cloud into the LIS LCC CRS via pyproj, bins observations into LIS pixels by floor-division, computes per-pixel mean via `np.add.at`, writes output NetCDF to S3. No weight files. |
| `sliderule-fetch` | Lambda | Issues a SlideRule `atl06x` query with the full LIS domain polygon and date, writes resulting GeoParquet to S3. |
| `stack-outputs` | Lambda | Reads per-dataset output NetCDF files from S3, merges into a single flat `swe_combined_{date}.nc`, writes to S3, updates DynamoDB execution record. |

---

## Architecture

### Design Principles

The system is built across three layers:

1. **Shared microservices** — Lambda functions and Batch/Fargate job definitions that
   each implement a single well-defined capability (check cache, write virtual store,
   generate weights, regrid, fetch). They are stateless, parameterized, and do not know
   which dataset is calling them. Where datasets share a compute method (e.g. AMSR2,
   CEDA, and VIIRS all use the ESMF → scipy sparse path), they invoke the same deployed
   job definition with different parameters.

2. **Dataset workflows** — One Standard SFN per dataset. Each SFN owns its full pipeline:
   cache checks, data access, virtualization, weight generation, and regridding to the LIS
   grid. Assembled from shared microservices with dataset-specific parameters. Dataset
   workflows are independently deployable — updating one has no effect on the others.

3. **Orchestration layer** — A thin, stateless Lambda that fans out the requested dataset
   SFNs in parallel and returns immediately. Completion is tracked asynchronously via a
   Collector Lambda and DynamoDB. This avoids Lambda's 15-minute timeout limit while
   keeping the orchestration layer dynamic and configurable without redeploying any Step
   Functions definitions.

All Step Functions use the **Standard** workflow type. The total pipeline duration is
bounded by the slowest dataset (potentially 10+ minutes for VIIRS or large AMSR2 date
ranges). Standard workflows have no duration limit and provide full execution history
and per-state audit logging.

---

### Full System Flow

```
API Gateway
     │
     │  POST { date, bbox, datasets? }
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Lambda: Orchestrator                                           │
│                                                                 │
│  - Defaults datasets → ["amsr2", "ceda", "viirs", "icesat2"]   │
│  - Looks up dataset → SFN ARN from config (SSM or env vars)    │
│  - For each dataset:                                            │
│      StartExecution(sfn_arn, { date, bbox, execution_id })     │
│      (async — does not wait for result)                         │
│  - Writes to DynamoDB:                                          │
│      { execution_id, pending: [all datasets], completed: [] }  │
│  - Returns execution_id immediately                             │
└──────────────┬──────────────────────────────────────────────────┘
               │  fires and exits (~1 second)
               │
     ┌─────────┴──────────────────────────────────┐
     │         (all run concurrently)              │
     ▼         ▼              ▼           ▼        │
  amsr2     ceda           viirs       icesat2     │
  SFN       SFN            SFN         SFN         │
  (Standard)(Standard)    (Standard)  (Standard)  │
     │         │              │           │        │
     └─────────┴──────────────┴───────────┘        │
               │  each SFN calls SendTaskSuccess    │
               │  with execution_id on completion   │
               ▼                                    │
┌──────────────────────────────────────────────────┘
│  Lambda: Collector  (triggered by each SFN completion)
│
│  - Updates DynamoDB: move dataset from pending → completed
│  - If pending is now empty for this execution_id:
│      invoke stack-outputs Lambda
│  - Otherwise: exit (wait for remaining datasets)
└──────────────────────────────────────────────────
               │
               ▼
┌──────────────────────────────────────────────────┐
│  Lambda: stack-outputs                           │
│                                                  │
│  - Reads per-dataset output .nc URIs from        │
│    DynamoDB execution record                     │
│  - xarray merge → swe_combined_{date}.nc → S3   │
│  - Updates DynamoDB: status = complete,          │
│    output_uri = s3://...swe_combined_{date}.nc   │
└──────────────────────────────────────────────────┘
```

---

### Shared Microservices Detail

Deployed once. Invoked by dataset workflows with dataset-specific parameters.

---

#### `preflight-check` (Lambda)

| | |
|---|---|
| **Accepts** | List of S3 URIs |
| **Returns** | Per-URI exists/missing flags |
| **Used by** | All four dataset workflows at start |

---

#### `icechunk-writer` (Lambda)

| | |
|---|---|
| **Accepts** | `store_uri`, `strategy`, `source_uris`, `credentials` |
| **Does** | Opens or creates IceChunk repo, writes virtual chunk references, commits |
| **Used by** | AMSR2, CEDA, VIIRS workflows |

Dataset-specific behavior is injected via a strategy registry. The service structure is
identical across datasets — only the functions differ.

```python
STRATEGY_REGISTRY = {
    "amsr2": {
        "init_fn":       preallocate_amsr2_schema,  # writes coord arrays + preallocated
        "virtualize_fn": virtualize_amsr2,           # time slots 2012-2030 on create
        "credential_manager": None
    },
    "ceda": {
        "init_fn":       None,                       # no schema init needed
        "virtualize_fn": virtualize_ceda,            # open_virtual_mfdataset over all .nc
        "credential_manager": None
    },
    "viirs": {
        "init_fn":       None,                       # no schema init needed
        "credential_manager": earthaccess
        "virtualize_fn": virtualize_viirs,           # open_virtual_dataset per tile ×36×18;
                                                     # ManifestArray fill for missing tiles;
                                                     # concat YDim → XDim → time
    },
}

def run(event):
    strategy = STRATEGY_REGISTRY[event["strategy"]]
    storage  = make_storage(event)           # same for all datasets
    config   = make_config(event)            # same for all datasets
    repo     = open_or_create(storage, config, init_fn=strategy["init_fn"])
    manifest = strategy["virtualize_fn"](event["source_uris"])
    write_and_commit(repo, manifest)         # same for all datasets
```

Adding a new dataset = add one entry to `STRATEGY_REGISTRY`. No changes to `run()`,
`make_storage()`, `make_config()`, or `write_and_commit()`.

**Config differences by dataset:**

| Dataset | VirtualChunkContainer | Credentials | Notes |
|---|---|---|---|
| AMSR2 | HTTPS, JAXA G-Portal URL | anonymous | preallocated time dimension |
| CEDA | **HTTPS**, CEDA Archive URL (`data.ceda.ac.uk`) | anonymous? | no slot index; byte-range GETs supported by CEDA's HTTPS server, so chunk-wise virtual reads work the same as any other HTTPS `VirtualChunkContainer` |
| VIIRS | **S3**, named `"viirs-snow-s3"`, prefix `JOIN/VIIRS/VJ110A1F/`, bucket `nsidc-cumulus-prod-protected` (`us-west-2`) | `s3_credentials(from_env=True)` — **short-lived (1-hour) STS credentials** obtained from NSIDC's `/s3credentials` endpoint via Earthdata Login; must be refreshed if the writer runs longer than ~1 hour | fills missing tiles with `ManifestArray.with_fill_value_only(_FillValue)`; source bucket is **non-listable** — object keys for each of the 36×18 tiles must be resolved ahead of time (CMR query or pre-built granule index), not discovered via bucket listing at virtualization time |

---

#### `esmf-weight-gen` (Batch/Fargate)

| | |
|---|---|
| **Accepts** | `source_grid_spec`, `lis_grid_uri`, `weights_output_uri`, `method` |
| **Does** | Builds SCRIP source and target grids, runs `ESMF_RegridWeightGen`, uploads weights `.nc4` to S3 |
| **Used by** | AMSR2, CEDA, VIIRS workflows |

**Dataset-specific params:**

| Dataset | Source grid | ESMF flags | Invocations |
|---|---|---|---|
| AMSR2 | Hardcoded 1800×3600 equirectangular constants | `--method bilinear` | 1 |
| CEDA | Read lat/lon from first CEDA `.nc` file (fetched over HTTPS, byte-range) | `--method bilinear` | 1 |
| VIIRS | IceChunk store XDim/YDim sinusoidal coords transformed to lat/lon via cartopy `Sinusoidal` CRS | `--method bilinear -r` (regional, not global) | 16 in parallel (one per 4×4 sub-tile of the LIS bbox in sinusoidal projection space); outputs merged by `merge-viirs-weights` |

---

#### `merge-viirs-weights` (Lambda)

| | |
|---|---|
| **Accepts** | List of 16 per-tile weight `.nc4` S3 URIs, `output_uri` |
| **Does** | Calls `merge_esmf_weights_2d()` to re-index 16 tile-local sparse triplets into a single global source grid index, uploads merged `.nc4` to S3 |
| **Used by** | VIIRS workflow only, after the 16-way `esmf-weight-gen` Map state |

The merge re-indexes local per-tile flat ESMF indices into global sinusoidal grid
positions using `row_band_offsets` / `col_band_offsets` accounting for the 2D tile layout.

---

#### `sparse-regrid` (Batch/Fargate)

| | |
|---|---|
| **Accepts** | `source_uri`, `weights_uri`, `lis_grid_uri`, `output_uri`, `fill_value`, `scale_factor` |
| **Does** | Loads weights `.nc4` as scipy CSR matrix, applies NaN-safe sparse matrix multiply, writes regridded NetCDF to S3 |
| **Used by** | AMSR2, CEDA, VIIRS workflows |

Does not use xESMF or the ESMF binary at runtime. Depends only on scipy.
The weights file is a COO sparse matrix serialized as NetCDF (`S`, `row`, `col` arrays).

**Weight application (NaN-safe two-multiply pattern):**

```
W         = csr_matrix((S, (row, col)), shape=(n_dst, n_src))
src_clean = where(valid, src_flat, 0.0)   # NaN pixels → 0
dst_vals  = W @ src_clean                 # weighted sum
dst_wsum  = W @ valid_mask                # sum of valid weights
result    = dst_vals / dst_wsum           # renormalise; no valid coverage → NaN
```

**Dataset-specific params:**

| Dataset | Source encoding | Variable | Notes |
|---|---|---|---|
| AMSR2 | int16, fill/scale factor | `Geophysical Data` | IceChunk store URI |
| CEDA | float32, no scale factor | `swe` | IceChunk store URI (virtual chunks resolved over HTTPS at read time — regrid job needs CEDA credentials available, same as the writer) |
| VIIRS | uint8 | `CGF_NDSI_Snow_Cover` | IceChunk store URI; virtual chunks resolved via S3 at read time — regrid job needs valid (non-expired) NSIDC temporary credentials, same 1-hour expiry constraint as the writer; YDim is descending — slice as `slice(ymax, ymin)` |

---

#### `point-cloud-regrid` (Batch/Fargate)

| | |
|---|---|
| **Accepts** | `parquet_uri`, `lis_grid_uri`, `output_uri` |
| **Does** | Projects 1-D lon/lat point cloud into LIS LCC CRS via pyproj, bins observations into LIS pixels by floor-division, computes per-pixel mean, writes output NetCDF to S3 |
| **Used by** | ICESat-2 workflow |

No weight files. No KDTree or radius-of-influence. Pure in-memory NumPy binning:

```
Transformer: EPSG:4326 → LIS LCC CRS  (pyproj)
col = floor((xs - x_min) / dx)
row = floor((y_max - ys) / dy)          # row 0 = northernmost
np.add.at(val_sum, row*nx + col, values)
np.add.at(count,   row*nx + col, 1)
result = where(count > 0, val_sum / count, nan).reshape(ny, nx)
```

Each source point contributes equally to its containing pixel. Pixels with no
observations are NaN. `float64` accumulator, `float32` output.

---

#### `sliderule-fetch` (Lambda)

| | |
|---|---|
| **Accepts** | `lis_grid_uri`, `date`, `output_parquet_uri` |
| **Does** | Derives the full LIS domain bounding polygon from the LIS grid file, issues a SlideRule `atl06x` query, writes resulting GeoParquet to S3 |
| **Used by** | ICESat-2 workflow |

The spatial filter is the full LIS domain polygon (CCW-closed list of lon/lat vertices
derived from the LIS grid's lat/lon extents), not the caller's `bbox`. The caller's `bbox`
is not forwarded to SlideRule.

---

#### `stack-outputs` (Lambda)

| | |
|---|---|
| **Accepts** | List of per-dataset output `.nc` S3 URIs, `output_uri` |
| **Does** | xarray merge of all per-dataset variables → `swe_combined_{date}.nc` → S3; updates DynamoDB execution record |
| **Used by** | Collector Lambda after all dataset workflows complete |

---

### Dataset Workflow Details

Each dataset workflow receives `{ date, bbox, lis_grid_uri, execution_id }` as input
and on completion calls `SendTaskSuccess` with its output `.nc` S3 URI, triggering the
Collector.

#### amsr2-pipeline

```
Task: preflight-check
  Check: IceChunk store slot populated for this date
  Check: ESMF weights file on S3
  → { slot_populated, weights_exist }
│
Parallel State:  (both branches run concurrently — neither depends on the other)
│
├── Choice: slot_populated?
│     Yes → Pass
│     No  → Map State (one icechunk-writer Lambda per date)
│             icechunk-writer: HTTPS → JAXA G-Portal → IceChunk slot
│             (preallocated slots, safe to fan out — different chunk keys per date)
│
└── Choice: weights_exist?
      Yes → Pass
      No  → Task: esmf-weight-gen Batch/Fargate
              (hardcoded 1800×3600 AMSR2 grid + LIS grid → weights .nc4 → S3)
│
(join — wait for both branches)
│
Task: sparse-regrid Batch/Fargate
  source: IceChunk store URI
  weights: ESMF weights .nc4
  → JOIN/outputs/amsr2_{date}.nc
```

#### ceda-pipeline

Structurally identical to amsr2-pipeline. Virtualization and weight generation run
concurrently — neither depends on the other. **CEDA is HTTPS-sourced, like AMSR2 — not
S3 — so the writer authenticates with a CEDA account rather than an anonymous or
S3-credentialed access path.**

```
Task: preflight-check
  Check: IceChunk store populated for CEDA (JOIN/icechunk-stores/CEDA)
  Check: ESMF weights file on S3
  → { store_populated, weights_exist }
│
Parallel State:  (both branches run concurrently — neither depends on the other)
│
├── Choice: store_populated?
│     Yes → Pass
│     No  → Task: icechunk-writer Lambda
│             open_virtual_mfdataset over all CEDA .nc files via HTTPS
│             (CEDA Archive, data.ceda.ac.uk — byte-range GETs, CEDA account login)
│             HTTPS VirtualChunkContainer
│             → IceChunk store at JOIN/icechunk-stores/CEDA
│
└── Choice: weights_exist?
      Yes → Pass
      No  → Task: esmf-weight-gen Batch/Fargate
              (read grid coords from first CEDA .nc file over HTTPS + LIS grid → weights .nc → S3)
│
(join — wait for both branches)
│
Task: sparse-regrid Batch/Fargate
  source: IceChunk store URI (JOIN/icechunk-stores/CEDA)
  weights: ESMF weights .nc
  → JOIN/outputs/ceda_{date}.nc
```

Differences from AMSR2:

| Building block | AMSR2 params | CEDA params |
|---|---|---|
| icechunk-writer | HTTPS VirtualChunkContainer, manual per-chunk slot insertion, JAXA G-Portal URL, preallocated time dimension, **anonymous access** | HTTPS VirtualChunkContainer, `open_virtual_mfdataset` over all `.nc` files, no preallocation, **requires CEDA account login (not anonymous)** |
| esmf-weight-gen | Hardcoded 1800×3600 grid constants | Read grid coords from first CEDA `.nc` file (HTTPS byte-range) |
| sparse-regrid | int16 fill/scale encoding | float32, no scale factor; needs CEDA credentials available at read time to resolve virtual chunks |

#### viirs-pipeline

```
Task: preflight-check
  Check: VIIRS IceChunk store populated for this date
  Check: merged ESMF weights file on S3
  → { store_populated, weights_exist }
│
Parallel State:  (both branches run concurrently — neither depends on the other)
│
├── Choice: store_populated?
│     Yes → Pass
│     No  → Task: icechunk-writer Lambda
│             Fetch NSIDC temporary S3 credentials via Earthdata Login
│             (/s3credentials endpoint — 1-hour expiry; refresh if writer runtime
│             risks exceeding this window)
│             open_virtual_dataset per tile × 36×18 MODIS sinusoidal grid
│             (object keys resolved ahead of time — bucket is non-listable)
│             ManifestArray.with_fill_value_only() for missing granules
│             concat YDim → XDim → time → IceChunk store
│             S3 VirtualChunkContainer ("viirs-snow-s3", nsidc-cumulus-prod-protected,
│             us-west-2)
│             → IceChunk store at JOIN/icechunk-stores/VIIRS/VJ110A1F
│
└── Choice: weights_exist?
      Yes → Pass
      No  → Map State (16 parallel esmf-weight-gen Batch/Fargate jobs)
              Each job receives one (x_slice, y_slice) sub-tile of the LIS
              bbox decomposed in sinusoidal projection space.
              Per job:
                - Slice IceChunk store on XDim/YDim to sub-tile extent
                - Transform sinusoidal XDim/YDim → lat/lon via cartopy
                - Write SCRIP source grid
                - Run ESMF_RegridWeightGen --method bilinear -r
                  (regional, not global) → per-tile weights .nc4 → S3
              After all 16 complete:
              Task: merge-viirs-weights Lambda
                merge_esmf_weights_2d(16 × .nc4) → merged weights .nc4
                → JOIN/cached-weights/VIIRS/lis-1km-missouri.nc4
│
(join — wait for both branches)
│
Task: sparse-regrid Batch/Fargate
  source: IceChunk store URI (JOIN/icechunk-stores/VIIRS/VJ110A1F)
  weights: merged ESMF weights .nc4
  variable: CGF_NDSI_Snow_Cover
  note: YDim is descending — slice as slice(ymax, ymin)
  note: needs valid (non-expired) NSIDC temporary credentials at read time to
        resolve virtual chunks — same 1-hour expiry constraint as the writer
  → JOIN/outputs/viirs_{date}.nc
```

Differences from AMSR2/CEDA:

| Building block | AMSR2/CEDA params | VIIRS params |
|---|---|---|
| icechunk-writer | Per-chunk slot insert (AMSR2) or `open_virtual_mfdataset` (CEDA), both HTTPS | Per-tile `open_virtual_dataset` × 36×18, **genuinely S3-native** (NSIDC DAAC / Earthdata Cloud); empty `ManifestArray` fill for missing tiles; requires **short-lived STS credentials** from Earthdata Login (1-hour expiry) rather than a static login/token; source bucket **non-listable** — tile object keys must be pre-resolved via CMR, not discovered by listing |
| esmf-weight-gen | Single invocation, equirectangular grid | 16 parallel invocations, sinusoidal SCRIP grid, `-r` regional flag, outputs merged by `merge-viirs-weights` |
| sparse-regrid | float32 or int16, equirectangular | uint8, `CGF_NDSI_Snow_Cover`; YDim descending requires `slice(ymax, ymin)`; read-time chunk resolution also needs fresh (non-expired) temporary S3 credentials |

#### icesat2-pipeline

```
Task: preflight-check
  Check: cached .parquet for this date on S3
  → { parquet_exists }
│
Choice: parquet_exists?
  No  → Task: sliderule-fetch Lambda
          SlideRule atl06x query: full LIS domain polygon + date → .parquet → S3
  Yes → Pass
│
Task: point-cloud-regrid Batch/Fargate
  source: .parquet URI (h_li column)
  algorithm: pyproj CRS projection + NumPy floor-division pixel-binning mean
  → JOIN/outputs/icesat2_{date}.nc
```

---

## Container Images

One image per compute environment, shared across all job definitions that share
the same dependencies. Deployed once, referenced by multiple Batch job definitions.

| Image | Contents | Used by |
|---|---|---|
| `swe-esmf` | ESMF_RegridWeightGen binary + join_scratch | `esmf-weight-gen` |
| `swe-scipy` | scipy + xarray + h5netcdf + icechunk + join_scratch | `sparse-regrid` |
| `swe-pointcloud` | pyproj + geopandas + xarray + join_scratch | `point-cloud-regrid` |
| `swe-icechunk` | icechunk + virtualizarr + obstore + join_scratch | `icechunk-writer` (Lambda) |
| `swe-sliderule` | sliderule-python + geopandas + join_scratch | `sliderule-fetch` (Lambda) |

---

## Compute Allocation

| Service | Type | Reason |
|---|---|---|
| `api-gateway-handler` | Lambda | Fires SFN executions and exits — ~1 second duration |
| `collector` | Lambda | DynamoDB read/write + conditional invoke — lightweight |
| `status-check` | Lambda | DynamoDB read only — lightweight |
| `preflight-check` | Lambda | S3 HEAD requests only, no compute |
| `icechunk-writer` | Lambda | No data downloaded — HTTP(S)/S3 range metadata + S3 writes only |
| `sliderule-fetch` | Lambda | HTTP API call to SlideRule, low compute |
| `merge-viirs-weights` | Lambda | scipy sparse concatenation of 16 small NC4 files — low compute (see open question #1) |
| `stack-outputs` | Lambda | xarray merge of 4 small 2D NetCDF outputs |
| `esmf-weight-gen` | Batch/Fargate | Requires `ESMF_RegridWeightGen` compiled binary |
| `sparse-regrid` | Batch/Fargate | Memory-intensive: scipy sparse matrix multiply over large LIS grid |
| `point-cloud-regrid` | Batch/Fargate | Memory: full ATL06 point cloud + LIS grid in memory; O(n_points) NumPy ops |

All Batch jobs run on **Fargate** — serverless containers, no EC2 fleet to manage.

---

## State: DynamoDB Execution Record

The Orchestrator Lambda writes one record per API call. The Collector Lambda updates it
as dataset SFNs complete. The status endpoint reads it.

```json
{
  "execution_id": "swe-20190115-a3f9c2",
  "date": "2019-01-15",
  "bbox": { "lat_min": ..., "lat_max": ..., "lon_min": ..., "lon_max": ... },
  "datasets_requested": ["amsr2", "ceda", "viirs", "icesat2"],
  "pending": ["ceda", "viirs"],
  "completed": {
    "amsr2": "s3://.../JOIN/outputs/amsr2_20190115.nc",
    "icesat2": "s3://.../JOIN/outputs/icesat2_20190115.nc"
  },
  "status": "running",          // running | complete | failed
  "output_uri": null,           // set when stack-outputs completes
  "created_at": "2026-07-01T12:00:00Z",
  "updated_at": "2026-07-01T12:04:23Z"
}
```

---

## Caching and Idempotency

The `preflight-check` Lambda runs at the start of each dataset workflow and gates all
downstream compute. Two artifact categories:

### Grid-geometry-dependent (reusable across dates)

Depend only on source grid shape — computed once, reused forever.

| Artifact | S3 path | Used by |
|---|---|---|
| AMSR2 ESMF weights | `JOIN/cached-weights/GCOM-W1-AMSR2-L3-SND/lis-1km-missouri.nc4` | `sparse-regrid` (AMSR2) |
| CEDA ESMF weights | `JOIN/cached-weights/CEDA/lis-1km-missouri.nc` | `sparse-regrid` (CEDA) |
| VIIRS ESMF weights | `JOIN/cached-weights/VIIRS/lis-1km-missouri.nc4` | `sparse-regrid` (VIIRS) |

### Date-dependent (one entry per date)

| Artifact | S3 path | Used by |
|---|---|---|
| AMSR2 IceChunk slot | `JOIN/icechunk-stores/GCOM-W1-AMSR2-L3-SND` (preallocated slot per date) | `sparse-regrid` (AMSR2) |
| CEDA IceChunk store | `JOIN/icechunk-stores/CEDA` (all dates, written once) | `sparse-regrid` (CEDA) |
| VIIRS IceChunk store | `JOIN/icechunk-stores/VIIRS/VJ110A1F` (all dates, written once per date range) | `sparse-regrid` (VIIRS) |
| ICESat-2 Parquet | `JOIN/results/swe/atl06_{date}.parquet` | `point-cloud-regrid` (ICESat-2) |

**AMSR2 IceChunk concurrency:** The store is preallocated with fixed slots per date
(2012–2030). Per-date `icechunk-writer` Lambdas write to non-overlapping chunk keys —
parallel fan-out is safe. Concurrent writes to the same date slot produce a benign
IceChunk `ConflictError`: one commit wins, Step Functions retries the loser with
exponential backoff.

---

## Output

All intermediate per-dataset outputs land in `JOIN/outputs/`. The `stack-outputs` Lambda
merges them into a single flat 2D dataset:

| Variable | Source |
|---|---|
| `amsr2_snow_depth_mean` | AMSR2 (ascending + descending averaged) |
| `amsr2_snow_depth_uncertainty` | AMSR2 |
| `ceda_swe` | CEDA |
| `ceda_swe_std` | CEDA |
| `viirs_cgf_ndsi_snow_cover` | VIIRS |
| `icesat2_h_li` | ICESat-2 |

**Output file:** `JOIN/results/swe/swe_combined_{date}.nc`

All variables share the same 2D LIS lat/lon coordinates (`[north_south, east_west]`).
One file per date. No temporal stacking.

---

## Extensibility

Adding a new dataset requires:
1. Deploy a new dataset workflow (assembling from existing shared microservices where
   possible, adding new ones only if the compute method is genuinely novel)
2. Add the dataset name → SFN ARN mapping to the Orchestrator config (SSM parameter)
3. No changes to the Orchestrator Lambda, Collector Lambda, or any existing dataset workflow

---

## Open Questions

### 1. VIIRS weight merge step compute size

The `merge-viirs-weights` step reads 16 per-tile `.nc4` files from S3, calls
`merge_esmf_weights_2d()` (scipy sparse concatenation + index re-mapping), and uploads
the merged `.nc4`. Currently allocated as a Lambda.

**Decision needed:** Is the sparse concatenation over 16 small NC4 files sufficiently
memory-light for Lambda, or does it warrant a Batch/Fargate job?

### 2. AMSR2 icechunk-writer fan-out scale

With a large date range (~90 dates), the Map state launches ~90 concurrent
`icechunk-writer` Lambdas committing to the same IceChunk store. The preallocated design
makes this safe but retry cascades under high concurrency could become slow.

- Set `MaxConcurrency` on the Map state (e.g. 10–20) to throttle
- Accept full parallelism with exponential backoff retry

**Decision needed:** What is the expected typical date range per API call?

### 3. CEDA weight gen compute size

ESMF weight generation over the CEDA equirectangular grid — Lambda (up to 10 GB, 15 min)
may be sufficient. Needs profiling to confirm before deciding Lambda vs Batch.

### 4. ICESat-2 regrid compute size

Single-date point cloud over the LIS domain is likely Lambda-sized. Larger date ranges
may push to Batch. Needs profiling.

### 5. Intermediate output cleanup

Should `stack-outputs` delete per-dataset `.nc` files from `JOIN/outputs/` after merging,
or retain them for debugging and potential reuse?

### 6. API response pattern

Caller receives `execution_id` immediately. Status endpoint options:
- Poll `GET /status/{execution_id}` → reads DynamoDB record
- Webhook: caller provides a callback URL in the request body, Collector POSTs on completion

**Decision needed:** Does the caller expect polling or webhook callback?

### 7. CEDA credential storage and refresh

CEDA account credentials (used for HTTPS access to the CEDA Archive) need to be stored
somewhere the `icechunk-writer` and `sparse-regrid` Lambdas/Batch jobs can retrieve them
at both write time and read time — likely Secrets Manager, parallel to how VIIRS
temporary S3 credentials are obtained live from NSIDC's `/s3credentials` endpoint.
Unlike VIIRS's 1-hour STS tokens, CEDA login sessions are longer-lived, but the
credential-retrieval path still needs to be added to `make_config()` (or a
dataset-specific credential resolver) for the CEDA strategy.

**Decision needed:** Store a CEDA service-account login in Secrets Manager and inject it
per-request, or use a longer-lived session token refreshed on a schedule?

### 8. VIIRS temporary credential refresh mid-pipeline

NSIDC's temporary S3 credentials expire after 1 hour. For large VIIRS date ranges, the
per-tile `icechunk-writer` Map state, the 16-way `esmf-weight-gen` Map state, and the
final `sparse-regrid` read could collectively exceed that window.

**Decision needed:** Refresh credentials at the start of each Step Functions state that
touches VIIRS S3 data (safest, more API calls to NSIDC), or pass a single credential set
through the whole `viirs-pipeline` execution and accept a hard 1-hour ceiling on total
VIIRS pipeline duration per date?
