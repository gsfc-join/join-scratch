# JOIN Cloud Pipeline — Cloud Architecture

## Overview

The JOIN cloud pipeline is an asynchronous, event-driven data processing system deployed on AWS.
Users (or automated pipelines) submit a job via a REST API specifying a workflow type, spatial
extent, and temporal extent. The system returns a `job_id` immediately and processes the request
in the background using a set of AWS Step Functions, Lambda functions, and AWS Batch jobs. When
processing completes, a `.tar.gz` output bundle is written to S3 and a presigned URL is made
available through the job status endpoint.

---

## API

Two HTTP endpoints are exposed via AWS HTTP API Gateway.

### `POST /jobs` — Submit a job

**Request body:**

```json
{
  "workflow": "swe",
  "spatial_extent": {
    "lat_min": 30.0,
    "lat_max": 50.0,
    "lon_min": -110.0,
    "lon_max": -90.0
  },
  "temporal_extent": {
    "start": "20230101",
    "end":   "20230131"
  },
  "params": {}
}
```

| Field | Required | Description |
|---|---|---|
| `workflow` | Yes | Workflow identifier. Must match a configured workflow (e.g. `swe`, `iwp`, `precipitation`). |
| `spatial_extent` | Yes | Bounding box in decimal degrees. |
| `temporal_extent.start` | Yes | Start date `YYYYMMDD`. |
| `temporal_extent.end` | Yes | End date `YYYYMMDD` (inclusive). |
| `params` | No | Workflow-specific overrides (e.g. interpolation method, force flags). |

**Response `202 Accepted`:**

```json
{ "job_id": "a1b2c3d4-..." }
```

---

### `GET /jobs/{job_id}` — Poll job status

**Response — in progress:**

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "RUNNING",
  "workflow": "swe",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:05:00Z"
}
```

**Response — succeeded:**

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "SUCCEEDED",
  "workflow": "swe",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:45:00Z",
  "output_url": "https://s3.amazonaws.com/...",
  "output_expires_at": "2025-01-08T00:45:00Z"
}
```

**Response — failed:**

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "FAILED",
  "error": "GenerateWeights: ESMF exit code 1 ..."
}
```

Status values: `PENDING | RUNNING | SUCCEEDED | FAILED`

---

## Components

### HTTP API Gateway

AWS HTTP API (not REST API — simpler, cheaper, sufficient for pipeline integration use). Routes:

| Method | Route | Integration |
|---|---|---|
| `POST` | `/jobs` | Dispatch Lambda |
| `GET` | `/jobs/{job_id}` | Job Status Lambda |

No authentication is configured in the initial deployment. IAM SigV4 or API key auth will be
added in a future iteration (see [Open / Future Work](#open--future-work)).

---

### Dispatch Lambda

Triggered by `POST /jobs`. Responsibilities:

1. Validate the request (workflow must exist, all extent fields must be present).
2. Generate a UUID4 `job_id`.
3. Write an initial record to DynamoDB (`status=PENDING`).
4. Look up the list of child Step Function ARNs for the requested workflow from Lambda environment
   variables (injected by Terraform at deploy time — see [Workflow Definitions](#workflow-definitions)).
5. Start each child Step Function execution, passing `job_id`, `branch_name`, `spatial_extent`,
   `temporal_extent`, and `params` as the execution input.
6. Update the DynamoDB record to `status=RUNNING` and write `expected_children` (the count of
   child SFs started).
7. Return `{ "job_id": "..." }` with HTTP 202.

---

### Job Status Lambda

Triggered by `GET /jobs/{job_id}`. Queries the DynamoDB `jobs` table by `job_id` and returns the
current record. If `status=SUCCEEDED` and the presigned URL has expired (past `output_expires_at`),
the Lambda regenerates a fresh presigned URL from the still-existing S3 object (as long as the
object has not been removed by the bucket lifecycle policy).

---

### DynamoDB — `jobs` Table

| Attribute | Type | Description |
|---|---|---|
| `job_id` | String (PK) | UUID4 |
| `status` | String | `PENDING \| RUNNING \| SUCCEEDED \| FAILED` |
| `workflow` | String | Workflow identifier (e.g. `swe`) |
| `expected_children` | Number | Number of child Step Functions started for this job |
| `completed_children` | Number | Atomically incremented by the ReportCompletion Lambda |
| `child_outputs` | Map | Keyed by `branch_name`; values are S3 URIs of per-branch outputs |
| `output_s3_key` | String | S3 key of the final output tarball (set on `SUCCEEDED`) |
| `output_url` | String | 7-day presigned URL (set on `SUCCEEDED`) |
| `output_expires_at` | String | ISO 8601 expiry of the presigned URL |
| `error` | String | Error message (set on `FAILED`) |
| `created_at` | String | ISO 8601 timestamp |
| `updated_at` | String | ISO 8601 timestamp |

---

### Child Step Functions

One Standard Step Function per **data source group**. A data source group is a named collection
of source data that is processed through a shared chain of steps. Multiple groups run in parallel
for a single job (e.g. a `swe` workflow starts `swe_data_products` and `swe_validation`
simultaneously). The number of groups per workflow is variable and defined in Terraform.

Each SF receives as execution input: `job_id`, `branch_name`, `spatial_extent`,
`temporal_extent`, and any workflow-specific `params`.

#### Available Steps

Not all steps are required by every Step Function. The set of steps included is defined per
branch in Terraform.

| Step | Compute | Required | Description |
|---|---|---|---|
| EnumerateDates | Lambda | Always | Expands `start`/`end` into a list of `YYYYMMDD` strings for `Map` fan-out. |
| Access & Virtualize | Lambda + Batch/Fargate Spot | Always | Creates/updates the IceChunk virtual store (`InitStore` Lambda, idempotent) and populates virtual chunk references (`PopulateRefs` Batch). Also checks for and generates ESMF regrid weight files (`CheckWeights` Lambda, `GenerateWeights` Batch). |
| Collocate | Lambda | Optional | Spatially subsets or co-locates data to the requested extent. |
| Regrid | Lambda + Batch/Fargate Spot | Optional | Applies the ESMF sparse weight matrix to interpolate to the destination grid. |
| Mask | Lambda | Optional | Applies quality flags or land/sea/cloud masks. |
| ReportCompletion | Lambda | Always | Final step. Writes branch outputs to DynamoDB, atomically increments `completed_children`, and invokes the Packaging Lambda if all children are done. |

#### State Machine Structure

```
EnumerateDates (Lambda)
  → Parallel:
      Branch A — IceChunk store:
        InitStore (Lambda, idempotent)
        → Map(dates) → PopulateRefs (Batch/Fargate Spot)
      Branch B — ESMF weights:
        CheckWeightsCache (Lambda)
        → [if miss] GenerateWeights (Batch/Fargate Spot)
  → [optional] Collocate (Lambda)
  → [optional] Regrid (Lambda)
  → [optional] Mask (Lambda)
  → ReportCompletion (Lambda)

[Catch *] → ReportFailure (Lambda)
```

**Error handling:** A `Catch` on all states calls the **ReportFailure Lambda**, which sets
`status=FAILED` and writes the error message to DynamoDB.

---

### ReportCompletion Lambda

The final step of every child Step Function. Uses a DynamoDB conditional update (`ADD
completed_children 1`) to atomically increment the counter. After the update, reads the new
value back and compares to `expected_children`. If equal, asynchronously invokes the Packaging
Lambda with the `job_id`. This avoids a master orchestrator Step Function or EventBridge fan-in:
the last branch to finish naturally triggers packaging.

---

### Packaging Lambda

Invoked asynchronously by the ReportCompletion Lambda when all children are done.
Responsibilities:

1. Read all `child_outputs` from DynamoDB for the `job_id`.
2. Stream each output file from S3.
3. Build a `.tar.gz` bundle containing all outputs and a `manifest.json`.
4. Write the bundle to `s3://<outputs-bucket>/jobs/<job_id>/output.tar.gz`.
5. Generate a presigned URL with a 7-day TTL.
6. Update DynamoDB: `status=SUCCEEDED`, `output_s3_key`, `output_url`, `output_expires_at`,
   `updated_at`.

#### `manifest.json` Schema

```json
{
  "job_id": "a1b2c3d4-...",
  "workflow": "swe",
  "created_at": "2025-01-01T00:00:00Z",
  "completed_at": "2025-01-01T00:45:00Z",
  "spatial_extent": { "lat_min": 30, "lat_max": 50, "lon_min": -110, "lon_max": -90 },
  "temporal_extent": { "start": "20230101", "end": "20230131" },
  "files": [
    {
      "filename": "swe_data_products.nc",
      "branch": "swe_data_products",
      "description": "Regridded SWE data product"
    },
    {
      "filename": "swe_validation.nc",
      "branch": "swe_validation",
      "description": "SWE validation collocated output"
    }
  ]
}
```

---

### S3 Outputs Bucket

A dedicated Terraform-managed S3 bucket for job output tarballs.

| Property | Value |
|---|---|
| Path pattern | `jobs/<job_id>/output.tar.gz` |
| Access | Private; all access via presigned URLs only |
| Versioning | Disabled (each `job_id` is unique) |
| Lifecycle policy | Configurable retention period via Terraform variable. Objects are deleted after the retention window, invalidating the presigned URL. The Job Status Lambda can regenerate a presigned URL while the object still exists. |

The lifecycle retention period should be set longer than the presigned URL TTL (7 days) to allow
URL regeneration via the status endpoint after a URL expires but before the object is deleted.

---

## Workflow Definitions

A workflow is a named configuration that maps to one or more child Step Functions (branches).
Each branch has a `branch_name` and a defined set of steps. The mapping is defined in Terraform
and injected into the Dispatch Lambda as environment variables at deploy time.

**Environment variable format:**

The Dispatch Lambda reads `SF_ARNS_<WORKFLOW>` (uppercased workflow name), which contains a
JSON array of `{ "branch_name": "...", "arn": "..." }` objects. Adding a new workflow requires
a `terraform apply` but no Lambda code change.

**Example — `swe` workflow (2 branches):**

```json
[
  { "branch_name": "swe_data_products", "arn": "arn:aws:states:...:stateMachine:swe-data-products" },
  { "branch_name": "swe_validation",    "arn": "arn:aws:states:...:stateMachine:swe-validation" }
]
```

**Example — `iwp` workflow (3 branches):**

```json
[
  { "branch_name": "iwp_inputs",       "arn": "arn:aws:states:...:stateMachine:iwp-inputs" },
  { "branch_name": "iwp_validation_a", "arn": "arn:aws:states:...:stateMachine:iwp-validation-a" },
  { "branch_name": "iwp_validation_b", "arn": "arn:aws:states:...:stateMachine:iwp-validation-b" }
]
```

The `expected_children` value written to DynamoDB at dispatch time is the length of this array.

---

## Job Lifecycle

1. Client sends `POST /jobs` with `workflow`, `spatial_extent`, `temporal_extent`.
2. Dispatch Lambda validates, creates `job_id`, writes `status=PENDING` to DynamoDB.
3. Dispatch Lambda reads SF ARNs from env vars, starts N child Step Functions.
4. Dispatch Lambda writes `status=RUNNING`, `expected_children=N` to DynamoDB and returns
   `{ job_id }` with HTTP 202.
5. Each child Step Function runs its steps independently and in parallel with the other branches.
6. Each child SF ends with ReportCompletion Lambda:
   - Writes branch outputs (S3 URIs) to `child_outputs` in DynamoDB.
   - Atomically increments `completed_children`.
   - If `completed_children == expected_children`: invokes Packaging Lambda.
7. Packaging Lambda builds the tarball, writes to S3, generates presigned URL, updates DynamoDB
   to `status=SUCCEEDED`.
8. Client polling `GET /jobs/{job_id}` eventually receives `status=SUCCEEDED` with `output_url`.
9. Client downloads the tarball from the presigned URL and reads `manifest.json`.

---

## Terraform Module Layout

```
cloud/terraform/
├── main.tf
├── variables.tf
├── outputs.tf
└── modules/
    ├── ecr/                       existing — ECR repositories
    ├── iam/                       existing + new roles for dispatch, status,
    │                                reporter, packaging lambdas, and API Gateway
    ├── batch/                     existing — Batch compute environment and job definitions
    ├── dynamodb/                  NEW — jobs table
    ├── api_gateway/               NEW — HTTP API, routes, Lambda integrations
    ├── lambda/                    existing + dispatch, status, reporter, packaging
    ├── s3_outputs/                NEW — dedicated outputs bucket + lifecycle configuration
    └── stepfunctions/
        ├── workflow_pipeline/     NEW — shared module, parameterized by source/steps
        ├── swe/
        │   ├── data_products/     refactored from existing amsr2 SF
        │   └── validation/        NEW
        ├── iwp/
        │   ├── data_products/     NEW
        │   └── validation/        NEW
        └── precipitation/
            ├── data_products/     NEW
            └── validation/        NEW
```

The `workflow_pipeline` shared module is instantiated by each branch module. It accepts the
source name, the set of optional steps to include (boolean flags), and the relevant Batch job
definition and Lambda ARNs. This avoids duplicating the Step Function state machine pattern
across sources.

---

## Adding a New Workflow

1. Add data source modules as needed — see `ARCHITECTURE.md` (*Adding a new data source*) for
   the batch and Lambda source interface.
2. Create a new directory under `cloud/terraform/modules/stepfunctions/<workflow>/` for each
   branch.
3. Instantiate the `workflow_pipeline` Terraform module for each branch, passing the appropriate
   step inclusion flags and ARNs.
4. Add a `SF_ARNS_<WORKFLOW>` environment variable to the Dispatch Lambda Terraform resource,
   populated from the new Step Function module outputs.
5. Run `terraform apply`.
6. No Lambda code changes required — the Dispatch Lambda reads its workflow registry from
   environment variables at runtime.

---

## Open / Future Work

| Item | Notes |
|---|---|
| API authentication | Add IAM SigV4 or API key auth to API Gateway. Deferred from initial deployment. |
| S3 lifecycle retention period | Choose a retention window longer than the 7-day presigned URL TTL. Expose as a Terraform variable. |
| Collocate implementation | Not yet implemented. Will be a Lambda function; implementation is per data source. |
| Mask implementation | Not yet implemented. Will be a Lambda function; implementation is per data source. |
| Push notification | Currently pure polling. SNS/SQS/EventBridge notification could be added for pipeline consumers that prefer push-based delivery. |
