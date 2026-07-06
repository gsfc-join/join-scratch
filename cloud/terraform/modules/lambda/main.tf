# Lambda functions are packaged from the lambda/ directory at deploy time.
# Terraform archives each handler directory into a zip and uploads to Lambda.

data "archive_file" "enumerate_dates" {
  type        = "zip"
  source_file = "${var.repo_root}/cloud/lambda/enumerate_dates/handler.py"
  output_path = "${path.module}/.builds/enumerate_dates.zip"
}

data "archive_file" "check_weights" {
  type        = "zip"
  source_file = "${var.repo_root}/cloud/lambda/check_weights/handler.py"
  output_path = "${path.module}/.builds/check_weights.zip"
}

data "archive_file" "apply_regrid" {
  type        = "zip"
  source_file = "${var.repo_root}/cloud/lambda/apply_regrid/handler.py"
  output_path = "${path.module}/.builds/apply_regrid.zip"
}

# ── enumerate_dates ────────────────────────────────────────────────────────────

resource "aws_lambda_function" "enumerate_dates" {
  function_name    = "${var.prefix}-enumerate-dates"
  description      = "Builds the list of YYYYMMDD dates for Step Functions fan-out."
  role             = var.lambda_role_arn
  runtime          = "python3.12"
  handler          = "handler.handler"
  filename         = data.archive_file.enumerate_dates.output_path
  source_code_hash = data.archive_file.enumerate_dates.output_base64sha256
  timeout          = 30
  memory_size      = 128

  environment {
    variables = {
      POWERTOOLS_SERVICE_NAME = "${var.prefix}-enumerate-dates"
    }
  }
}

resource "aws_cloudwatch_log_group" "enumerate_dates" {
  name              = "/aws/lambda/${aws_lambda_function.enumerate_dates.function_name}"
  retention_in_days = 14
}

# ── check_weights ──────────────────────────────────────────────────────────────

resource "aws_lambda_function" "check_weights" {
  function_name    = "${var.prefix}-check-weights"
  description      = "Checks whether an ESMF weights file already exists on S3."
  role             = var.lambda_role_arn
  runtime          = "python3.12"
  handler          = "handler.handler"
  filename         = data.archive_file.check_weights.output_path
  source_code_hash = data.archive_file.check_weights.output_base64sha256
  timeout          = 30
  memory_size      = 128
}

resource "aws_cloudwatch_log_group" "check_weights" {
  name              = "/aws/lambda/${aws_lambda_function.check_weights.function_name}"
  retention_in_days = 14
}

# ── apply_regrid (placeholder) ─────────────────────────────────────────────────

resource "aws_lambda_function" "apply_regrid" {
  function_name    = "${var.prefix}-apply-regrid"
  description      = "TODO: apply ESMF weights + write regridded output. Currently a placeholder."
  role             = var.lambda_role_arn
  runtime          = "python3.12"
  handler          = "handler.handler"
  filename         = data.archive_file.apply_regrid.output_path
  source_code_hash = data.archive_file.apply_regrid.output_base64sha256
  timeout          = 30
  memory_size      = 128
}

resource "aws_cloudwatch_log_group" "apply_regrid" {
  name              = "/aws/lambda/${aws_lambda_function.apply_regrid.function_name}"
  retention_in_days = 14
}

# ── init_store (container image) ───────────────────────────────────────────────

resource "aws_lambda_function" "init_store" {
  function_name = "${var.prefix}-init-store"
  description   = "Initialises the preallocated AMSR2 IceChunk Zarr v3 store on S3."
  role          = var.lambda_role_arn
  package_type  = "Image"
  image_uri     = var.init_store_image_uri
  timeout       = 300   # store creation typically ~3-8 s; 5 min is ample headroom
  memory_size   = 1024

  environment {
    variables = {
      RUST_LOG = "error"
    }
  }
}

resource "aws_cloudwatch_log_group" "init_store" {
  name              = "/aws/lambda/${aws_lambda_function.init_store.function_name}"
  retention_in_days = 14
}
