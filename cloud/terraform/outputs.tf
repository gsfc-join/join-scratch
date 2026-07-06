# ── ECR ────────────────────────────────────────────────────────────────────────

output "regrid_ecr_repository_url" {
  description = "ECR repository URL for the ESMF regrid image. Pass to build_and_push.sh via ECR_REPO_URI."
  value       = module.ecr.regrid_repository_url
}

output "icechunk_ecr_repository_url" {
  description = "ECR repository URL for the IceChunk image. Pass to batch/icechunk/build_and_push.sh via ECR_REPO_URI."
  value       = module.ecr.icechunk_repository_url
}

output "init_store_lambda_ecr_repository_url" {
  description = "ECR repository URL for the init-store Lambda image. Pass to cloud/lambda/init_store/build_and_push.sh via ECR_REPO_URI."
  value       = module.ecr.init_store_lambda_repository_url
}

output "init_store_lambda_function_name" {
  description = "Name of the init-store Lambda function."
  value       = module.lambda.init_store_function_name
}

# ── Batch ──────────────────────────────────────────────────────────────────────

output "batch_job_queue_name" {
  description = "Name of the shared AWS Batch job queue."
  value       = module.batch.job_queue_name
}

output "batch_job_queue_arn" {
  description = "ARN of the shared AWS Batch job queue."
  value       = module.batch.job_queue_arn
}

output "regrid_job_definition_name" {
  description = "Name of the ESMF regrid Batch job definition."
  value       = module.batch.regrid_job_definition_name
}

output "icechunk_populate_job_definition_name" {
  description = "Name of the IceChunk populate-date Batch job definition."
  value       = module.batch.icechunk_populate_job_definition_name
}

# ── Step Functions ─────────────────────────────────────────────────────────────

output "state_machine_arn" {
  description = "ARN of the AMSR2 pipeline Step Functions state machine."
  value       = module.stepfunctions.state_machine_arn
}

output "state_machine_name" {
  description = "Name of the AMSR2 pipeline Step Functions state machine."
  value       = module.stepfunctions.state_machine_name
}

# ── Example execution input ────────────────────────────────────────────────────

output "example_execution_input" {
  description = "Example JSON input for a Step Functions execution (single date)."
  value = jsonencode({
    start_date       = "20230115"
    end_date         = "20230115"
    store_uri        = var.default_store_uri
    weights_uri      = var.default_weights_uri
    lis_path         = var.default_lis_path
    method           = "bilinear"
    recreate_store   = false
    force_repopulate = false
    force_regenerate = false
    src_grid_spec    = {}
  })
}

output "start_execution_example" {
  description = "CLI command to start a pipeline execution for a date range."
  value       = <<-EOT
    aws stepfunctions start-execution \
      --state-machine-arn ${module.stepfunctions.state_machine_arn} \
      --name "amsr2-$(date +%Y%m%d-%H%M%S)" \
      --input '{
        "start_date": "20230115",
        "end_date":   "20230115",
        "store_uri":  "${var.default_store_uri}",
        "weights_uri": "${var.default_weights_uri}",
        "lis_path":   "${var.default_lis_path}",
        "method":     "bilinear",
        "recreate_store":   false,
        "force_repopulate": false,
        "force_regenerate": false,
        "src_grid_spec":    {}
      }'
  EOT
}
