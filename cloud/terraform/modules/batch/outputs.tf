output "job_queue_arn" {
  description = "ARN of the shared Batch job queue."
  value       = aws_batch_job_queue.main.arn
}

output "job_queue_name" {
  description = "Name of the shared Batch job queue."
  value       = aws_batch_job_queue.main.name
}

output "regrid_job_definition_arn" {
  description = "Unversioned ARN of the ESMF regrid job definition (always resolves to latest active revision)."
  value       = "arn:aws:batch:${var.aws_region}:${data.aws_caller_identity.current.account_id}:job-definition/${aws_batch_job_definition.regrid.name}"
}

output "regrid_job_definition_name" {
  description = "Name of the ESMF regrid job definition."
  value       = aws_batch_job_definition.regrid.name
}

output "icechunk_populate_job_definition_arn" {
  description = "Unversioned ARN of the IceChunk populate-date job definition (always resolves to latest active revision)."
  value       = "arn:aws:batch:${var.aws_region}:${data.aws_caller_identity.current.account_id}:job-definition/${aws_batch_job_definition.icechunk_populate.name}"
}

output "icechunk_populate_job_definition_name" {
  description = "Name of the IceChunk populate-date job definition."
  value       = aws_batch_job_definition.icechunk_populate.name
}

output "regrid_log_group_name" {
  description = "CloudWatch log group for regrid Batch jobs."
  value       = aws_cloudwatch_log_group.regrid.name
}

output "icechunk_log_group_name" {
  description = "CloudWatch log group for IceChunk Batch jobs."
  value       = aws_cloudwatch_log_group.icechunk.name
}
