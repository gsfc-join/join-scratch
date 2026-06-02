output "ecr_repository_url" {
  description = "URL of the join-virtualize ECR repository"
  value       = aws_ecr_repository.virtualize.repository_url
}

output "ecr_registry" {
  description = "ECR registry (account.dkr.ecr.region.amazonaws.com)"
  value       = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
}

output "batch_job_queue_arn" {
  description = "ARN of the join-virtualize Batch job queue"
  value       = aws_batch_job_queue.virtualize.arn
}

output "job_definition_earthcare_arn" {
  description = "ARN of the EarthCARE Batch job definition"
  value       = aws_batch_job_definition.earthcare.arn
}

output "job_definition_gpmdpr_arn" {
  description = "ARN of the GPM-DPR Batch job definition"
  value       = aws_batch_job_definition.gpmdpr.arn
}

output "job_role_arn" {
  description = "ARN of the IAM role assumed by Batch job containers"
  value       = aws_iam_role.job.arn
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for Batch job output"
  value       = aws_cloudwatch_log_group.batch.name
}
