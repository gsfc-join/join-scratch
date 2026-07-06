output "execution_role_arn" {
  description = "ARN of the ECS task execution role (shared by all Batch job defs)."
  value       = aws_iam_role.execution.arn
}

output "regrid_task_role_arn" {
  description = "ARN of the ECS task role for the ESMF regrid Batch job."
  value       = aws_iam_role.regrid_task.arn
}

output "icechunk_task_role_arn" {
  description = "ARN of the ECS task role for IceChunk Batch jobs."
  value       = aws_iam_role.icechunk_task.arn
}

output "batch_service_role_arn" {
  description = "ARN of the AWS Batch service role."
  value       = aws_iam_role.batch_service.arn
}

output "lambda_role_arn" {
  description = "ARN of the shared Lambda execution role."
  value       = aws_iam_role.lambda.arn
}

output "sfn_role_arn" {
  description = "ARN of the Step Functions execution role."
  value       = aws_iam_role.sfn.arn
}
