output "regrid_repository_url" {
  description = "ECR repository URL for the ESMF regrid image."
  value       = aws_ecr_repository.regrid.repository_url
}

output "regrid_repository_arn" {
  description = "ECR repository ARN for the ESMF regrid image."
  value       = aws_ecr_repository.regrid.arn
}

output "icechunk_repository_url" {
  description = "ECR repository URL for the IceChunk init/populate image."
  value       = aws_ecr_repository.icechunk.repository_url
}

output "icechunk_repository_arn" {
  description = "ECR repository ARN for the IceChunk init/populate image."
  value       = aws_ecr_repository.icechunk.arn
}

output "init_store_lambda_repository_url" {
  description = "ECR repository URL for the init-store Lambda container image."
  value       = aws_ecr_repository.init_store_lambda.repository_url
}

output "init_store_lambda_repository_arn" {
  description = "ECR repository ARN for the init-store Lambda container image."
  value       = aws_ecr_repository.init_store_lambda.arn
}
