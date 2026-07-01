output "enumerate_dates_arn" {
  description = "ARN of the enumerate-dates Lambda function."
  value       = aws_lambda_function.enumerate_dates.arn
}

output "check_weights_arn" {
  description = "ARN of the check-weights Lambda function."
  value       = aws_lambda_function.check_weights.arn
}

output "apply_regrid_arn" {
  description = "ARN of the apply-regrid Lambda function (placeholder)."
  value       = aws_lambda_function.apply_regrid.arn
}

output "init_store_arn" {
  description = "ARN of the init-store Lambda function."
  value       = aws_lambda_function.init_store.arn
}

output "init_store_function_name" {
  description = "Name of the init-store Lambda function."
  value       = aws_lambda_function.init_store.function_name
}

output "all_function_arns" {
  description = "List of all Lambda function ARNs (used to grant Step Functions invoke permission)."
  value = [
    aws_lambda_function.enumerate_dates.arn,
    aws_lambda_function.check_weights.arn,
    aws_lambda_function.apply_regrid.arn,
    aws_lambda_function.init_store.arn,
  ]
}
