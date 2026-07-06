output "state_machine_arn" {
  description = "ARN of the AMSR2 pipeline Step Functions state machine."
  value       = aws_sfn_state_machine.amsr2_pipeline.arn
}

output "state_machine_name" {
  description = "Name of the AMSR2 pipeline Step Functions state machine."
  value       = aws_sfn_state_machine.amsr2_pipeline.name
}

output "log_group_name" {
  description = "CloudWatch log group for Step Functions execution history."
  value       = aws_cloudwatch_log_group.sfn.name
}
