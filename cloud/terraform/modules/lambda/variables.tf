variable "prefix" {
  description = "Name prefix for Lambda function names."
  type        = string
  default     = "join"
}

variable "lambda_role_arn" {
  description = "IAM role ARN for all Lambda functions."
  type        = string
}

variable "repo_root" {
  description = "Absolute path to the repository root (for source_file paths)."
  type        = string
}

variable "init_store_image_uri" {
  description = "ECR image URI for the init-store Lambda container (e.g. 123456.dkr.ecr.us-west-2.amazonaws.com/join-init-store-lambda:latest)."
  type        = string
}
