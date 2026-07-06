variable "prefix" {
  description = "Name prefix applied to all IAM role names."
  type        = string
  default     = "join"
}

variable "s3_data_bucket" {
  description = "S3 bucket that holds LIS data, weights, and the IceChunk store."
  type        = string
  default     = "airborne-smce-prod-user-bucket"
}

variable "weights_prefix" {
  description = "S3 key prefix under s3_data_bucket where weights files are written."
  type        = string
  default     = "JOIN/cached-weights/"
}

variable "icechunk_store_prefix" {
  description = "S3 key prefix under s3_data_bucket for the IceChunk store root."
  type        = string
  default     = "JOIN/icechunk-stores/"
}

variable "lambda_function_arns" {
  description = "List of Lambda function ARNs that the Step Functions role may invoke."
  type        = list(string)
  default     = []
}
