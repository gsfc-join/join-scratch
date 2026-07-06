variable "prefix" {
  description = "Name prefix for Step Functions resources."
  type        = string
  default     = "join"
}

variable "sfn_role_arn" {
  description = "IAM role ARN for the Step Functions state machine."
  type        = string
}

# ── Lambda ARNs ────────────────────────────────────────────────────────────────
variable "enumerate_dates_arn" {
  description = "ARN of the enumerate-dates Lambda function."
  type        = string
}

variable "check_weights_arn" {
  description = "ARN of the check-weights Lambda function."
  type        = string
}

variable "apply_regrid_arn" {
  description = "ARN of the apply-regrid Lambda function."
  type        = string
}

variable "init_store_arn" {
  description = "ARN of the init-store Lambda function."
  type        = string
}

# ── Batch resource ARNs ────────────────────────────────────────────────────────
variable "batch_job_queue_arn" {
  description = "ARN of the shared Batch job queue."
  type        = string
}

variable "icechunk_populate_job_definition_arn" {
  description = "ARN of the IceChunk populate-date Batch job definition."
  type        = string
}

variable "regrid_job_definition_arn" {
  description = "ARN of the ESMF regrid Batch job definition."
  type        = string
}
