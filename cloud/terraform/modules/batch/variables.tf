variable "prefix" {
  description = "Name prefix for all Batch resources."
  type        = string
  default     = "join"
}

variable "aws_region" {
  description = "AWS region."
  type        = string
  default     = "us-west-2"
}

variable "subnet_ids" {
  description = "Subnet IDs for the Fargate compute environment."
  type        = list(string)
}

variable "security_group_ids" {
  description = "Security group IDs for the Fargate compute environment."
  type        = list(string)
}

variable "max_vcpus" {
  description = "Maximum vCPUs for the shared Fargate compute environment."
  type        = number
  default     = 64
}

# ── ECR image URLs ─────────────────────────────────────────────────────────────
variable "regrid_ecr_url" {
  description = "ECR repository URL for the ESMF regrid image (no tag)."
  type        = string
}

variable "icechunk_ecr_url" {
  description = "ECR repository URL for the IceChunk image (no tag)."
  type        = string
}

# ── IAM role ARNs (from iam module) ───────────────────────────────────────────
variable "execution_role_arn" {
  description = "ECS task execution role ARN (ECR pull + CloudWatch logs)."
  type        = string
}

variable "regrid_task_role_arn" {
  description = "ECS task role ARN for the regrid job (S3 read/write weights)."
  type        = string
}

variable "icechunk_task_role_arn" {
  description = "ECS task role ARN for IceChunk jobs (S3 read/write store)."
  type        = string
}

variable "batch_service_role_arn" {
  description = "AWS Batch service role ARN."
  type        = string
}

# ── Resource sizing ───────────────────────────────────────────────────────────
variable "regrid_vcpus" {
  description = "vCPUs for the ESMF regrid job."
  type        = number
  default     = 4
}

variable "regrid_memory_mb" {
  description = "Memory (MB) for the ESMF regrid job."
  type        = number
  default     = 16384
}

variable "icechunk_vcpus" {
  description = "vCPUs for IceChunk init/populate jobs."
  type        = number
  default     = 2
}

variable "icechunk_memory_mb" {
  description = "Memory (MB) for IceChunk init/populate jobs."
  type        = number
  default     = 8192
}

# ── Default parameter values ──────────────────────────────────────────────────
variable "default_lis_path" {
  description = "Default S3 path to the LIS NetCDF file."
  type        = string
  default     = "s3://airborne-smce-prod-user-bucket/JOIN/lis_input_NMP_1000m_missouri.nc"
}

variable "default_weights_uri" {
  description = "Default S3 URI for the cached ESMF weights file."
  type        = string
  default     = "s3://airborne-smce-prod-user-bucket/JOIN/cached-weights/GCOM-W1-AMSR2-L3-SND/lis-1km-missouri.nc4"
}

variable "default_extra_args" {
  description = "Default extra args for ESMF_RegridWeightGen."
  type        = string
  default     = "--no_log --ignore_unmapped"
}

variable "default_store_uri" {
  description = "Documentation hint: example S3 URI for an IceChunk store (AMSR2). Always provided by the execution input at runtime."
  type        = string
  default     = "s3://airborne-smce-prod-user-bucket/JOIN/icechunk-stores/GCOM-W1-AMSR2-L3-SND"
}

variable "default_source" {
  description = "Documentation hint: example source identifier. SOURCE must always be provided by the execution input at runtime."
  type        = string
  default     = ""
}
