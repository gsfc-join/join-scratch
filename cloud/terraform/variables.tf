variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-west-2"
}

variable "project" {
  description = "Project name prefix (e.g. 'join')."
  type        = string
  default     = "join"
}

variable "name" {
  description = "Stack name suffix (e.g. 'amsr2')."
  type        = string
  default     = "amsr2"
}

variable "tags" {
  description = "Default tags applied to all AWS resources."
  type        = map(string)
  default = {
    Project     = "JOIN"
    ManagedBy   = "terraform"
  }
}

# ── Networking ────────────────────────────────────────────────────────────────
variable "subnet_ids" {
  description = "Subnet IDs for the Fargate compute environment."
  type        = list(string)
  default     = [
    "subnet-0accdb3722561992d",
    "subnet-004cbadac59fb684b",
    "subnet-08163938d20e0b6df",
    "subnet-0f016de5719efbe32",
  ]
}

variable "security_group_ids" {
  description = "Security group IDs for the Fargate compute environment."
  type        = list(string)
  default     = ["sg-02cefe7bb9b397494"]
}

# ── S3 ─────────────────────────────────────────────────────────────────────────
variable "s3_data_bucket" {
  description = "S3 bucket holding LIS data, weights, and the IceChunk store."
  type        = string
  default     = "airborne-smce-prod-user-bucket"
}

variable "weights_prefix" {
  description = "S3 key prefix for ESMF weights files within s3_data_bucket."
  type        = string
  default     = "JOIN/cached-weights/"
}

variable "icechunk_store_prefix" {
  description = "S3 key prefix for the IceChunk store root within s3_data_bucket."
  type        = string
  default     = "JOIN/icechunk-stores/"
}

# ── Resource sizing ────────────────────────────────────────────────────────────
variable "max_vcpus" {
  description = "Maximum vCPUs for the shared Fargate compute environment."
  type        = number
  default     = 64
}

variable "regrid_vcpus" {
  description = "vCPUs for the ESMF regrid Batch job."
  type        = number
  default     = 4
}

variable "regrid_memory_mb" {
  description = "Memory (MB) for the ESMF regrid Batch job."
  type        = number
  default     = 16384
}

variable "icechunk_vcpus" {
  description = "vCPUs for IceChunk init/populate Batch jobs."
  type        = number
  default     = 2
}

variable "icechunk_memory_mb" {
  description = "Memory (MB) for IceChunk init/populate Batch jobs."
  type        = number
  default     = 8192
}

# ── Default job parameters ────────────────────────────────────────────────────
variable "default_lis_path" {
  description = "Default S3 path to the LIS NetCDF file."
  type        = string
  default     = "s3://airborne-smce-prod-user-bucket/JOIN/lis_input_NMP_1000m_missouri.nc"
}

variable "default_weights_uri" {
  description = "Default S3 URI for the cached ESMF weights."
  type        = string
  default     = "s3://airborne-smce-prod-user-bucket/JOIN/cached-weights/GCOM-W1-AMSR2-L3-SND/lis-1km-missouri.nc4"
}

variable "default_extra_args" {
  description = "Default extra args for ESMF_RegridWeightGen."
  type        = string
  default     = "--no_log --ignore_unmapped"
}

variable "default_store_uri" {
  description = "Default S3 URI for the AMSR2 IceChunk store."
  type        = string
  default     = "s3://airborne-smce-prod-user-bucket/JOIN/icechunk-stores/GCOM-W1-AMSR2-L3-SND"
}

# ── Lambda packaging ──────────────────────────────────────────────────────────
variable "repo_root" {
  description = "Absolute path to the repository root. Used to locate cloud/lambda/ source files."
  type        = string
  default     = "/home/edlang/join-scratch"
}
