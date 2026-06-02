variable "aws_region" {
  type        = string
  description = "AWS region"
  default     = "us-west-2"
}

variable "vpc_id" {
  type        = string
  description = "VPC ID for Batch compute environment"
  default     = "vpc-037d001e0edc35ee7"  # default VPC in us-west-2
}

variable "source_s3_bucket" {
  type        = string
  description = "S3 bucket containing source EarthCARE and GPM-DPR granule files"
  default     = "airborne-smce-prod-user-bucket"
}

variable "store_s3_bucket" {
  type        = string
  description = "S3 bucket where IceChunk stores are written (may be the same as source_s3_bucket)"
  default     = "airborne-smce-prod-user-bucket"
}

variable "instance_types" {
  type        = list(string)
  description = "EC2 instance types for the Batch Spot compute environment"
  default     = ["m7i.4xlarge"]
}

variable "max_vcpus" {
  type        = number
  description = "Maximum vCPUs in the Batch compute environment"
  default     = 64
}

variable "image_tag" {
  type        = string
  description = "Container image tag to use in job definitions"
  default     = "latest"
}
