variable "aws_region" {
  type        = string
  description = "AWS region to deploy into"
  default     = "us-west-2"
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type"
  default     = "m7i.2xlarge"
}

variable "ami_id" {
  type        = string
  description = "AMI ID for the instance. Leave empty to use the latest Amazon Linux 2023 AMI."
  default     = ""
}

variable "s3_access" {
  type        = string
  description = "S3 access level: 'full' = full read/write/list to all buckets, 'readonly' = read-only, 'none' = no S3 access"
  default     = "full"

  validation {
    condition     = contains(["full", "readonly", "none"], var.s3_access)
    error_message = "s3_access must be one of: full, readonly, none."
  }
}

variable "vpc_id" {
  type        = string
  description = "VPC ID to deploy into. Leave empty to use the default VPC."
  default     = ""
}

variable "subnet_id" {
  type        = string
  description = "Subnet ID to deploy into. Leave empty to use a default public subnet (random AZ)."
  default     = ""
}

variable "availability_zone" {
  type        = string
  description = "Availability zone for the instance. Leave empty to select at random."
  default     = ""
}

variable "root_volume_size" {
  type        = number
  description = "Root volume size in GB"
  default     = 100
}

variable "root_volume_type" {
  type        = string
  description = "Root volume type (gp3, gp2, io1, io2, etc.)"
  default     = "gp3"
}

variable "instance_state" {
  type        = string
  description = "Desired instance state: 'running' or 'stopped'"
  default     = "running"

  validation {
    condition     = contains(["running", "stopped"], var.instance_state)
    error_message = "instance_state must be either 'running' or 'stopped'."
  }
}

variable "instance_name" {
  type        = string
  description = "Name tag for the EC2 instance"
  default     = "sandbox"
}
