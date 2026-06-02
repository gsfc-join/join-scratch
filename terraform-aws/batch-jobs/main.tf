terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # State stored locally — change backend block to use S3 for shared state.
  backend "local" {
    path = "terraform.tfstate"
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      PROJECT   = "JOIN"
      ManagedBy = "Terraform"
      Component = "iwp-virtualize"
    }
  }
}

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  id = var.vpc_id
}

data "aws_subnets" "batch" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

# ---------------------------------------------------------------------------
# ECR repository
# ---------------------------------------------------------------------------

resource "aws_ecr_repository" "virtualize" {
  name                 = "join-virtualize"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "join-virtualize"
  }
}

resource "aws_ecr_lifecycle_policy" "virtualize" {
  repository = aws_ecr_repository.virtualize.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# IAM — Batch service role
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "batch_service_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["batch.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_service" {
  name               = "join-virtualize-batch-service-role"
  assume_role_policy = data.aws_iam_policy_document.batch_service_assume.json
}

resource "aws_iam_role_policy_attachment" "batch_service_managed" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

# ---------------------------------------------------------------------------
# IAM — Job execution role (ECS task / container role)
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "job_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "job_execution" {
  name               = "join-virtualize-job-execution-role"
  assume_role_policy = data.aws_iam_policy_document.job_assume.json
}

# ECS task execution (pull ECR image, push logs to CloudWatch)
resource "aws_iam_role_policy_attachment" "job_execution_ecs" {
  role       = aws_iam_role.job_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ---------------------------------------------------------------------------
# IAM — Job role (what the container can do at runtime)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "job" {
  name               = "join-virtualize-job-role"
  assume_role_policy = data.aws_iam_policy_document.job_assume.json
}

# S3 access: read source data + read/write/delete icechunk store
data "aws_iam_policy_document" "job_s3" {
  statement {
    sid = "ReadSourceData"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
      "s3:GetBucketLocation",
    ]
    resources = [
      "arn:aws:s3:::${var.source_s3_bucket}",
      "arn:aws:s3:::${var.source_s3_bucket}/*",
    ]
  }

  statement {
    sid = "ReadWriteIceChunkStore"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:GetBucketLocation",
    ]
    resources = [
      "arn:aws:s3:::${var.store_s3_bucket}",
      "arn:aws:s3:::${var.store_s3_bucket}/*",
    ]
  }
}

resource "aws_iam_role_policy" "job_s3" {
  name   = "join-virtualize-job-s3"
  role   = aws_iam_role.job.id
  policy = data.aws_iam_policy_document.job_s3.json
}

# SSM access (security requirement)
data "aws_iam_policy_document" "job_ssm" {
  statement {
    sid = "SSMAccess"
    actions = [
      "ssm:GetParameter",
      "ssm:GetParameters",
      "ssm:GetParametersByPath",
      "ssm:DescribeParameters",
    ]
    resources = ["arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/join/*"]
  }
}

resource "aws_iam_role_policy" "job_ssm" {
  name   = "join-virtualize-job-ssm"
  role   = aws_iam_role.job.id
  policy = data.aws_iam_policy_document.job_ssm.json
}

# ---------------------------------------------------------------------------
# Security group for Batch compute (outbound only — containers pull from S3)
# ---------------------------------------------------------------------------

resource "aws_security_group" "batch" {
  name        = "join-virtualize-batch-sg"
  description = "Egress-only SG for join-virtualize Batch compute"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound (S3, ECR, SSM)"
  }

  tags = { Name = "join-virtualize-batch-sg" }
}

# ---------------------------------------------------------------------------
# Batch compute environment — Spot, m7i.4xlarge
# ---------------------------------------------------------------------------

resource "aws_batch_compute_environment" "virtualize" {
  compute_environment_name = "join-virtualize"
  type                     = "MANAGED"
  service_role             = aws_iam_role.batch_service.arn

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"

    instance_type = var.instance_types
    min_vcpus     = 0
    max_vcpus     = var.max_vcpus
    desired_vcpus = 0

    subnets            = data.aws_subnets.batch.ids
    security_group_ids = [aws_security_group.batch.id]

    # Instance profile for Batch EC2 instances
    instance_role = aws_iam_instance_profile.batch_ec2.arn

    tags = { Name = "join-virtualize-batch" }
  }

  depends_on = [aws_iam_role_policy_attachment.batch_service_managed]

  lifecycle {
    # Prevent TF from destroying the CE when updating instance_type list
    create_before_destroy = true
  }
}

# Instance profile that Batch attaches to each EC2 node
resource "aws_iam_role" "batch_ec2" {
  name = "join-virtualize-batch-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_ec2_ecs" {
  role       = aws_iam_role.batch_ec2.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch_ec2" {
  name = "join-virtualize-batch-ec2-profile"
  role = aws_iam_role.batch_ec2.name
}

# ---------------------------------------------------------------------------
# Batch job queue
# ---------------------------------------------------------------------------

resource "aws_batch_job_queue" "virtualize" {
  name     = "join-virtualize"
  state    = "ENABLED"
  priority = 10

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.virtualize.arn
  }

  tags = { Name = "join-virtualize" }
}

# ---------------------------------------------------------------------------
# Batch job definitions
# ---------------------------------------------------------------------------

locals {
  ecr_image = "${aws_ecr_repository.virtualize.repository_url}:${var.image_tag}"

  # Shared resource requirements for all virtualization jobs
  job_vcpus  = 8
  job_memory = 32768  # 32 GB MiB

  common_env = [
    { name = "AWS_DEFAULT_REGION", value = var.aws_region },
  ]
}

resource "aws_batch_job_definition" "earthcare" {
  name = "join-virtualize-earthcare"
  type = "container"

  platform_capabilities = ["EC2"]

  container_properties = jsonencode({
    image            = local.ecr_image
    jobRoleArn       = aws_iam_role.job.arn
    executionRoleArn = aws_iam_role.job_execution.arn

    command = [
      "python", "scripts/earthcare_virtualize.py",
      "--execution-type", "prod",
    ]

    environment = local.common_env

    resourceRequirements = [
      { type = "VCPU",   value = tostring(local.job_vcpus) },
      { type = "MEMORY", value = tostring(local.job_memory) },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/aws/batch/join-virtualize"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "earthcare"
      }
    }
  })

  retry_strategy {
    attempts = 2
  }

  timeout {
    attempt_duration_seconds = 14400  # 4 hours
  }

  tags = { Name = "join-virtualize-earthcare" }
}

resource "aws_batch_job_definition" "gpmdpr" {
  name = "join-virtualize-gpmdpr"
  type = "container"

  platform_capabilities = ["EC2"]

  container_properties = jsonencode({
    image            = local.ecr_image
    jobRoleArn       = aws_iam_role.job.arn
    executionRoleArn = aws_iam_role.job_execution.arn

    command = [
      "python", "scripts/gpmdpr_virtualize.py",
      "--execution-type", "prod",
    ]

    environment = local.common_env

    resourceRequirements = [
      { type = "VCPU",   value = tostring(local.job_vcpus) },
      { type = "MEMORY", value = tostring(local.job_memory) },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/aws/batch/join-virtualize"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "gpmdpr"
      }
    }
  })

  retry_strategy {
    attempts = 2
  }

  timeout {
    attempt_duration_seconds = 21600  # 6 hours (88 large granules)
  }

  tags = { Name = "join-virtualize-gpmdpr" }
}

# ---------------------------------------------------------------------------
# CloudWatch log group
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/batch/join-virtualize"
  retention_in_days = 30
  tags              = { Name = "join-virtualize-batch-logs" }
}
