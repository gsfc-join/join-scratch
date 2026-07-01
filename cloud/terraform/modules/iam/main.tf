data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.id
}

# ── ECS task execution role ────────────────────────────────────────────────────
# Used by ECS/Fargate for both Batch job types: pulls images from ECR and
# sends logs to CloudWatch.

resource "aws_iam_role" "execution" {
  name = "${var.prefix}-batch-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ── Batch task role: ESMF regrid job ──────────────────────────────────────────
# Reads source grids and LIS file; writes weights to S3.

resource "aws_iam_role" "regrid_task" {
  name = "${var.prefix}-regrid-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "regrid_task_s3" {
  name = "s3-regrid"
  role = aws_iam_role.regrid_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadSourceData"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:HeadObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.s3_data_bucket}",
          "arn:aws:s3:::${var.s3_data_bucket}/*",
        ]
      },
      {
        Sid    = "WriteWeights"
        Effect = "Allow"
        Action = ["s3:PutObject", "s3:AbortMultipartUpload"]
        Resource = [
          "arn:aws:s3:::${var.s3_data_bucket}/${var.weights_prefix}*",
        ]
      }
    ]
  })
}

# ── Batch task role: IceChunk jobs (init-store + populate-date) ───────────────
# Reads/writes the IceChunk store on S3 and reads virtual chunks from JAXA
# G-Portal over HTTPS (egress; no AWS policy needed for outbound HTTPS).

resource "aws_iam_role" "icechunk_task" {
  name = "${var.prefix}-icechunk-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "icechunk_task_s3" {
  name = "s3-icechunk"
  role = aws_iam_role.icechunk_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadWriteIceChunkStore"
        Effect = "Allow"
        Action = [
          "s3:GetObject", "s3:HeadObject", "s3:ListBucket",
          "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload",
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_data_bucket}",
          "arn:aws:s3:::${var.s3_data_bucket}/${var.icechunk_store_prefix}*",
        ]
      }
    ]
  })
}

# ── AWS Batch service role ─────────────────────────────────────────────────────

resource "aws_iam_role" "batch_service" {
  name = "${var.prefix}-batch-service"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "batch.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service_managed" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

# ── Lambda execution role (shared by all three Lambda functions) ───────────────

resource "aws_iam_role" "lambda" {
  name = "${var.prefix}-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_s3_head" {
  name = "s3-head-weights"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "HeadWeightsObject"
      Effect = "Allow"
      Action = ["s3:HeadObject", "s3:GetObject", "s3:ListBucket"]
      Resource = [
        "arn:aws:s3:::${var.s3_data_bucket}",
        "arn:aws:s3:::${var.s3_data_bucket}/*",
      ]
    }]
  })
}

resource "aws_iam_role_policy" "lambda_s3_icechunk" {
  name = "s3-icechunk-store"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "ReadWriteIceChunkStore"
      Effect = "Allow"
      Action = [
        "s3:GetObject", "s3:HeadObject", "s3:ListBucket",
        "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload",
      ]
      Resource = [
        "arn:aws:s3:::${var.s3_data_bucket}",
        "arn:aws:s3:::${var.s3_data_bucket}/${var.icechunk_store_prefix}*",
      ]
    }]
  })
}

# ── Step Functions execution role ──────────────────────────────────────────────

resource "aws_iam_role" "sfn" {
  name = "${var.prefix}-sfn"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "states.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "sfn_permissions" {
  name = "sfn-orchestration"
  role = aws_iam_role.sfn.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "InvokeLambda"
        Effect = "Allow"
        Action = ["lambda:InvokeFunction"]
        Resource = var.lambda_function_arns
      },
      {
        Sid    = "SubmitBatchJobs"
        Effect = "Allow"
        Action = [
          "batch:SubmitJob",
          "batch:DescribeJobs",
          "batch:TerminateJob",
        ]
        Resource = "*"
      },
      {
        Sid    = "BatchEventBridge"
        Effect = "Allow"
        Action = [
          "events:PutTargets",
          "events:PutRule",
          "events:DescribeRule",
        ]
        Resource = "arn:aws:events:${local.region}:${local.account_id}:rule/StepFunctionsGetEventsForBatchJobsRule"
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogDelivery",
          "logs:GetLogDelivery",
          "logs:UpdateLogDelivery",
          "logs:DeleteLogDelivery",
          "logs:ListLogDeliveries",
          "logs:PutLogEvents",
          "logs:PutResourcePolicy",
          "logs:DescribeResourcePolicies",
          "logs:DescribeLogGroups",
        ]
        Resource = "*"
      }
    ]
  })
}
