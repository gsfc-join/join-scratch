terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = var.tags
  }
}

locals {
  prefix = "${var.project}-${var.name}"
}

# ── ECR repositories ───────────────────────────────────────────────────────────

module "ecr" {
  source = "./modules/ecr"
  prefix = var.project
}

# ── IAM roles ─────────────────────────────────────────────────────────────────
# Lambda ARNs aren't known until the lambda module runs, so we pass them
# in after both modules are planned (Terraform handles the dependency graph).

module "iam" {
  source = "./modules/iam"

  prefix                = local.prefix
  s3_data_bucket        = var.s3_data_bucket
  weights_prefix        = var.weights_prefix
  icechunk_store_prefix = var.icechunk_store_prefix

  # Grant Step Functions permission to invoke all three Lambda functions.
  lambda_function_arns = module.lambda.all_function_arns
}

# ── Lambda functions ───────────────────────────────────────────────────────────

module "lambda" {
  source = "./modules/lambda"

  prefix               = local.prefix
  lambda_role_arn      = module.iam.lambda_role_arn
  repo_root            = var.repo_root
  init_store_image_uri = "${module.ecr.init_store_lambda_repository_url}:latest"
}

# ── Batch resources ────────────────────────────────────────────────────────────

module "batch" {
  source = "./modules/batch"

  prefix     = local.prefix
  aws_region = var.aws_region

  subnet_ids         = var.subnet_ids
  security_group_ids = var.security_group_ids
  max_vcpus          = var.max_vcpus

  regrid_ecr_url   = module.ecr.regrid_repository_url
  icechunk_ecr_url = module.ecr.icechunk_repository_url

  execution_role_arn     = module.iam.execution_role_arn
  regrid_task_role_arn   = module.iam.regrid_task_role_arn
  icechunk_task_role_arn = module.iam.icechunk_task_role_arn
  batch_service_role_arn = module.iam.batch_service_role_arn

  regrid_vcpus       = var.regrid_vcpus
  regrid_memory_mb   = var.regrid_memory_mb
  icechunk_vcpus     = var.icechunk_vcpus
  icechunk_memory_mb = var.icechunk_memory_mb

  default_lis_path    = var.default_lis_path
  default_weights_uri = var.default_weights_uri
  default_extra_args  = var.default_extra_args
  default_store_uri   = var.default_store_uri
}

# ── Step Functions state machine ───────────────────────────────────────────────

module "stepfunctions" {
  source = "./modules/stepfunctions"

  prefix       = local.prefix
  sfn_role_arn = module.iam.sfn_role_arn

  enumerate_dates_arn = module.lambda.enumerate_dates_arn
  check_weights_arn   = module.lambda.check_weights_arn
  apply_regrid_arn    = module.lambda.apply_regrid_arn
  init_store_arn      = module.lambda.init_store_arn

  batch_job_queue_arn                  = module.batch.job_queue_arn
  icechunk_populate_job_definition_arn = module.batch.icechunk_populate_job_definition_arn
  regrid_job_definition_arn            = module.batch.regrid_job_definition_arn
}
