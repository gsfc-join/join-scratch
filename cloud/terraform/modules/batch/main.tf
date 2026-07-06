# ── CloudWatch log groups ──────────────────────────────────────────────────────

data "aws_caller_identity" "current" {}

resource "aws_cloudwatch_log_group" "regrid" {
  name              = "/aws/batch/${var.prefix}-esmf-regrid"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "icechunk" {
  name              = "/aws/batch/${var.prefix}-icechunk"
  retention_in_days = 30
}

# ── Shared Fargate compute environment ────────────────────────────────────────
# All three job definitions share one compute environment and job queue to
# keep the Fargate spot pool consolidated.

resource "aws_batch_compute_environment" "main" {
  name_prefix  = "${var.prefix}-main-"
  type         = "MANAGED"
  service_role = var.batch_service_role_arn

  compute_resources {
    type      = "FARGATE_SPOT"
    max_vcpus = var.max_vcpus

    subnets            = var.subnet_ids
    security_group_ids = var.security_group_ids
  }
}

# ── Job queue ──────────────────────────────────────────────────────────────────

resource "aws_batch_job_queue" "main" {
  name     = "${var.prefix}-main"
  state    = "ENABLED"
  priority = 10

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.main.arn
  }
}

# ── Job definition: ESMF weight generation ────────────────────────────────────

resource "aws_batch_job_definition" "regrid" {
  name = "${var.prefix}-esmf-regrid"
  type = "container"

  platform_capabilities = ["FARGATE"]

  parameters = {
    source           = ""
    dst_grid         = var.default_lis_path
    weights_uri      = var.default_weights_uri
    extra_args       = var.default_extra_args
    method           = "bilinear"
    force_regenerate = "false"
  }

  container_properties = jsonencode({
    image = "${var.regrid_ecr_url}:latest"

    resourceRequirements = [
      { type = "VCPU",   value = tostring(var.regrid_vcpus) },
      { type = "MEMORY", value = tostring(var.regrid_memory_mb) },
    ]

    environment = [
      { name = "SOURCE",           value = "Ref::source" },
      { name = "DST_GRID",          value = "Ref::dst_grid" },
      { name = "WEIGHTS_URI",       value = "Ref::weights_uri" },
      { name = "EXTRA_ARGS",        value = "Ref::extra_args" },
      { name = "METHOD",            value = "Ref::method" },
      { name = "FORCE_REGENERATE",  value = "Ref::force_regenerate" },
      { name = "AWS_REGION",        value = var.aws_region },
    ]

    executionRoleArn = var.execution_role_arn
    jobRoleArn       = var.regrid_task_role_arn

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.regrid.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "batch"
      }
    }

    user                   = "1000"
    readonlyRootFilesystem = false

    networkConfiguration = {
      assignPublicIp = "ENABLED"
    }

    fargatePlatformConfiguration = {
      platformVersion = "LATEST"
    }
  })

  retry_strategy {
    attempts = 2
    evaluate_on_exit {
      on_exit_code = "1"
      action       = "RETRY"
    }
    evaluate_on_exit {
      on_status_reason = "Host EC2*"
      action           = "RETRY"
    }
  }

  timeout {
    attempt_duration_seconds = 1800
  }
}

# ── Job definition: IceChunk populate (one date) ─────────────────────────────

resource "aws_batch_job_definition" "icechunk_populate" {
  name = "${var.prefix}-icechunk-populate"
  type = "container"

  platform_capabilities = ["FARGATE"]

  parameters = {
    source           = ""
    store_uri        = var.default_store_uri
    date             = ""
    force_repopulate = "false"
  }

  container_properties = jsonencode({
    image   = "${var.icechunk_ecr_url}:latest"
    command = ["python", "/app/populate_job.py"]

    resourceRequirements = [
      { type = "VCPU",   value = tostring(var.icechunk_vcpus) },
      { type = "MEMORY", value = tostring(var.icechunk_memory_mb) },
    ]

    environment = [
      { name = "SOURCE",           value = "Ref::source" },
      { name = "STORE_URI",        value = "Ref::store_uri" },
      { name = "DATE",             value = "Ref::date" },
      { name = "FORCE_REPOPULATE", value = "Ref::force_repopulate" },
      { name = "AWS_REGION",       value = var.aws_region },
    ]

    executionRoleArn = var.execution_role_arn
    jobRoleArn       = var.icechunk_task_role_arn

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.icechunk.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "populate"
      }
    }

    user                   = "1000"
    readonlyRootFilesystem = false

    networkConfiguration = {
      assignPublicIp = "ENABLED"
    }

    fargatePlatformConfiguration = {
      platformVersion = "LATEST"
    }
  })

  retry_strategy {
    attempts = 3
    evaluate_on_exit {
      on_exit_code = "1"
      action       = "RETRY"
    }
    evaluate_on_exit {
      on_status_reason = "Host EC2*"
      action           = "RETRY"
    }
  }

  timeout {
    # G-Portal manifest downloads are usually fast but may be slow for large dates.
    attempt_duration_seconds = 900
  }
}
