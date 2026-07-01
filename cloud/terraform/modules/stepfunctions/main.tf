resource "aws_cloudwatch_log_group" "sfn" {
  name              = "/aws/states/${var.prefix}-amsr2-pipeline"
  retention_in_days = 30
}

resource "aws_sfn_state_machine" "amsr2_pipeline" {
  name     = "${var.prefix}-amsr2-pipeline"
  role_arn = var.sfn_role_arn

  definition = templatefile("${path.module}/state_machine.asl.json", {
    enumerate_dates_arn                  = var.enumerate_dates_arn
    check_weights_arn                    = var.check_weights_arn
    apply_regrid_arn                     = var.apply_regrid_arn
    init_store_arn                       = var.init_store_arn
    batch_job_queue_arn                  = var.batch_job_queue_arn
    icechunk_populate_job_definition_arn = var.icechunk_populate_job_definition_arn
    regrid_job_definition_arn            = var.regrid_job_definition_arn
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.sfn.arn}:*"
    include_execution_data = true
    level                  = "ERROR"
  }

  tracing_configuration {
    enabled = true
  }
}
