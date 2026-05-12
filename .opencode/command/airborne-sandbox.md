---
description: Create a Terraform EC2 instance for development.
---

Create a folder called `_aws-sandbox`. This folder will be version controlled; it should not be gitignored. 

In in `_aws-sandbox`, create a basic Terraform deployment for an AWS EC2 instance in `us-west-2`.
Prompt the user for the following parameters with these as the defaults:

- Instance type -- default =  `m7i` with 8 VCPUs
- OS --- default = Amazon Linux 2023 operating system (use the latest AL2023 AMI)
- S3 access -- default = Full read/write/list access to all S3 buckets
- Networking --- default = default VPC, default public subnet
- AZ --- default = select at random
- Root volume size and type --- default = 100 GB gp3

Also, assume the following defaults without prompting the user:

- These IAM policies for SSM: "AmazonSSMManagedInstanceCore", "CloudWatchAgentAdminPolicy", "CloudWatchAgentServerPolicy"

Include a local-exec provisioner like this (but update if you think of something better) to make sure the command doesn't complete until ready.

```
  provisioner "local-exec" {
    command = <<EOT
    if [ ${var.instance_state} != "stopped" ]; then
      aws ec2 wait instance-status-ok --instance-ids ${self.id} --region ${var.aws_region}
    fi
    EOT
  }
```

Include the instance ID in the outputs.

Create a `justfile` in the repo root (or add if it doesn't exist) with the following commands:

- sandbox_create --- `terraform apply` + write outputs to `_tfout.json` (which should be gitignored). don't skip terraform's confirmation.
- sandbox_connect --- AWS SSM command to connect to the instance interactively
- sandbox_status --- Check the instance status (running, stopped, etc.)
- sandbox_start --- Start the instance if it isn't already running. Wait until the instance is ready (using similar command to the provisioner above)
- sandbox_stop --- Stop the instance if it isn't already stopped.
- sandbox_destroy --- `terraform destroy` (don't skip terraform's confirmation)

Ask the user if you should just create the instance or also launch it.
If launching, check that AWS is already authenticated via AWS CLI STS command. If not, tell the user you are not connected; then, proceed with creating the configuration files but don't apply them.

## Auto-stop schedule

The deployment includes an optional auto-stop feature using **EventBridge Scheduler** (`aws_scheduler_schedule`). When enabled, it automatically stops the instance on a configurable schedule (default: 6pm ET daily). Stopping an already-stopped instance is a no-op, so this is safe to leave enabled.

Key details:
- Controlled by the `auto_stop_schedule` variable (EventBridge Scheduler cron expression). Set to `""` to disable.
- Default: `"cron(0 18 * * ? *)"` — 6pm every day
- Timezone: `America/New_York` (DST-aware; no manual UTC offset adjustment needed)
- Mechanism: EventBridge Scheduler → SSM Automation `AWS-StopEC2Instance` → EC2 `StopInstances`
- A dedicated IAM role is created for the scheduler with least-privilege permissions scoped to the specific instance
- All resources are conditionally created (`count = var.auto_stop_schedule != "" ? 1 : 0`) 
