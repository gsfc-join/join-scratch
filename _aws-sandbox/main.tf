terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

# Latest Amazon Linux 2023 AMI
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Default VPC
data "aws_vpc" "default" {
  count   = var.vpc_id == "" ? 1 : 0
  default = true
}

locals {
  vpc_id = var.vpc_id != "" ? var.vpc_id : data.aws_vpc.default[0].id
  ami_id = var.ami_id != "" ? var.ami_id : data.aws_ami.al2023.id
}

# Default subnets in the selected VPC
data "aws_subnets" "default" {
  count = var.subnet_id == "" ? 1 : 0

  filter {
    name   = "vpc-id"
    values = [local.vpc_id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

locals {
  # Pick a random subnet from the defaults if none specified
  subnet_ids       = var.subnet_id == "" ? data.aws_subnets.default[0].ids : [var.subnet_id]
  selected_subnet  = var.subnet_id != "" ? var.subnet_id : local.subnet_ids[random_integer.subnet_index.result % length(local.subnet_ids)]
}

resource "random_integer" "subnet_index" {
  min = 0
  max = 99
}

# ---------------------------------------------------------------------------
# IAM role and instance profile
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sandbox" {
  name               = "${var.instance_name}-instance-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json

  tags = {
    Name      = var.instance_name
    ManagedBy = "Terraform"
  }
}

# SSM managed policies
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.sandbox.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "cw_agent_admin" {
  role       = aws_iam_role.sandbox.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentAdminPolicy"
}

resource "aws_iam_role_policy_attachment" "cw_agent_server" {
  role       = aws_iam_role.sandbox.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

# S3 access policy
data "aws_iam_policy_document" "s3_full" {
  count = var.s3_access == "full" ? 1 : 0

  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:ListAllMyBuckets",
      "s3:GetBucketLocation",
    ]
    resources = ["*"]
  }
}

data "aws_iam_policy_document" "s3_readonly" {
  count = var.s3_access == "readonly" ? 1 : 0

  statement {
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
      "s3:ListAllMyBuckets",
      "s3:GetBucketLocation",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "s3_access" {
  count  = var.s3_access != "none" ? 1 : 0
  name   = "s3-access"
  role   = aws_iam_role.sandbox.id
  policy = var.s3_access == "full" ? data.aws_iam_policy_document.s3_full[0].json : data.aws_iam_policy_document.s3_readonly[0].json
}

resource "aws_iam_instance_profile" "sandbox" {
  name = "${var.instance_name}-instance-profile"
  role = aws_iam_role.sandbox.name
}

# ---------------------------------------------------------------------------
# Security group
# ---------------------------------------------------------------------------

resource "aws_security_group" "sandbox" {
  name        = "${var.instance_name}-sg"
  description = "Security group for sandbox EC2 instance"
  vpc_id      = local.vpc_id

  # SSM does not require inbound SSH; all outbound allowed for updates
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = {
    Name      = "${var.instance_name}-sg"
    ManagedBy = "Terraform"
  }
}

# ---------------------------------------------------------------------------
# EC2 instance
# ---------------------------------------------------------------------------

resource "aws_instance" "sandbox" {
  ami                    = local.ami_id
  instance_type          = var.instance_type
  subnet_id              = local.selected_subnet
  iam_instance_profile   = aws_iam_instance_profile.sandbox.name
  vpc_security_group_ids = [aws_security_group.sandbox.id]

  # Place in specified AZ (or let AWS pick based on subnet if empty)
  availability_zone = var.availability_zone != "" ? var.availability_zone : null

  root_block_device {
    volume_size = var.root_volume_size
    volume_type = var.root_volume_type
    encrypted   = true

    tags = {
      Name      = "${var.instance_name}-root"
      ManagedBy = "Terraform"
    }
  }

  tags = {
    Name      = var.instance_name
    ManagedBy = "Terraform"
  }

  provisioner "local-exec" {
    command = <<EOT
if [ "${var.instance_state}" != "stopped" ]; then
  aws ec2 wait instance-status-ok --instance-ids ${self.id} --region ${var.aws_region}
fi
EOT
  }
}

# Manage instance state (start/stop)
resource "aws_ec2_instance_state" "sandbox" {
  instance_id = aws_instance.sandbox.id
  state       = var.instance_state
}
