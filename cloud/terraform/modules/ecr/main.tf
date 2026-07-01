# ── ECR repositories ───────────────────────────────────────────────────────────
# Two repositories:
#   - join-esmf-regrid  : ESMF weight-generation Batch image
#   - join-icechunk     : IceChunk init/populate Batch image

locals {
  lifecycle_policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Keep only the 10 most recent tagged images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v", "latest"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = { type = "expire" }
      }
    ]
  })
}

resource "aws_ecr_repository" "regrid" {
  name                 = "${var.prefix}-esmf-regrid"
  image_tag_mutability = "MUTABLE"

  encryption_configuration {
    encryption_type = "AES256"
  }

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "regrid" {
  repository = aws_ecr_repository.regrid.name
  policy     = local.lifecycle_policy
}

resource "aws_ecr_repository" "icechunk" {
  name                 = "${var.prefix}-icechunk"
  image_tag_mutability = "MUTABLE"

  encryption_configuration {
    encryption_type = "AES256"
  }

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "icechunk" {
  repository = aws_ecr_repository.icechunk.name
  policy     = local.lifecycle_policy
}

resource "aws_ecr_repository" "init_store_lambda" {
  name                 = "${var.prefix}-init-store-lambda"
  image_tag_mutability = "MUTABLE"

  encryption_configuration {
    encryption_type = "AES256"
  }

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "init_store_lambda" {
  repository = aws_ecr_repository.init_store_lambda.name
  policy     = local.lifecycle_policy
}
