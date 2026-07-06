#!/usr/bin/env bash
# build_and_push.sh — Build the ESMF regrid Docker image and push to ECR.
#
# Usage:
#   ./build_and_push.sh [--tag <image-tag>] [--region <aws-region>]
#
# The ECR repository URI is read from Terraform outputs automatically if
# terraform is on PATH and the workspace is initialised, otherwise set
# ECR_REPO_URI in the environment.
#
# Example (explicit URI):
#   ECR_REPO_URI=123456789012.dkr.ecr.us-west-2.amazonaws.com/join-esmf-regrid \
#     ./build_and_push.sh --tag v1.0.0
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TERRAFORM_DIR="${REPO_ROOT}/cloud/terraform"

IMAGE_TAG="${IMAGE_TAG:-latest}"
AWS_REGION="${AWS_REGION:-us-west-2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)    IMAGE_TAG="$2";  shift 2 ;;
    --region) AWS_REGION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── resolve ECR repository URI ─────────────────────────────────────────────────
if [[ -z "${ECR_REPO_URI:-}" ]]; then
  if command -v terraform &>/dev/null && [[ -d "${TERRAFORM_DIR}" ]]; then
    echo ">>> Reading ECR repository URI from Terraform outputs …"
    ECR_REPO_URI="$(terraform -chdir="${TERRAFORM_DIR}" output -raw regrid_ecr_repository_url 2>/dev/null)" || true
  fi
fi

if [[ -z "${ECR_REPO_URI:-}" ]]; then
  echo "ERROR: ECR_REPO_URI is not set and could not be read from Terraform outputs." >&2
  echo "       Set ECR_REPO_URI=<account>.dkr.ecr.<region>.amazonaws.com/<repo-name>" >&2
  exit 1
fi

FULL_IMAGE="${ECR_REPO_URI}:${IMAGE_TAG}"
echo ">>> Target image : ${FULL_IMAGE}"
echo ">>> AWS region   : ${AWS_REGION}"

# ── authenticate with ECR ──────────────────────────────────────────────────────
echo ">>> Logging in to ECR …"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REPO_URI%%/*}"

# ── build ──────────────────────────────────────────────────────────────────────
echo ">>> Building Docker image …"
# Build context is the repo root so Docker can access pyproject.toml, pixi.lock,
# and batch/regrid/ in a single build context (as referenced in the Dockerfile).
docker build \
  --platform linux/amd64 \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --tag "${FULL_IMAGE}" \
  --tag "${ECR_REPO_URI}:latest" \
  "${REPO_ROOT}"

# ── push ───────────────────────────────────────────────────────────────────────
echo ">>> Pushing ${FULL_IMAGE} …"
docker push "${FULL_IMAGE}"

if [[ "${IMAGE_TAG}" != "latest" ]]; then
  echo ">>> Pushing ${ECR_REPO_URI}:latest …"
  docker push "${ECR_REPO_URI}:latest"
fi

echo ""
echo ">>> Done. Image: ${FULL_IMAGE}"
echo ""
echo "    Submit a test job:"
echo "      aws batch submit-job \\"
echo "        --job-name esmf-regrid-test \\"
echo "        --job-queue  \$(terraform -chdir=${TERRAFORM_DIR} output -raw batch_job_queue_name) \\"
echo "        --job-definition \$(terraform -chdir=${TERRAFORM_DIR} output -raw batch_job_definition_name) \\"
echo "        --parameters 'lis_path=s3://airborne-smce-prod-user-bucket/JOIN/lis_input_NMP_1000m_missouri.nc,weights_uri=s3://airborne-smce-prod-user-bucket/JOIN/cached-weights/test-weights.nc4'"
