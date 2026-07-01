#!/usr/bin/env bash
# Build and push the init-store Lambda container image to ECR.
#
# Usage:
#   ./build_and_push.sh [--tag <tag>] [--region <region>]
#
# The ECR repository URL is resolved from:
#   1. terraform output -raw init_store_lambda_ecr_repository_url  (if terraform is on PATH)
#   2. ECR_REPO_URI environment variable
#
# The build context must be the repository root so that
# cloud/lambda/init_store/ files are accessible.
#
# Example:
#   cd /path/to/join-scratch
#   ./cloud/lambda/init_store/build_and_push.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ── defaults ───────────────────────────────────────────────────────────────────
TAG="latest"
REGION="us-west-2"

# ── parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)    TAG="$2";    shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── resolve ECR repo URI ───────────────────────────────────────────────────────
if [[ -z "${ECR_REPO_URI:-}" ]]; then
  TF_DIR="${REPO_ROOT}/cloud/terraform"
  if command -v terraform &>/dev/null && [[ -d "${TF_DIR}" ]]; then
    echo "Resolving ECR repo URI from terraform output …"
    ECR_REPO_URI="$(terraform -chdir="${TF_DIR}" output -raw init_store_lambda_ecr_repository_url 2>/dev/null || true)"
  fi
fi

if [[ -z "${ECR_REPO_URI:-}" ]]; then
  echo "ERROR: Could not determine ECR_REPO_URI." >&2
  echo "  Set ECR_REPO_URI env var or ensure 'terraform output init_store_lambda_ecr_repository_url' works." >&2
  exit 1
fi

ACCOUNT_ID="$(echo "${ECR_REPO_URI}" | cut -d. -f1)"
IMAGE_URI="${ECR_REPO_URI}:${TAG}"
GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
SHA_URI="${ECR_REPO_URI}:${GIT_SHA}"

echo "ECR repo : ${ECR_REPO_URI}"
echo "Tag      : ${TAG}  (also pushing :${GIT_SHA})"
echo "Region   : ${REGION}"
echo ""

# ── ECR login ─────────────────────────────────────────────────────────────────
echo "Authenticating with ECR …"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# ── build ──────────────────────────────────────────────────────────────────────
echo "Building image (context: ${REPO_ROOT}) …"
docker build \
  --platform linux/amd64 \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --tag "${IMAGE_URI}" \
  --tag "${SHA_URI}" \
  "${REPO_ROOT}"

# ── push ───────────────────────────────────────────────────────────────────────
echo "Pushing ${IMAGE_URI} …"
docker push "${IMAGE_URI}"
echo "Pushing ${SHA_URI} …"
docker push "${SHA_URI}"

# ── update Lambda to use the new image ────────────────────────────────────────
FUNCTION_NAME="$(terraform -chdir="${REPO_ROOT}/cloud/terraform" output -raw init_store_lambda_function_name 2>/dev/null || echo "")"
if [[ -n "${FUNCTION_NAME}" ]]; then
  echo "Updating Lambda function ${FUNCTION_NAME} to ${IMAGE_URI} …"
  aws --region "${REGION}" lambda update-function-code \
    --function-name "${FUNCTION_NAME}" \
    --image-uri "${IMAGE_URI}" \
    --query 'FunctionName' \
    --output text
  echo "Lambda updated."
else
  echo "WARNING: Could not resolve Lambda function name from terraform output — skipping lambda update-function-code."
  echo "  Run manually: aws lambda update-function-code --function-name <name> --image-uri ${IMAGE_URI}"
fi

echo ""
echo "Done.  Example Lambda invoke:"
echo ""
echo "  aws lambda invoke \\"
echo "    --function-name join-amsr2-init-store \\"
echo "    --payload '{\"store_uri\":\"s3://airborne-smce-prod-user-bucket/JOIN/icechunk-stores/GCOM-W1-AMSR2-L3-SND\",\"recreate_store\":false}' \\"
echo "    --cli-binary-format raw-in-base64-out \\"
echo "    /tmp/response.json && cat /tmp/response.json"
