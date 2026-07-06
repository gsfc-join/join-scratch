#!/usr/bin/env bash
# Build and push the IceChunk Batch container image to ECR.
#
# Usage:
#   ./build_and_push.sh [--tag <tag>] [--region <region>]
#
# The ECR repository URL is resolved from:
#   1. terraform output -raw icechunk_ecr_repository_url  (if terraform is on PATH)
#   2. ECR_REPO_URI environment variable
#
# The build context must be the repository root so that pyproject.toml,
# pixi.lock, and both batch/icechunk/*.py files are accessible.
#
# Example:
#   cd /path/to/join-scratch
#   ./batch/icechunk/build_and_push.sh

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
    ECR_REPO_URI="$(terraform -chdir="${TF_DIR}" output -raw icechunk_ecr_repository_url 2>/dev/null || true)"
  fi
fi

if [[ -z "${ECR_REPO_URI:-}" ]]; then
  echo "ERROR: Could not determine ECR_REPO_URI." >&2
  echo "  Set ECR_REPO_URI env var or ensure 'terraform output icechunk_ecr_repository_url' works." >&2
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

echo ""
echo "Done.  Example Batch submit commands:"
echo ""
echo "  # Init store (idempotent)"
echo "  aws batch submit-job \\"
echo "    --job-name icechunk-init-store \\"
echo "    --job-queue join-icechunk \\"
echo "    --job-definition join-icechunk-init-store \\"
echo "    --parameters store_uri=s3://your-bucket/path/to/store"
echo ""
echo "  # Populate one date"
echo "  aws batch submit-job \\"
echo "    --job-name icechunk-populate-20230115 \\"
echo "    --job-queue join-icechunk \\"
echo "    --job-definition join-icechunk-populate \\"
echo "    --parameters store_uri=s3://your-bucket/path/to/store,date=20230115"
