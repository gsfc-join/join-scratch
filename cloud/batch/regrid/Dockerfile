# ── AMSR2 ESMF Weight Generation — AWS Batch container image ──────────────────
#
# Multi-stage build using pixi:
#   Stage 1 (build):  pixi installs the `batch` environment from the lockfile.
#   Stage 2 (final):  plain ubuntu:24.04 — only the resolved env is copied in.
#                     No pixi, no conda machinery, no package cache at runtime.
#
# The `batch` pixi feature is defined in pyproject.toml and uses its own
# solve-group so it can pin python=3.11 (esmf on conda-forge requires <=3.12)
# independently of the main python=3.14 development environment.
#
# Approximate uncompressed final image size: ~600–750 MB
# Recommended Fargate task: 4 vCPU / 16 GB RAM
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: install the pixi batch environment ────────────────────────────────
FROM ghcr.io/prefix-dev/pixi:0.41.4 AS build

WORKDIR /app

# Copy only what pixi needs to resolve and install the environment.
# Copying the lockfile ensures a fully reproducible install (--locked).
COPY cloud/batch/regrid/pyproject.toml ./

RUN pixi install -e batch

# Generate a self-contained shell-hook activation script that sets PATH,
# LD_LIBRARY_PATH, etc. without requiring pixi to be present at runtime.
RUN pixi shell-hook -e batch -s bash > /shell-hook.sh && \
    printf '#!/bin/bash\n' > /entrypoint.sh && \
    cat /shell-hook.sh >> /entrypoint.sh && \
    printf 'exec "$@"\n' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# ── Stage 2: lean production image ────────────────────────────────────────────
FROM ubuntu:24.04 AS final

# Minimal runtime system deps: ca-certificates for S3 HTTPS calls.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the fully resolved pixi environment (the path must match the build stage).
COPY --from=build /app/.pixi/envs/batch /app/.pixi/envs/batch

# Copy the activation entrypoint.
COPY --from=build --chmod=0755 /entrypoint.sh /entrypoint.sh

# Copy application code.
# Source modules are taken directly from workflows/swe/ in the repo root.
# sources/ is shared with the icechunk job — it provides SRC_GRID_SPEC per source.
COPY cloud/batch/regrid/regrid_job.py \
     workflows/swe/lis_grid.py \
     workflows/swe/s3_utils.py \
     ./
COPY cloud/batch/regrid/sources/ ./sources/

# ESMF writes PET*.ESMF_LogFile to CWD by default; redirect to /tmp.
# (Suppressed entirely by --no_log in EXTRA_ARGS, but set as a safety net.)
ENV ESMF_LOGFILE_PATH=/tmp
ENV RUST_LOG=error

# Run as a non-root user.
RUN useradd -m appuser
USER appuser

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "/app/regrid_job.py"]
