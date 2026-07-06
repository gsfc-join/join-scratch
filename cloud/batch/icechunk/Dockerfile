# ── AMSR2 IceChunk store jobs — AWS Batch container image ─────────────────────
#
# Multi-stage build using pixi (same pattern as batch/regrid/Dockerfile).
#   Stage 1 (build):  pixi installs the `icechunk` environment from the lockfile.
#   Stage 2 (final):  plain ubuntu:24.04 — only the resolved env is copied in.
#
# Approximate uncompressed final image size: ~500–650 MB
# Recommended Fargate task: 2 vCPU / 8 GB RAM
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: install the pixi icechunk environment ────────────────────────────
FROM ghcr.io/prefix-dev/pixi:0.41.4 AS build

WORKDIR /app

# Copy only what pixi needs to resolve and install the environment.
COPY cloud/batch/icechunk/pyproject.toml ./

RUN pixi install -e icechunk

# Generate a self-contained activation script that sets PATH, LD_LIBRARY_PATH,
# etc. without requiring pixi to be present at runtime.
RUN pixi shell-hook -e icechunk -s bash > /shell-hook.sh && \
    printf '#!/bin/bash\n' > /entrypoint.sh && \
    cat /shell-hook.sh >> /entrypoint.sh && \
    printf 'exec "$@"\n' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# ── Stage 2: lean production image ────────────────────────────────────────────
FROM ubuntu:24.04 AS final

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the fully resolved pixi environment.
COPY --from=build /app/.pixi/envs/icechunk /app/.pixi/envs/icechunk

# Copy the activation entrypoint.
COPY --from=build --chmod=0755 /entrypoint.sh /entrypoint.sh

# Copy application code.
COPY cloud/batch/icechunk/populate_job.py ./
COPY cloud/batch/icechunk/sources/ ./sources/

# IceChunk/Zarr use Rust-based logging; silence at runtime unless overridden.
ENV RUST_LOG=error

# Run as a non-root user.
RUN useradd -m appuser
USER appuser

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "/app/populate_job.py"]
