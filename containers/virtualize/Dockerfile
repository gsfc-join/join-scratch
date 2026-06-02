# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# join-virtualize — shared container for EarthCARE and GPM-DPR IceChunk
# virtual store creation.
#
# Base: mambaorg/micromamba (conda-forge, linux/amd64)
# Target arch: linux/amd64  (AWS Batch x86 instances)
# ---------------------------------------------------------------------------
FROM mambaorg/micromamba:latest AS base

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# ---------------------------------------------------------------------------
# System packages needed by h5py / obstore (libhdf5, libssl, ca-certs)
# ---------------------------------------------------------------------------
USER root
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# ---------------------------------------------------------------------------
# Conda environment — all packages from conda-forge for reproducibility
# ---------------------------------------------------------------------------
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# ---------------------------------------------------------------------------
# Application scripts
# ---------------------------------------------------------------------------
COPY --chown=$MAMBA_USER:$MAMBA_USER scripts/ /app/scripts/

WORKDIR /app

# Default: print usage.  Batch jobs override this via `command`.
CMD ["python", "-c", \
     "print('Usage: python scripts/earthcare_virtualize.py --execution-type <local-test|prod>\\n       python scripts/gpmdpr_virtualize.py --execution-type <local-test|prod>')"]
