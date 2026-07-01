# ── AMSR2 IceChunk init-store — AWS Lambda container image ────────────────────
#
# Simple single-stage build on the official Lambda Python 3.12 base image.
# All deps installed directly with pip — no pixi/conda involved.
# ──────────────────────────────────────────────────────────────────────────────

FROM public.ecr.aws/lambda/python:3.12

# Install dependencies directly into the Lambda task root.
RUN pip install --no-cache-dir \
    "icechunk>=2.0.4,<3" \
    "zarr>=3.1,<4" \
    "obstore>=0.9.2,<0.10" \
    "numpy>=1.26" \
    "pandas>=2.0" \
    "boto3>=1.34"

ENV RUST_LOG=error

COPY cloud/lambda/init_store/handler.py ${LAMBDA_TASK_ROOT}/
COPY cloud/lambda/init_store/schemas/ ${LAMBDA_TASK_ROOT}/schemas/

CMD ["handler.handler"]
