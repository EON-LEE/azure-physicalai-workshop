FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d

ARG ACCELERATOR=cuda
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    WANDB_MODE=disabled \
    WANDB_DISABLED=true \
    DO_NOT_TRACK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/learning-venv \
    PATH=/opt/learning-venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential linux-libc-dev libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.12.1

WORKDIR /opt/dependencies
COPY learning/pyproject.toml learning/uv.lock learning/.python-version ./
RUN case "$ACCELERATOR" in cpu|cuda) ;; *) exit 2 ;; esac \
    && uv sync --frozen --no-dev --extra "$ACCELERATOR" --python /usr/local/bin/python \
    && python -c "from importlib.metadata import version; assert version('lerobot') == '0.4.4'"

WORKDIR /work
