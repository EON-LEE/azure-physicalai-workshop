FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    WANDB_MODE=disabled \
    WANDB_DISABLED=true \
    DO_NOT_TRACK=1 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    TORCH_CUDA_ARCH_LIST="8.0;8.6" \
    MAX_JOBS=4 \
    UV_PROJECT_ENVIRONMENT=/opt/gr00t-venv \
    PATH=/opt/gr00t-venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.12.1 ninja==1.11.1.3 packaging==24.2

WORKDIR /opt/dependencies
COPY learning/gr00t/pyproject.toml learning/gr00t/uv.lock ./
RUN uv venv /opt/gr00t-venv --python /opt/conda/bin/python --system-site-packages \
    && uv sync --frozen --extra runtime --extra azure --python /opt/conda/bin/python \
    && git clone --quiet --filter=blob:none --no-checkout \
       https://github.com/NVIDIA/Isaac-GR00T.git /opt/isaac-gr00t \
    && git -C /opt/isaac-gr00t checkout --quiet --detach 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96 \
    && test "$(git -C /opt/isaac-gr00t rev-parse HEAD)" = 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96 \
    && python -c "import torch, flash_attn, av, pyarrow; assert torch.__version__.split('+')[0] == '2.5.1'"

WORKDIR /work
