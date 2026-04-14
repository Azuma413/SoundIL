FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch \
    WANDB_DIR=/workspace/myproject/outputs/wandb

WORKDIR /workspace/myproject

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libglib2.0-0 \
    libgl1 \
    libglu1-mesa \
    libgtk2.0-dev \
    libportaudio2 \
    libsndfile1 \
    libusb-1.0-0 \
    libvulkan-dev \
    ninja-build \
    pkg-config \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv python install 3.10

COPY pyproject.toml README.md ./
COPY Genesis/pyproject.toml Genesis/README.md ./Genesis/
COPY lerobot/pyproject.toml lerobot/README.md ./lerobot/
RUN mkdir -p docs src env docker apptainer

COPY . .

RUN uv sync --python 3.10 && \
    uv pip install --python /opt/venv/bin/python -e ./Genesis && \
    uv pip install --python /opt/venv/bin/python -e "./lerobot/[pi]"

CMD ["/bin/bash"]
