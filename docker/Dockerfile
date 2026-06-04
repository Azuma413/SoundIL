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
COPY libs/Genesis/pyproject.toml libs/Genesis/README.md ./libs/Genesis/
COPY libs/lerobot/pyproject.toml libs/lerobot/README.md ./libs/lerobot/
RUN mkdir -p docs src env docker libs/Genesis libs/lerobot

COPY . .

RUN uv sync --python 3.10

CMD ["/bin/bash"]
