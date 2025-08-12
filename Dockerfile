# ==========================================
# Build: docker build -t lerobot .
# Run:   docker run -it --rm --gpus all lerobot bash
# ==========================================

FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# -------- Python & 环境 --------
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    PATH="/opt/venv/bin:$PATH"

# 安装依赖 & Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher libgeos-dev curl wget vim \
    python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
 && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
 && python -m venv /opt/venv \
 && echo "source /opt/venv/bin/activate" >> /root/.bashrc \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /lerobot
COPY . .

RUN /opt/venv/bin/pip install --upgrade --no-cache-dir pip \
 && /opt/venv/bin/pip install --no-cache-dir -e ".[test, aloha, xarm, pusht, dynamixel, serving, pi0]" \
 && /opt/venv/bin/pip install --no-cache-dir esdk-obs-python==3.22.2

# -------- 下载模型 (可选，允许构建时跳过) --------
ARG DOWNLOAD_MODELS=true
RUN if [ "$DOWNLOAD_MODELS" = "true" ]; then \
    mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget -q --show-progress -P /root/.cache/torch/hub/checkpoints \
    https://download.pytorch.org/models/resnet18-f37072fd.pth ; \
    fi
