# tmg/isaaclab-soft: Isaac Sim 5.1 (pip) + the SoftMimicGen fork of Isaac Lab + SoftMimicGen + tmg. GPU required
# (RTX class; the pip wheels of isaacsim 5.1 need an NVIDIA driver on the host and `--gpus all`).
# Steps follow the 2026-10 reproduction's bootstrap (python 3.11, torch 2.7.0+cu128, editable Isaac Lab packages).
# Build from the repository root:  make build IMAGE=isaaclab-soft
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ARG TMG_REVISION=unknown
ARG SOFTMIMICGEN_COMMIT=c9d146ba57358a544167de8ebe946caaac8f6220
ARG ISAACLAB_FORK_COMMIT=25c69e088fa8122ad189b3a566d38ef24fa74d6f
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES \
    UV_LINK_MODE=copy UV_CACHE_DIR=/root/.cache/uv OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev git curl ca-certificates build-essential cmake ninja-build \
      libgl1 libglu1-mesa libxt6 libxrandr2 libxinerama1 libxcursor1 libxi6 libglib2.0-0 libsm6 libxext6 ffmpeg \
      libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && ln -s /root/.local/bin/uv /usr/local/bin/uv
RUN uv venv --python python3.11 --seed /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python 'pip<26' 'setuptools<77' wheel toml ninja 'cmake<4' \
    && uv pip install --python /opt/venv/bin/python --index-strategy unsafe-best-match --extra-index-url https://pypi.nvidia.com 'isaacsim[all,extscache]==5.1.0' \
    && uv pip install --python /opt/venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0
# official sources: the SoftMimicGen repository with its Isaac Lab fork and rapprentice as third-party checkouts
RUN mkdir -p /opt/src && cd /opt/src \
    && git clone -q https://github.com/NVlabs/SoftMimicGen SoftMimicGen && git -C SoftMimicGen checkout -q $SOFTMIMICGEN_COMMIT \
    && git clone -q -b softmimicgen https://github.com/masoudmoghani/IsaacLab IsaacLab && git -C IsaacLab checkout -q $ISAACLAB_FORK_COMMIT \
    && git clone -q --depth 1 https://github.com/masoudmoghani/rapprentice rapprentice
RUN --mount=type=cache,target=/root/.cache/uv \
    for p in isaaclab isaaclab_assets isaaclab_tasks isaaclab_mimic; do uv pip install --python /opt/venv/bin/python --no-build-isolation -e /opt/src/IsaacLab/source/$p; done \
    && uv pip install --python /opt/venv/bin/python -e /opt/src/rapprentice qpsolvers==4.8.1 h5py huggingface_hub \
    && for p in softmimicgen softmimicgen_assets softmimicgen_tasks; do uv pip install --python /opt/venv/bin/python --no-build-isolation -e /opt/src/SoftMimicGen/source/$p; done \
    && uv pip install --python /opt/venv/bin/python --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' 'torchaudio==2.7.0+cu128'
# Kit (pip isaacsim) writes its cache, data and logs under the package folder; the containers run as the host user,
# so those folders must be writable. Vulkan (libvulkan1) plus the driver libraries that NVIDIA_DRIVER_CAPABILITIES=all
# mounts give Kit a GPU device inside docker (`--gpus all`).
RUN K=/opt/venv/lib/python3.11/site-packages/isaacsim/kit && mkdir -p $K/cache $K/data $K/logs && chmod -R a+rwX $K/cache $K/data $K/logs
WORKDIR /opt/tmg
COPY pyproject.toml README.md ./
COPY tmg/ tmg/
COPY configs/ configs/
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --python /opt/venv/bin/python --no-deps -e /opt/tmg \
    && uv pip install --python /opt/venv/bin/python 'hydra-core>=1.3,<1.4' 'omegaconf>=2.3,<2.4' pyyaml scipy msgpack websockets
ENV TMG_DATA_ROOT=/data TMG_IMAGE=tmg/isaaclab-soft TMG_REVISION=$TMG_REVISION TMG_ISAACLAB_SRC=/opt/src/IsaacLab TMG_SOFTMIMICGEN_SRC=/opt/src/SoftMimicGen \
    HF_HUB_DISABLE_TELEMETRY=1 NVIDIA_DRIVER_CAPABILITIES=all
LABEL org.opencontainers.image.revision=$TMG_REVISION
VOLUME ["/data"]
ENTRYPOINT ["tmg"]
CMD ["--help"]
