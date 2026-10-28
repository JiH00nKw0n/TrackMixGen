# tmg/isaaclab-dex: Isaac Sim 5.1 (pip) + Isaac Lab @ 3c6e67bb (isaaclab_mimic with the GR1 tasks) + tmg. GPU required.
# Pins that the 2026-10 reproduction needed: robomimic v0.4.0, osqp 0.6.7.post3 with qpsolvers 4.3.1 (newer qpsolvers
# import a SolverStatus that this OSQP lacks), pinocchio through Isaac Lab's `--enable_pinocchio`.
# Build from the repository root:  make build IMAGE=isaaclab-dex
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ARG TMG_REVISION=unknown
ARG ISAACLAB_COMMIT=3c6e67bb5c7ada942a6d1884ab69338f57596f77
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES \
    UV_LINK_MODE=copy UV_CACHE_DIR=/root/.cache/uv OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev git curl ca-certificates build-essential cmake ninja-build \
      libgl1 libglu1-mesa libxt6 libxrandr2 libxinerama1 libxcursor1 libxi6 libglib2.0-0 libsm6 libxext6 ffmpeg \
      libvulkan1 vulkan-tools libegl1 libgbm1 libdrm2 libgles2 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && ln -s /root/.local/bin/uv /usr/local/bin/uv
RUN uv venv --python python3.11 --seed /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python 'pip<26' 'setuptools<77' wheel toml ninja 'cmake<4' \
    && uv pip install --python /opt/venv/bin/python --index-strategy unsafe-best-match --extra-index-url https://pypi.nvidia.com 'isaacsim[all,extscache]==5.1.0' \
    && uv pip install --python /opt/venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0
RUN mkdir -p /opt/src && cd /opt/src \
    && git clone -q https://github.com/isaac-sim/IsaacLab IsaacLab && git -C IsaacLab checkout -q $ISAACLAB_COMMIT
RUN --mount=type=cache,target=/root/.cache/uv \
    for p in isaaclab isaaclab_assets isaaclab_tasks isaaclab_mimic; do uv pip install --python /opt/venv/bin/python --no-build-isolation -e /opt/src/IsaacLab/source/$p; done \
    && uv pip install --python /opt/venv/bin/python h5py 'robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0' \
    && uv pip install --python /opt/venv/bin/python 'osqp==0.6.7.post3' 'qpsolvers==4.3.1' \
    && uv pip install --python /opt/venv/bin/python --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' 'torchaudio==2.7.0+cu128'
# Pillow follows Isaac Sim's exact pin (isaacsim-kernel 5.1: Pillow==11.3.0). Isaac Sim also pins packaging, click, idna and
# psutil exactly, but those old versions break huggingface_hub and httpx (the source downloads), so they stay at the versions
# the other packages resolved; `pip check` lists those four isaacsim lines, which the 2026-10-28 generation run showed harmless.
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --python /opt/venv/bin/python pillow==11.3.0
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
ENV TMG_DATA_ROOT=/data TMG_IMAGE=tmg/isaaclab-dex TMG_REVISION=$TMG_REVISION TMG_ISAACLAB_SRC=/opt/src/IsaacLab \
    HF_HUB_DISABLE_TELEMETRY=1 NVIDIA_DRIVER_CAPABILITIES=all
LABEL org.opencontainers.image.revision=$TMG_REVISION
VOLUME ["/data"]
ENTRYPOINT ["tmg"]
CMD ["--help"]
