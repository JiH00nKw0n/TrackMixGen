# tmg/isaaclab-dex: the DexMimicGen stack. NVIDIA AutoData (isaac-sim/AutoData @ 835f4aab, the DexMimicGen rewrite) on
# Isaac Lab-Arena @ 481f5ae5 with its nested Isaac Lab 3.0 (28a386f0) and pip Isaac Sim 6.1.0, python 3.12, torch 2.11
# (CUDA 12.8). Runs the three GR1 tasks of the paper plan: Pick Place (Isaac Lab Mimic's environment, registered by
# isaaclab_mimic of the nested Isaac Lab) and the two Arena tasks Microwave Opening (gr1_open_microwave) and Bottle
# Placement and Door Closing (put_item_in_fridge_and_close_door). The package set is Arena's own uv lock (what AutoData's
# conda_installer.sh installs), so every version equals the upstream validated stack. GPU required.
# Until 2026-10-29 this name was an Isaac Lab 2.x (3c6e67bb, Isaac Sim 5.1) image running isaaclab_mimic directly.
# Build from the repository root (long: Isaac Sim is about 10 GB):  make build IMAGE=isaaclab-dex
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ARG TMG_REVISION=unknown
ARG ARENA_COMMIT=481f5ae5f19df7bc5d24a17240fc0d96ac81eaf7
ARG AUTODATA_COMMIT=835f4aab0fff759a789586454d939f12a9a40997
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    UV_LINK_MODE=copy UV_CACHE_DIR=/root/.cache/uv OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev git git-lfs curl ca-certificates build-essential cmake ninja-build \
      libgl1 libglu1-mesa libxt6 libxrandr2 libxinerama1 libxcursor1 libxi6 libglib2.0-0 libsm6 libxext6 ffmpeg \
      libvulkan1 vulkan-tools libegl1 libgbm1 libdrm2 libgles2 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && ln -s /root/.local/bin/uv /usr/local/bin/uv
RUN uv venv --python python3.12 --seed /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# the submodule URLs are ssh; read them over https
RUN git config --global url."https://github.com/".insteadOf git@github.com: \
    && mkdir -p /opt/src && cd /opt/src \
    && git clone -q https://github.com/isaac-sim/IsaacLab-Arena IsaacLab-Arena && git -C IsaacLab-Arena checkout -q $ARENA_COMMIT \
    && git -C IsaacLab-Arena submodule update --init submodules/IsaacLab \
    && git clone -q https://github.com/isaac-sim/AutoData AutoData && git -C AutoData checkout -q $AUTODATA_COMMIT
# Arena's lock = the validated Sim/Lab/torch stack (isaacsim 6.1.0.0 pip wheels from pypi.nvidia.com, editable Isaac Lab
# packages from the submodule, Arena editable). Same two exports as AutoData's conda_installer.sh.
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /opt/src/IsaacLab-Arena \
    && uv export --frozen --no-dev --format pylock.toml --output-file pylock.toml --quiet \
    && uv export --frozen --no-dev --format requirements.txt --no-emit-local --no-hashes --output-file constraints.txt --quiet \
    && uv pip install --no-config --python /opt/venv/bin/python --requirements pylock.toml \
    && uv pip install --no-config --python /opt/venv/bin/python --constraint constraints.txt --editable /opt/src/AutoData
# Kit (pip isaacsim) writes its cache, data and logs under the package folder; the containers run as the host user
RUN K=/opt/venv/lib/python3.12/site-packages/isaacsim/kit && mkdir -p $K/cache $K/data $K/logs && chmod -R a+rwX $K/cache $K/data $K/logs
WORKDIR /opt/tmg
COPY pyproject.toml README.md ./
COPY tmg/ tmg/
COPY configs/ configs/
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-config --python /opt/venv/bin/python --no-deps -e /opt/tmg \
    && uv pip install --no-config --python /opt/venv/bin/python --constraint /opt/src/IsaacLab-Arena/constraints.txt \
       'hydra-core>=1.3,<1.4' 'omegaconf>=2.3,<2.4' pyyaml scipy msgpack websockets h5py huggingface_hub pillow
ENV TMG_DATA_ROOT=/data TMG_IMAGE=tmg/isaaclab-dex TMG_REVISION=$TMG_REVISION \
    TMG_AUTODATA_SRC=/opt/src/AutoData TMG_ARENA_SRC=/opt/src/IsaacLab-Arena TMG_ISAACLAB_SRC=/opt/src/IsaacLab-Arena/submodules/IsaacLab \
    HF_HUB_DISABLE_TELEMETRY=1 NVIDIA_DRIVER_CAPABILITIES=all
LABEL org.opencontainers.image.revision=$TMG_REVISION
VOLUME ["/data"]
ENTRYPOINT ["tmg"]
CMD ["--help"]
