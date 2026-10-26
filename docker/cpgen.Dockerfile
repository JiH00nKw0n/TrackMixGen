# tmg/cpgen: CP-Gen (kevin-thankyou-lin/cpgen @ 1e7c1fc4) on robosuite f687ae7c with MuJoCo 3.2.6, cuRobo v0.7.8
# (GPU, CUDA 12.8 toolkit for its extension build), warp-lang 1.6.0 (cuRobo needs warp.torch), pytorch3d v0.7.7,
# MimicGen (environments only, no dependencies) and tmg. Pins and order follow the 2026-10 reproduction's bootstrap.
# Build from the repository root (takes long: cuRobo and pytorch3d compile):  make build IMAGE=cpgen
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ARG TMG_REVISION=unknown
ARG CPGEN_COMMIT=1e7c1fc4a467c7678c8542ca762e332d1757d8d5
ARG CPGEN_ENVS_COMMIT=c4eea73e
ARG ROBOSUITE_COMMIT=f687ae7c2f30a2e6e2dcfc31ed46504e91039417
ARG MIMICGEN_COMMIT=72bd767c255545f462e7ccfb2731f2e5d4c1d9bb
ARG TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0"
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    UV_LINK_MODE=copy UV_CACHE_DIR=/root/.cache/uv PYTORCH3D_NO_EXTENSION=1 MAX_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-dev git curl ca-certificates build-essential cmake ninja-build ffmpeg \
      libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libgles2 libglew-dev libosmesa6 libglfw3 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && ln -s /root/.local/bin/uv /usr/local/bin/uv
RUN uv venv --python python3.10 --seed /opt/venv
ENV PATH=/opt/venv/bin:$PATH CUDA_HOME=/usr/local/cuda
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python 'pip<26' 'setuptools<70' wheel ninja 'setuptools_scm==6.4.2' \
    && uv pip install --python /opt/venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0 \
    && uv pip install --python /opt/venv/bin/python numpy==1.26.4 mujoco==3.2.6 warp-lang==1.6.0
RUN mkdir -p /opt/src && cd /opt/src \
    && git clone -q https://github.com/ARISE-Initiative/robosuite robosuite && git -C robosuite checkout -q $ROBOSUITE_COMMIT \
    && sed -i 's/mujoco[=><][^"'"'"']*/mujoco==3.2.6/' robosuite/setup.py \
    && git clone -q https://github.com/kevin-thankyou-lin/cpgen cpgen && git -C cpgen checkout -q $CPGEN_COMMIT \
    && git clone -q https://github.com/kevin-thankyou-lin/cpgen-envs cpgen_envs && git -C cpgen_envs checkout -q $CPGEN_ENVS_COMMIT \
    && git clone -q https://github.com/NVlabs/mimicgen mimicgen && git -C mimicgen checkout -q $MIMICGEN_COMMIT \
    && git clone -q -b v0.7.8 --depth 1 https://github.com/NVlabs/curobo curobo
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python -e /opt/src/robosuite \
    && uv pip install --python /opt/venv/bin/python 'gym==0.25.2' gymnasium mink tyro lxml trimesh usd-core xmltodict psutil h5py \
       'robomimic @ git+https://github.com/ARISE-Initiative/robomimic@v0.4.0' egl_probe \
    && uv pip install --python /opt/venv/bin/python --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7' \
    && uv pip install --python /opt/venv/bin/python --no-deps -e /opt/src/mimicgen -e /opt/src/cpgen -e /opt/src/cpgen_envs \
    && uv pip install --python /opt/venv/bin/python --no-build-isolation -e /opt/src/curobo
# imported by demo_aug.generate at load time (not needed for generation itself): the openai client of the LLM
# segmentation utilities, wandb, and robosuite_task_zoo for cpgen_envs' kitchen environment
ARG TASK_ZOO_COMMIT=74eab7f
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone -q https://github.com/ARISE-Initiative/robosuite-task-zoo /opt/src/robosuite_task_zoo && git -C /opt/src/robosuite_task_zoo checkout -q $TASK_ZOO_COMMIT \
    && uv pip install --python /opt/venv/bin/python --no-deps -e /opt/src/robosuite_task_zoo \
    && uv pip install --python /opt/venv/bin/python openai wandb
WORKDIR /opt/tmg
COPY pyproject.toml README.md ./
COPY tmg/ tmg/
COPY configs/ configs/
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --python /opt/venv/bin/python --no-deps -e /opt/tmg \
    && uv pip install --python /opt/venv/bin/python 'hydra-core>=1.3,<1.4' 'omegaconf>=2.3,<2.4' pyyaml scipy msgpack websockets
ENV TMG_DATA_ROOT=/data TMG_IMAGE=tmg/cpgen TMG_REVISION=$TMG_REVISION TMG_CPGEN_SRC=/opt/src/cpgen TMG_MIMICGEN_SRC=/opt/src/mimicgen \
    HF_HUB_DISABLE_TELEMETRY=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
LABEL org.opencontainers.image.revision=$TMG_REVISION
VOLUME ["/data"]
ENTRYPOINT ["tmg"]
CMD ["--help"]
