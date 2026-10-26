# tmg/robosuite: MuJoCo 2.3.2 + robosuite 1.4.1 + mimicgen 72bd767 + robosuite_task_zoo 74eab7f + tmg. CPU only; offscreen rendering through
# Mesa's software EGL. Build from the repository root:  make build IMAGE=robosuite
ARG PYTHON_VERSION=3.10
FROM ubuntu:22.04
ARG PYTHON_VERSION
ARG TMG_REVISION=unknown
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    UV_LINK_MODE=copy UV_CACHE_DIR=/root/.cache/uv CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN apt-get update && apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
      git curl ca-certificates build-essential cmake ffmpeg \
      libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libgles2 libglew-dev libosmesa6 libglfw3 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && ln -s /root/.local/bin/uv /usr/local/bin/uv
RUN uv venv --python python${PYTHON_VERSION} /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /opt/tmg
COPY requirements/robosuite.lock requirements/robosuite-torch.txt requirements/
# three steps: PyPI pins, torch from its own index, then the official packages from pinned checkouts without
# dependency resolution, installed in editable mode so that their data files (MimicGen's object xml assets and
# config templates, the task zoo's cabinet xml) are read from the checkouts under /opt/src.
ARG MIMICGEN_COMMIT=72bd767c255545f462e7ccfb2731f2e5d4c1d9bb
ARG ROBOMIMIC_COMMIT=d309eaecc18acf4152a830a895a6984b8ac71b05
# robosuite_task_zoo: optional MimicGen dependency for Hammer Cleanup (and Kitchen); its setup.py pins mujoco-py,
# so it is installed without dependencies.
ARG TASK_ZOO_COMMIT=74eab7f88214c21ca1ae8617c2b2f8d19718a9ed
RUN --mount=type=cache,target=/root/.cache/uv \
    grep -v -E '^(robomimic|mimicgen|robosuite_task_zoo) @' requirements/robosuite.lock > /tmp/pypi.txt \
    && uv pip install --python /opt/venv/bin/python -r /tmp/pypi.txt \
    && uv pip install --python /opt/venv/bin/python -r requirements/robosuite-torch.txt \
    && mkdir -p /opt/src && cd /opt/src \
    && git clone -q https://github.com/ARISE-Initiative/robomimic robomimic && git -C robomimic checkout -q $ROBOMIMIC_COMMIT \
    && git clone -q https://github.com/NVlabs/mimicgen mimicgen && git -C mimicgen checkout -q $MIMICGEN_COMMIT \
    && git clone -q https://github.com/ARISE-Initiative/robosuite-task-zoo robosuite_task_zoo && git -C robosuite_task_zoo checkout -q $TASK_ZOO_COMMIT \
    && rm -rf robomimic/.git mimicgen/.git robosuite_task_zoo/.git \
    && uv pip install --python /opt/venv/bin/python --no-deps -e /opt/src/robomimic -e /opt/src/mimicgen -e /opt/src/robosuite_task_zoo
# MimicGen's BlenderObject writes a per-process xml next to its ShapeNet assets at run time (mug_cleanup); the
# container runs as the host user, so that folder must be writable. Singularity adds --writable-tmpfs for it.
RUN chmod -R a+rwX /opt/src/mimicgen/mimicgen/models/robosuite/assets
COPY pyproject.toml README.md ./
COPY tmg/ tmg/
COPY configs/ configs/
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --python /opt/venv/bin/python --no-deps -e /opt/tmg
ENV TMG_DATA_ROOT=/data TMG_IMAGE=tmg/robosuite TMG_REVISION=$TMG_REVISION HF_HUB_DISABLE_TELEMETRY=1 \
    TMG_MIMICGEN_SRC=/opt/src/mimicgen
LABEL org.opencontainers.image.revision=$TMG_REVISION
VOLUME ["/data"]
ENTRYPOINT ["tmg"]
CMD ["--help"]
