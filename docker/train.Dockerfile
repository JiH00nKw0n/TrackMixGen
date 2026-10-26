# tmg/train: python 3.12 + torch 2.8 (CUDA 12.8) + LeRobot 0.6.1 + tmg. Training loop, policy server, π0.
# Build from the repository root:  make build IMAGE=train
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ARG TMG_REVISION=unknown
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_CACHE_DIR=/root/.cache/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev git curl ca-certificates build-essential ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && ln -s /root/.local/bin/uv /usr/local/bin/uv
RUN uv venv --python python3.12 /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /opt/tmg
COPY requirements/train.txt requirements/train-torch.txt requirements/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python -r requirements/train-torch.txt \
    && uv pip install --python /opt/venv/bin/python -r requirements/train.txt
COPY pyproject.toml README.md ./
COPY tmg/ tmg/
COPY configs/ configs/
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --python /opt/venv/bin/python --no-deps -e /opt/tmg
ENV TMG_DATA_ROOT=/data TMG_IMAGE=tmg/train TMG_REVISION=$TMG_REVISION HF_HUB_DISABLE_TELEMETRY=1
LABEL org.opencontainers.image.revision=$TMG_REVISION
VOLUME ["/data"]
ENTRYPOINT ["tmg"]
CMD ["--help"]
