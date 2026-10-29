# TrackMixGen. Every target runs inside a docker image; the repository is mounted over /opt/tmg so code and
# config edits need no rebuild, and $(DATA_ROOT) is mounted at /data.
#   make build IMAGE=robosuite
#   make sources  TASK=square
#   make generate TASK=square SCENE=official_d1 METHOD=none GENERATION=test/pilot20
#   make show     TASK=square
#   make check-scene TASK=hammer_cleanup SCENE=ours
#   make test
# CLUSTER=<name> reads runtime, image, sif and data_root from configs/machine/cluster/<name>.yaml (local, isaac, a100);
# RUNTIME, SIF and DATA_ROOT on the command line override it. RUNTIME=singularity runs the same commands with
# `singularity exec` on SIF (a .sif built from the docker image with `make sif ARCHIVE=<docker save tar>`).
SHELL := /bin/bash
.DEFAULT_GOAL := help
REPO       := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
IMAGE      ?= robosuite
GENERATOR  ?= mimicgen
TAG        ?= latest
CLUSTER    ?= local
# flat `key: value` lines of the cluster file; `null` counts as unset
cluster_value = $(filter-out null,$(strip $(shell sed -n 's/^$(1):[[:space:]]*\([^\#]*\).*/\1/p' $(REPO)/configs/machine/cluster/$(CLUSTER).yaml)))
DATA_ROOT  ?= $(or $(TMG_DATA_ROOT),$(call cluster_value,data_root),$(HOME)/tmg_data)
TASK       ?= square
SCENE      ?= official_d1
SELECTION  ?= official     # source-demonstration selection rule: official | ours (configs/experiment/selection)
METHOD     ?= none
GENERATION ?= test/pilot20
POLICY     ?= bc_rnn
OBSERVATION ?= image
SEED       ?= 1
SPLIT      ?= test
RUN        ?=
CKPT       ?=
TRAIN_IMAGE ?= train
OVERRIDES  ?=
RUNTIME    ?= $(or $(call cluster_value,runtime),docker)
SIF        ?= $(or $(call cluster_value,sif),$(DATA_ROOT)/images/tmg_$(IMAGE).sif)
ARCHIVE    ?= $(DATA_ROOT)/images/tmg_$(IMAGE).tar
# host folder bound to /tmp inside the Singularity containers: the session overlay that backs /tmp is capped at
# 64 MB (sessiondir max size) and MimicGen assembles each chunk's demo file through a temporary directory there
CONTAINER_TMP ?= $(or $(call cluster_value,container_tmp),$(DATA_ROOT)/tmp/container_tmp)
PREP        = mkdir -p $(DATA_ROOT) $(CONTAINER_TMP) $(PREP_EXTRA)
GEN_ARGS    = generator=$(GENERATOR) task=$(TASK) scene=$(SCENE) selection=$(SELECTION) method=$(METHOD) generation=$(GENERATION) cluster=$(CLUSTER) $(OVERRIDES)
REVISION   := $(shell git -C $(REPO) rev-parse --short HEAD 2>/dev/null || echo unknown)
# the Isaac Lab images (SoftMimicGen, DexMimicGen) run Isaac Sim: GPU for generation, resets and evaluation. Kit needs
# Vulkan; the NVIDIA runtime with NVIDIA_DRIVER_CAPABILITIES=all mounts the driver's Vulkan ICD (/etc/vulkan/icd.d/
# nvidia_icd.json) into the container, which  alone did not on the Isaac server (2026-10-28)
ifneq (,$(filter isaaclab-%,$(IMAGE)))
DOCKER_GPU = --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all
SING_NV    = --nv
DOCKER_VOL_EXTRA = --shm-size 8g
endif
# robosuite (1.4 and 1.5) picks the offscreen EGL rendering device by index; on a host whose /dev/dri the user cannot
# open (A100 under Singularity: Mesa software rendering, device index = cluster `egl_device_id`) the index must be set
# explicitly for both robosuite images, or image pools and image evaluations fail with "Cannot initialize a EGL device
# display" (2026-10-29). Docker hosts leave egl_device_id unset: Mesa inside the container finds its software device itself.
ifneq (,$(filter robosuite cpgen,$(IMAGE)))
EGL_DEVICE_ID ?= $(call cluster_value,egl_device_id)
SING_ENV_EXTRA = $(if $(EGL_DEVICE_ID),--env MUJOCO_EGL_DEVICE_ID=$(EGL_DEVICE_ID),)
endif
# the CP-Gen image needs the GPU (cuRobo motion planning) also for generation and evaluation
ifeq ($(IMAGE),cpgen)
DOCKER_GPU = --gpus all
SING_NV    = --nv
# CP-Gen writes into two folders of the image: cuRobo's task configs (it adds its straight-line cost inside the cuRobo
# checkout) and MimicGen's ShapeNet mug assets (Mug Cleanup writes a scaled mug xml next to the mesh). Both are read-only
# in the image (and root-owned under docker --user), so a writable host copy of each, filled from the image on first use,
# is mounted over the folder. Pairs are IMAGE_DIR:HOST_DIR.
CUROBO_TASK_DIR   = /opt/src/curobo/src/curobo/content/configs/task
MIMICGEN_MUGS_DIR = /opt/src/mimicgen/mimicgen/models/robosuite/assets/shapenet_core/mugs
CPGEN_WRITABLE    = $(CUROBO_TASK_DIR):$(DATA_ROOT)/cpgen_writable/curobo_task $(MIMICGEN_MUGS_DIR):$(DATA_ROOT)/cpgen_writable/mimicgen_mugs
img_dir  = $(word 1,$(subst :, ,$(1)))
host_dir = $(word 2,$(subst :, ,$(1)))
SING_BIND_EXTRA  = $(foreach p,$(CPGEN_WRITABLE),-B $(call host_dir,$(p)):$(call img_dir,$(p)))
DOCKER_VOL_EXTRA = $(foreach p,$(CPGEN_WRITABLE),-v $(call host_dir,$(p)):$(call img_dir,$(p)))
ifeq ($(RUNTIME),singularity)
writable_fill = singularity exec $(SIF) cp -r $(call img_dir,$(1))/. $(call host_dir,$(1))/
else
writable_fill = docker run --rm -v $(call host_dir,$(1)):/out --entrypoint cp tmg/$(IMAGE):$(TAG) -r $(call img_dir,$(1))/. /out/
endif
PREP_EXTRA = $(foreach p,$(CPGEN_WRITABLE),&& { test -n "$$(ls -A $(call host_dir,$(p)) 2>/dev/null)" || { mkdir -p $(call host_dir,$(p)) && $(call writable_fill,$(p)); }; })
endif
DOCKER_RUN  = docker run --rm -i $(if $(shell tty -s && echo t),-t,) --network host $(DOCKER_GPU) \
              -v $(DATA_ROOT):/data -v $(REPO):/opt/tmg $(DOCKER_VOL_EXTRA) \
              -e TMG_DATA_ROOT=/data -e HOME=/tmp -e HF_HOME=/data/.cache/huggingface \
              --user $(shell id -u):$(shell id -g) tmg/$(IMAGE):$(TAG)
SINGULARITY_RUN = singularity exec $(SING_NV) $(SING_ENV_EXTRA) --cleanenv --no-home --writable-tmpfs --pwd /opt/tmg -B $(DATA_ROOT):/data -B $(REPO):/opt/tmg -B $(CONTAINER_TMP):/tmp $(SING_BIND_EXTRA) \
              --env TMG_DATA_ROOT=/data,HOME=/tmp,HF_HOME=/data/.cache/huggingface,MUJOCO_GL=egl,PYOPENGL_PLATFORM=egl $(SIF) tmg
# the train image (python 3.12, torch, lerobot) runs with the GPU; the data root and repo are mounted the same way
TRAIN_SIF   ?= $(or $(call cluster_value,train_sif),$(DATA_ROOT)/images/tmg_$(TRAIN_IMAGE).sif)
# CUDA_VISIBLE_DEVICES in the caller's environment (the drivers' GPU variable) selects the GPU inside the train containers
DOCKER_GPU_PICK = $(if $(CUDA_VISIBLE_DEVICES),-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES),)
DOCKER_TRAIN = docker run --rm -i $(if $(shell tty -s && echo t),-t,) --gpus all --shm-size 16g $(DOCKER_GPU_PICK) \
              -v $(DATA_ROOT):/data -v $(REPO):/opt/tmg \
              -e TMG_DATA_ROOT=/data -e HOME=/tmp -e HF_HOME=/data/.cache/huggingface \
              --user $(shell id -u):$(shell id -g) tmg/$(TRAIN_IMAGE):$(TAG)
SINGULARITY_TRAIN = singularity exec --nv --cleanenv --no-home --writable-tmpfs --pwd /opt/tmg -B $(DATA_ROOT):/data -B $(REPO):/opt/tmg -B $(CONTAINER_TMP):/tmp \
              --env TMG_DATA_ROOT=/data,HOME=/tmp,HF_HOME=/data/.cache/huggingface $(TRAIN_SIF) tmg
ifeq ($(RUNTIME),singularity)
CONTAINER_RUN = $(SINGULARITY_RUN)
TRAIN_RUN = $(SINGULARITY_TRAIN)
else ifeq ($(RUNTIME),native)
NATIVE = python3 $(REPO)/scripts/native_runtime.py
CONTAINER_RUN = $(NATIVE) run --image $(IMAGE) --data-root "$(DATA_ROOT)" --
TRAIN_RUN = $(NATIVE) run --image train --data-root "$(DATA_ROOT)" --
export TMG_DATA_ROOT := $(DATA_ROOT)
export TMG_NATIVE_ROOT
else
CONTAINER_RUN = $(DOCKER_RUN)
TRAIN_RUN = $(DOCKER_TRAIN)
endif
CONTAINER_RUN_HOSTNET = true

.PHONY: help build sif sources generate verify-baseline rank-sources dataset resets train validate evaluate eval serve-detached serve-stop serve-ready print-data-root check-scene show shell test test-local lint

help:  ## this help
	@awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' $(firstword $(MAKEFILE_LIST))
	@echo; grep -E '^[a-zA-Z0-9_-]+:.*## ' $(firstword $(MAKEFILE_LIST)) | sed -E 's/^([a-zA-Z0-9_-]+):.*## /  \1|/' | awk -F '|' '{ printf "%-16s %s\n", $$1, $$2 }'

build:  ## build the docker image IMAGE (robosuite | cpgen | isaaclab-soft | isaaclab-dex (AutoData stack) | train)
	DOCKER_BUILDKIT=1 docker build -f $(REPO)/docker/$(IMAGE).Dockerfile --build-arg TMG_REVISION=$(REVISION) -t tmg/$(IMAGE):$(TAG) $(REPO)

sif:  ## build the singularity image SIF from a docker archive ARCHIVE (made with `docker save tmg/robosuite:latest -o ...`)
	mkdir -p $(dir $(SIF)) && singularity build $(SIF) docker-archive://$(ARCHIVE)

sources:  ## download and prepare the source demonstrations of TASK
	$(PREP) && $(CONTAINER_RUN) sources generator=$(GENERATOR) task=$(TASK)

generate:  ## generate one pool (TASK, SCENE, SELECTION=official|ours, METHOD, GENERATION, OVERRIDES)
	$(PREP) && $(CONTAINER_RUN) generate $(GEN_ARGS)

verify-baseline:  ## official generator vs adapter with method=none on one chunk; must be identical
	$(PREP) && $(CONTAINER_RUN) verify_baseline generator=$(GENERATOR) task=$(TASK) scene=$(SCENE) selection=$(SELECTION) generation=$(GENERATION) cluster=$(CLUSTER) $(OVERRIDES)

rank-sources:  ## order TASK's source demonstrations from its baseline pool (GENERATION=rank_human_demos, SELECTION as generated) and write the ranking next to the generator config
	$(PREP) && $(CONTAINER_RUN) rank_sources generator=$(GENERATOR) task=$(TASK) scene=$(SCENE) selection=$(SELECTION) generation=$(GENERATION) cluster=$(CLUSTER) $(OVERRIDES)

dataset:  ## write a training dataset from the finished pool (TASK, SCENE, METHOD, GENERATION, OBSERVATION=image|low_dim; OVERRIDES: size=... seed=...)
	$(PREP) && $(CONTAINER_RUN) dataset $(GEN_ARGS) observation=$(OBSERVATION)

resets:  ## draw the fixed evaluation scenes of (TASK, SCENE) in the simulator image
	$(PREP) && $(CONTAINER_RUN) resets generator=$(GENERATOR) task=$(TASK) scene=$(SCENE) cluster=$(CLUSTER) $(OVERRIDES)

train:  ## train POLICY on the dataset of (TASK, SCENE, METHOD, GENERATION, OBSERVATION) in the train image (GPU)
	$(PREP) && $(TRAIN_RUN) train $(GEN_ARGS) observation=$(OBSERVATION) policy=$(POLICY) seed=$(SEED)

select-checkpoint:  ## choose the checkpoint of the run (TASK, SCENE, METHOD, GENERATION, OBSERVATION, POLICY, SEED) by training.selection; writes SELECTED.json
	$(TRAIN_RUN) select_checkpoint $(GEN_ARGS) observation=$(OBSERVATION) policy=$(POLICY) seed=$(SEED)

validate:  ## evaluate checkpoint CKPT of run RUN on the validation scenes: policy server (train image) + rollouts (simulator image)
	RUN=$(RUN) CKPT=$(CKPT) TASK=$(TASK) SCENE=$(SCENE) OBSERVATION=$(OBSERVATION) CLUSTER=$(CLUSTER) SPLIT=val $(REPO)/scripts/evaluate_checkpoint.sh

evaluate:  ## evaluate checkpoint CKPT on the test scenes (SPLIT=test)
	RUN=$(RUN) CKPT=$(CKPT) TASK=$(TASK) SCENE=$(SCENE) OBSERVATION=$(OBSERVATION) CLUSTER=$(CLUSTER) SPLIT=$(SPLIT) $(REPO)/scripts/evaluate_checkpoint.sh

eval:  ## play the fixed scenes against a running policy server (OBSERVATION as trained; OVERRIDES: checkpoint=... split=val|test ...)
	$(PREP) && $(CONTAINER_RUN) eval generator=$(GENERATOR) task=$(TASK) scene=$(SCENE) observation=$(OBSERVATION) cluster=$(CLUSTER) $(OVERRIDES)

PORT        ?= 8765
SERVER_NAME ?= tmg-serve-$(PORT)
serve-detached:  ## start the policy server for CKPT (container path) in the background (docker: named container; singularity: setsid+nohup; refuses a busy port)
ifeq ($(RUNTIME),singularity)
	@python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(1 if s.connect_ex(('127.0.0.1', $(PORT)))==0 else 0)" \
	  || { echo "serve-detached: port $(PORT) is already in use (a server is still running there)"; exit 1; }
	$(PREP); setsid nohup $(SINGULARITY_TRAIN) serve checkpoint=$(CKPT) server.port=$(PORT) > $(DATA_ROOT)/$(SERVER_NAME).log 2>&1 & echo $$! > $(DATA_ROOT)/$(SERVER_NAME).pid
else ifeq ($(RUNTIME),native)
	$(NATIVE) serve-start --data-root "$(DATA_ROOT)" --name "$(SERVER_NAME)" --port $(PORT) --checkpoint "$(CKPT)"
else
	docker run -d --rm --name $(SERVER_NAME) --gpus all --shm-size 16g --network host $(DOCKER_GPU_PICK) -v $(DATA_ROOT):/data -v $(REPO):/opt/tmg \
	  -e TMG_DATA_ROOT=/data -e HOME=/tmp --user $(shell id -u):$(shell id -g) tmg/$(TRAIN_IMAGE):$(TAG) serve checkpoint=$(CKPT) server.port=$(PORT) >/dev/null
endif

serve-stop:  ## stop the background policy server SERVER_NAME (singularity: its session's process group (the pid file names the setsid leader), plus any server left on PORT)
ifeq ($(RUNTIME),singularity)
	-kill -- -$$(cat $(DATA_ROOT)/$(SERVER_NAME).pid) 2>/dev/null; pkill -u $$(id -u) -f "serve checkpoint=.* server.port=$(PORT)$$" 2>/dev/null; rm -f $(DATA_ROOT)/$(SERVER_NAME).pid
else ifeq ($(RUNTIME),native)
	$(NATIVE) serve-stop --data-root "$(DATA_ROOT)" --name "$(SERVER_NAME)"
else
	-docker stop $(SERVER_NAME) >/dev/null 2>&1
endif

serve-ready:  ## exit 0 when the policy server answers on PORT
	$(CONTAINER_RUN_HOSTNET) --help >/dev/null 2>&1 && python3 -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1', $(PORT)))==0 else 1)"

print-data-root:  ## print the data root of CLUSTER (used by scripts)
	@echo $(DATA_ROOT)

print-runtime:
	@echo $(RUNTIME)

print-runtime-data-root:
	@echo $(if $(filter native,$(RUNTIME)),$(DATA_ROOT),/data)

check-scene:  ## reset the (TASK, SCENE) environment many times and report placement problems (OVERRIDES: resets=...)
	$(PREP) && $(CONTAINER_RUN) check_scene generator=$(GENERATOR) task=$(TASK) scene=$(SCENE) $(OVERRIDES)

show:  ## print the composed generate config
	$(CONTAINER_RUN) show generate $(GEN_ARGS)

shell:  ## interactive shell inside the image
	$(PREP) && docker run --rm -it -v $(DATA_ROOT):/data -v $(REPO):/opt/tmg -e TMG_DATA_ROOT=/data --entrypoint bash tmg/$(IMAGE):$(TAG)

test:  ## unit tests inside the image (simulator available)
ifeq ($(RUNTIME),singularity)
	singularity exec --cleanenv --no-home --writable-tmpfs --pwd /opt/tmg -B $(REPO):/opt/tmg $(SIF) python -m pytest -q /opt/tmg/tests
else
	docker run --rm -v $(REPO):/opt/tmg --entrypoint python tmg/$(IMAGE):$(TAG) -m pytest -q /opt/tmg/tests
endif

test-local:  ## unit tests on this machine (no simulator): needs `uv sync` or pip install -e .[dev]
	cd $(REPO) && python -m pytest -q tests/unit

lint:
	cd $(REPO) && ruff check tmg tests
