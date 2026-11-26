# TrackMixGen. Every target runs inside a docker image; the repository is mounted over /opt/tmg so code and
# config edits need no rebuild, and $(DATA_ROOT) is mounted at /data.
#   make build IMAGE=robosuite
#   make sources  TASK=square
#   make generate TASK=square SCENE=official_d1 METHOD=none GENERATION=test/pilot20
#   make show     TASK=square
#   make check-scene TASK=hammer_cleanup SCENE=ours
#   make test
# CLUSTER=<name> reads runtime, image, sif, data_root and render from configs/machine/cluster/<name>.yaml (local, isaac, a100);
# RUNTIME, SIF and DATA_ROOT on the command line override it. RUNTIME=singularity runs the same commands with
# `singularity exec` on SIF (a .sif built from the docker image with `make sif ARCHIVE=<docker save tar>`).
SHELL := /bin/bash
.DEFAULT_GOAL := help
CLUSTER    ?= local
REPO       := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
GENERATOR  ?= mimicgen
# flat `key: value` lines of a config file; `null` counts as unset
config_value  = $(filter-out null,$(strip $(shell sed -n 's/^$(2):[[:space:]]*\([^\#]*\).*/\1/p' $(REPO)/configs/$(1).yaml)))
cluster_value = $(call config_value,machine/cluster/$(CLUSTER),$(1))
# every generator names the image it runs in (generator config `image: tmg/<IMAGE>`), so GENERATOR picks IMAGE unless given
IMAGE      ?= $(or $(notdir $(call config_value,experiment/generator/$(GENERATOR),image)),robosuite)
IMAGE      := $(IMAGE)
TAG        ?= latest
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
GEN_ARGS    = generator=$(GENERATOR) task=$(TASK) scene=$(SCENE) selection=$(SELECTION) method=$(METHOD) generation=$(GENERATION) cluster=$(CLUSTER) $(OVERRIDES)
REVISION   := $(shell git -C $(REPO) rev-parse --short HEAD 2>/dev/null || echo unknown)

# ---- what a simulator container needs, per image ------------------------------------------------------------------------
# The image sections below only fill these variables; DOCKER_RUN and SINGULARITY_RUN at the end turn them into the flags of
# the runtime, so a setting is written once for both runtimes:
#   CONTAINER_ENV    KEY=VALUE ...                     variables set inside the container
#   CONTAINER_BINDS  HOST_PATH:CONTAINER_PATH ...      host files and folders mounted into the container
#   CONTAINER_CUDA   CUDA_VISIBLE_DEVICES inside the container (only when CONTAINER_CUDA_SET is 1; may be empty)
#   DOCKER_GPU, DOCKER_EXTRA, SING_NV                  runtime-specific options (GPU access, shared memory)
#   WRITABLE_COPIES  HOST_DIR:IMAGE_DIR ...            image folders that need a writable host copy, filled on first use
CONTAINER_ENV   =
CONTAINER_BINDS =
CONTAINER_CUDA_SET =
WRITABLE_COPIES =
host_part  = $(word 1,$(subst :, ,$(1)))
image_part = $(word 2,$(subst :, ,$(1)))
comma  := ,

# -- Isaac Lab images (SoftMimicGen, DexMimicGen): Isaac Sim simulates and renders on the GPU in generation, resets and
# evaluation. Kit needs Vulkan; the NVIDIA runtime with NVIDIA_DRIVER_CAPABILITIES=all mounts the driver's Vulkan ICD
# (/etc/vulkan/icd.d/nvidia_icd.json) into the container, which `--gpus all` alone did not on the Isaac server (2026-10-28).
# ISAAC_GPU (a GPU index, e.g. exported by scripts/run_isaaclab_baseline.sh from its GPU variable) exposes only that GPU to
# the container, where it is device 0; empty exposes all GPUs and Isaac Sim takes device 0 of the host.
# On the shared Isaac server an empty ISAAC_GPU would put Isaac Sim on host GPU 0, which other users hold, so the targets that
# start Isaac Sim (generate and with it coefficients, verify-baseline, resets, eval, validate, evaluate, check-scene) refuse
# to run there without it (ISAAC_GPU_CHECK, the first line of their recipes; validate and evaluate stop before the policy
# server starts). The server is CLUSTER=isaac, or any CLUSTER on a host where the isaac cluster's data_root exists (a run
# that forgot CLUSTER=isaac and got the default, local).
ISAAC_GPU_CHECK = @true
ISAAC_HOST_ROOT := $(call config_value,machine/cluster/isaac,data_root)
ON_ISAAC_HOST   := $(if $(filter isaac,$(strip $(CLUSTER))),1,$(if $(ISAAC_HOST_ROOT),$(if $(wildcard $(ISAAC_HOST_ROOT)),1,),))
ifneq (,$(filter isaaclab-%,$(IMAGE)))
DOCKER_GPU   = --runtime=nvidia --gpus $(if $(ISAAC_GPU),device=$(ISAAC_GPU),all) -e NVIDIA_VISIBLE_DEVICES=$(or $(ISAAC_GPU),all) -e NVIDIA_DRIVER_CAPABILITIES=all
DOCKER_EXTRA = --shm-size 8g
SING_NV      = --nv
ifeq ($(ON_ISAAC_HOST),1)
ifeq ($(strip $(ISAAC_GPU)),)
ISAAC_GPU_CHECK = @echo "Isaac server (CLUSTER=$(strip $(CLUSTER))) with image $(IMAGE): set ISAAC_GPU to the GPU index Isaac Sim may use (e.g. ISAAC_GPU=6); without it the container sees every GPU and Isaac Sim takes host GPU 0" >&2; exit 1
endif
endif
endif

# -- CP-Gen image: the GPU also for generation and evaluation (cuRobo motion planning). CP-Gen writes into two folders of
# the image: cuRobo's task configs (it adds its straight-line cost inside the cuRobo checkout) and MimicGen's ShapeNet mug
# assets (Mug Cleanup writes a scaled mug xml next to the mesh). Both are read-only in the image (and root-owned under
# docker --user), so a writable host copy of each is mounted over the folder.
ifeq ($(IMAGE),cpgen)
DOCKER_GPU = --gpus all
SING_NV    = --nv
WRITABLE_COPIES = $(DATA_ROOT)/cpgen_writable/curobo_task:/opt/src/curobo/src/curobo/content/configs/task \
                  $(DATA_ROOT)/cpgen_writable/mimicgen_mugs:/opt/src/mimicgen/mimicgen/models/robosuite/assets/shapenet_core/mugs
CONTAINER_BINDS += $(WRITABLE_COPIES)
endif

# -- Offscreen rendering of the robosuite images (robosuite: MimicGen, cpgen: CP-Gen) in every stage that runs them
# (sources, generate, dataset, resets, eval, check-scene; for CP-Gen also the noisy-state replay and re-rendering of
# `make dataset`, which runs in the same container and inherits these variables): always on the GPU through NVIDIA EGL
# (RENDER=egl, cluster `render: egl`; about 4x faster per MimicGen image attempt than Mesa on the A100, 2026-11-21). Mesa's software rendering (RENDER=mesa) was removed on main: every pool, dataset and
# evaluation made before 2026-11-22 used it, and the two renderers do not draw the same pixels (50-61 % of the pixels
# differed, 2026-11-22), so a policy trained on such a dataset is evaluated from a checkout that still has it (e.g. branch
# skillmix-sparsemax). The container gets TMG_RENDER and every chunk spec records it; a spec without it counts as Mesa,
# so an old Mesa pool is refused rather than continued under EGL.
RENDER ?= $(or $(call cluster_value,render),egl)
ifneq (,$(filter robosuite cpgen,$(IMAGE)))
ifeq ($(RENDER),mesa)
$(error RENDER=mesa was removed on main (EGL only). Datasets made before 2026-11-22 used Mesa: evaluate policies trained on them from a checkout that still has it, e.g. branch skillmix-sparsemax)
endif
ifneq ($(RENDER),egl)
$(error RENDER must be egl, got '$(RENDER)')
endif
CONTAINER_ENV += TMG_RENDER=$(RENDER) $(if $(EGL_DEVICE_ID),MUJOCO_EGL_DEVICE_ID=$(EGL_DEVICE_ID),)

# Which EGL device. The GPU that renders is the first entry of the caller's CUDA_VISIBLE_DEVICES (the drivers' GPU variable;
# "0,3" gives 0), else 0. EGL numbers the GPUs in its own order: cluster `egl_device_of_gpu` lists the EGL index of GPU 0,
# 1, 2, ... (A100: [2, 3, 0, 1, 6, 7, 4, 5], read with EGL_CUDA_DEVICE_NV); unset means the same index (Isaac).
# EGL_DEVICE_ID on the command line overrides the lookup.
RENDER_GPU    ?= $(or $(firstword $(subst $(comma), ,$(CUDA_VISIBLE_DEVICES))),0)
EGL_MAP        = $(subst $(comma), ,$(subst [,,$(subst ],,$(call cluster_value,egl_device_of_gpu))))
EGL_DEVICE_ID ?= $(if $(EGL_MAP),$(word $(shell echo $$(( $(RENDER_GPU) + 1 ))),$(EGL_MAP)),$(RENDER_GPU))
# CUDA inside the container. robosuite asserts that MUJOCO_EGL_DEVICE_ID is listed in CUDA_VISIBLE_DEVICES whenever that is
# not empty, and an EGL index is not a GPU index on the A100. The robosuite image needs no CUDA, so its list is empty; the
# CP-Gen image keeps the rendering GPU for cuRobo (cuda:0) and tmg.core.egl.apply appends the EGL index when it differs.
CONTAINER_CUDA_SET = 1
CONTAINER_CUDA     = $(if $(filter cpgen,$(IMAGE)),$(RENDER_GPU),)
# robomimic asks egl_probe for the EGL devices, which renders a test on every GPU of the host (other users' included) and
# stalled with many chunk processes (A100, 2026-11-21); docker/egl_stub/egl_probe_init.py returns MUJOCO_EGL_DEVICE_ID only.
CONTAINER_BINDS += $(REPO)/docker/egl_stub/egl_probe_init.py:/opt/venv/lib/python3.10/site-packages/egl_probe/__init__.py
# GPU access with the graphics capability (docker). Singularity's --nv does not bring NVIDIA's EGL library (A100:
# nvidia-container-cli without the graphics capability), so cluster `nvidia_egl_dir` holds a copy of the driver's EGL
# libraries and its 10_nvidia.json vendor file (same driver version as the host), mounted into the container.
DOCKER_GPU      = --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all
SING_NV         = --nv
NVIDIA_EGL_DIR ?= $(call cluster_value,nvidia_egl_dir)
ifeq ($(RUNTIME),singularity)
ifneq (,$(NVIDIA_EGL_DIR))
CONTAINER_ENV   += LD_LIBRARY_PATH=/opt/nvegl:/.singularity.d/libs
CONTAINER_BINDS += $(NVIDIA_EGL_DIR):/opt/nvegl $(NVIDIA_EGL_DIR)/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json
endif
endif
endif

# ---- the two runtimes ---------------------------------------------------------------------------------------------------
# A writable copy that is still empty is filled from the image before the first command (PREP). docker --user: the copy
# belongs to the calling user like everything else under DATA_ROOT; without it the docker copy was root-owned, CP-Gen could
# not write its scaled mug xml there (PermissionError) and every validation of the CP-Gen Mug Cleanup policies failed
# (Isaac, 2026-11-22).
ifeq ($(RUNTIME),singularity)
fill_copy = singularity exec $(SIF) cp -r $(call image_part,$(1))/. $(call host_part,$(1))/
else
fill_copy = docker run --rm --user $(shell id -u):$(shell id -g) -v $(call host_part,$(1)):/out --entrypoint cp tmg/$(IMAGE):$(TAG) -r $(call image_part,$(1))/. /out/
endif
PREP = mkdir -p $(DATA_ROOT) $(CONTAINER_TMP) $(foreach p,$(WRITABLE_COPIES),&& { test -n "$$(ls -A $(call host_part,$(p)) 2>/dev/null)" || { mkdir -p $(call host_part,$(p)) && $(call fill_copy,$(p)); }; })
# Arena assets moved from the staging S3 mirror to production (2026-11-03): mount the patched path module into the Dex image
ARENA_PATCH = $(if $(filter isaaclab-dex,$(IMAGE)),-v $(DATA_ROOT)/arena_patch/nucleus.py:/opt/src/IsaacLab-Arena/isaaclab_arena/assets/nucleus.py:ro,)
DOCKER_RUN  = docker run --rm -i $(if $(shell tty -s && echo t),-t,) --network host $(DOCKER_GPU) $(DOCKER_EXTRA) \
              $(if $(CONTAINER_CUDA_SET),-e CUDA_VISIBLE_DEVICES=$(CONTAINER_CUDA),) $(addprefix -e ,$(CONTAINER_ENV)) \
              -v $(DATA_ROOT):/data -v $(REPO):/opt/tmg $(addprefix -v ,$(CONTAINER_BINDS)) $(ARENA_PATCH) \
              -e TMG_DATA_ROOT=/data -e HOME=/tmp -e HF_HOME=/data/.cache/huggingface \
              --user $(shell id -u):$(shell id -g) tmg/$(IMAGE):$(TAG)
# CUDA_VISIBLE_DEVICES reaches a --cleanenv container only through SINGULARITYENV_CUDA_VISIBLE_DEVICES, set for the one command
SINGULARITY_RUN = $(if $(CONTAINER_CUDA_SET),SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(CONTAINER_CUDA),) singularity exec $(SING_NV) \
              $(addprefix --env ,$(CONTAINER_ENV)) --cleanenv --no-home --writable-tmpfs --pwd /opt/tmg \
              -B $(DATA_ROOT):/data -B $(REPO):/opt/tmg -B $(CONTAINER_TMP):/tmp $(addprefix -B ,$(CONTAINER_BINDS)) \
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

.PHONY: help build sif sources generate verify-baseline fit-coefficients coefficients dataset resets train validate evaluate eval serve-detached serve-stop serve-ready print-data-root check-scene show shell test test-local lint

help:  ## this help
	@awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' $(firstword $(MAKEFILE_LIST))
	@echo; grep -E '^[a-zA-Z0-9_-]+:.*## ' $(firstword $(MAKEFILE_LIST)) | sed -E 's/^([a-zA-Z0-9_-]+):.*## /  \1|/' | awk -F '|' '{ printf "%-16s %s\n", $$1, $$2 }'

build:  ## build the docker image IMAGE (robosuite | cpgen | isaaclab-soft | isaaclab-dex (AutoData stack) | train)
	DOCKER_BUILDKIT=1 docker build -f $(REPO)/docker/$(IMAGE).Dockerfile --build-arg TMG_REVISION=$(REVISION) -t tmg/$(IMAGE):$(TAG) $(REPO)

sif:  ## build the singularity image SIF from a docker archive ARCHIVE (made with `docker save tmg/robosuite:latest -o ...`)
	mkdir -p $(dir $(SIF)) && singularity build $(SIF) docker-archive://$(ARCHIVE)

sources:  ## download and prepare the source demonstrations of TASK
	$(PREP) && $(CONTAINER_RUN) sources generator=$(GENERATOR) task=$(TASK)

generate:  ## generate one pool (TASK, SCENE, SELECTION=official|ours, METHOD, GENERATION, OVERRIDES); builds the SkillMix rule file first when METHOD needs a missing one
	$(ISAAC_GPU_CHECK)
	@METHOD=$(strip $(METHOD)) OVERRIDES="$(OVERRIDES)" GENERATOR=$(GENERATOR) IMAGE=$(IMAGE) TASK=$(TASK) SCENE=$(strip $(SCENE)) CLUSTER=$(CLUSTER) scripts/ensure_coefficients.sh
	$(PREP) && $(CONTAINER_RUN) generate $(GEN_ARGS)

verify-baseline:  ## official generator vs adapter with method=none on one chunk; must be identical
	$(ISAAC_GPU_CHECK)
	$(PREP) && $(CONTAINER_RUN) verify_baseline generator=$(GENERATOR) task=$(TASK) scene=$(SCENE) selection=$(SELECTION) generation=$(GENERATION) cluster=$(CLUSTER) $(OVERRIDES)

fit-coefficients:  ## fit TASK's SkillMix rule on its randmix pool (skillmix_randmix/pool_randmix) and unmix pool (none/GENERATION), SELECTION=ours GENERATION=pool_unmix; writes coefficients/<task>.yaml
	$(PREP) && $(CONTAINER_RUN) fit_coefficients generator=$(GENERATOR) task=$(TASK) scene=$(SCENE) selection=$(SELECTION) generation=$(GENERATION) cluster=$(CLUSTER) $(OVERRIDES)

coefficients:  ## build TASK's SkillMix rule file from scratch when missing (the unmix and randmix pools of GENERATION=pool_unmix, then fit-coefficients)
	GENERATOR=$(GENERATOR) IMAGE=$(IMAGE) TASK=$(TASK) SCENE=$(strip $(SCENE)) CLUSTER=$(CLUSTER) scripts/ensure_coefficients.sh

dataset:  ## write a training dataset from the finished pool (TASK, SCENE, METHOD, GENERATION, OBSERVATION=image|low_dim; OVERRIDES: size=... seed=...)
	$(PREP) && $(CONTAINER_RUN) dataset $(GEN_ARGS) observation=$(OBSERVATION)

resets:  ## draw the fixed evaluation scenes of (TASK, SCENE) in the simulator image
	$(ISAAC_GPU_CHECK)
	$(PREP) && $(CONTAINER_RUN) resets generator=$(GENERATOR) task=$(TASK) scene=$(SCENE) cluster=$(CLUSTER) $(OVERRIDES)

train:  ## train POLICY on the dataset of (TASK, SCENE, METHOD, GENERATION, OBSERVATION) in the train image (GPU)
	$(PREP) && $(TRAIN_RUN) train $(GEN_ARGS) observation=$(OBSERVATION) policy=$(POLICY) seed=$(SEED)

select-checkpoint:  ## choose the checkpoint of the run (TASK, SCENE, METHOD, GENERATION, OBSERVATION, POLICY, SEED) by training.selection; writes SELECTED.json
	$(TRAIN_RUN) select_checkpoint $(GEN_ARGS) observation=$(OBSERVATION) policy=$(POLICY) seed=$(SEED)

validate:  ## evaluate checkpoint CKPT of run RUN on the validation scenes: policy server (train image) + rollouts (simulator image)
	$(ISAAC_GPU_CHECK)
	RUN=$(RUN) CKPT=$(CKPT) TASK=$(TASK) SCENE=$(SCENE) OBSERVATION=$(OBSERVATION) CLUSTER=$(CLUSTER) SPLIT=val $(REPO)/scripts/evaluate_checkpoint.sh

evaluate:  ## evaluate checkpoint CKPT on the test scenes (SPLIT=test)
	$(ISAAC_GPU_CHECK)
	RUN=$(RUN) CKPT=$(CKPT) TASK=$(TASK) SCENE=$(SCENE) OBSERVATION=$(OBSERVATION) CLUSTER=$(CLUSTER) SPLIT=$(SPLIT) $(REPO)/scripts/evaluate_checkpoint.sh

eval:  ## play the fixed scenes against a running policy server (OBSERVATION as trained; OVERRIDES: checkpoint=... split=val|test ...)
	$(ISAAC_GPU_CHECK)
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
	$(ISAAC_GPU_CHECK)
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
