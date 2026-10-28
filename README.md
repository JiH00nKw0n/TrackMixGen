# TrackMixGen

TrackMixGen runs synthetic demonstration generators with and without the TrackMix method and measures the effect
on trained policies. The method has two parts: guided source mixing (a low-success source segment is blended
with a compatible high-success one) and error-adaptive path following (the executed path advances by the
measured tracking error, with pre-noise actions kept as labels). The official generator code is never copied or
edited: at run time the code reads the source of the official functions, checks their hashes and grafts one
call per anchor into their syntax trees.

Generators: MimicGen (robosuite), SoftMimicGen and DexMimicGen (Isaac Lab), CP-Gen (robosuite with cuRobo). The Isaac Lab
baseline runs end to end on one GPU machine with `scripts/run_isaaclab_baseline.sh`; `docs/ISAACLAB_BASELINE_RUNBOOK.md` is the hand-off guide.
Pipeline: source demonstrations, generation pools, source rankings, datasets, policies (BC-RNN, Diffusion
Policy, π0 through LeRobot 0.6.1), fixed-scene evaluation through a policy server. Every command composes one
config and writes one output folder with the resolved config and a provenance record.

This document is written to the rules of ASD-STE100 (Simplified Technical English). Four documents describe the
repository: this file (structure and how to run), `tmg/README.md` (the package, folder by folder and module by
module), `configs/README.md` (the configuration layers and groups) and `docs/GENERATORS.md` (how each official
generator is grafted; Korean). `docs/MIXUP_EXPLORATION.md` is the plan for exploring the mixing design (Korean).

## 1. Code structure

`tmg/README.md` describes every folder and module of the package. In short:

The package is `tmg`. Each folder has one task. Each module, class and function has a docstring that gives its
logic, its inputs and its outputs.

| Path | What it does |
|---|---|
| `tmg/cli.py` | Command line entry point. `tmg <stage> key=value ...` composes the config of the stage and runs it. `tmg show <stage> ...` prints the composed config. |
| `tmg/config/` | Hydra composition. `compose(stage, overrides)` reads `configs/command/<stage>.yaml` and the groups it names from the `machine` and `experiment` layers, then resolves every interpolation. |
| `tmg/core/records.py` | The attempt recorder. For each attempt it writes one `attempt_%06d.npz` with the target poses, the measured poses, the commands before and after noise, the segment ids and the follower state of every step, and 1 line in `attempts.jsonl`. |
| `tmg/core/chunks.py` | Chunk planning. A pool of N attempts is split into chunks of at most `chunk` attempts, each with its own seed, so that a pool can be resumed chunk by chunk. |
| `tmg/core/poses.py`, `actions.py`, `workers.py`, `provenance.py` | Pose arithmetic, action column grouping by joint name prefix, worker count selection, and the `provenance.json` writer (code revision, package versions, image, host). |
| `tmg/generation/base.py` | `SyntheticDataGenerator`, the abstract class every generator adapter implements. It defines `fetch_sources`, `prepare_chunk`, `chunk_command` and `chunk_result`. |
| `tmg/generation/runner.py` | `run_pool`. It plans the chunks, skips the finished ones, runs the others in parallel processes, validates each result, and writes `DONE.json`. |
| `tmg/generation/verify.py` | `verify_baseline`. It runs one chunk with the official generator and one chunk with the adapter and `method=none`, then compares the outputs byte for byte. |
| `tmg/generation/execution.py` | `TrajectoryExecution`. This is the step-by-step bookkeeping that is the same for every generator. It applies the path follower, counts the random draws, keeps the labels before noise and feeds the recorder. |
| `tmg/generation/method/` | The TrackMix method. `guided_mix.py` is the source mixer. `tracking.py` is the error-adaptive follower. `grippers.py` finds the gripper events. `interfaces.py` defines the `SegmentMixer`, `PathFollower` and `Method` protocols and `build_method(cfg)`. |
| `tmg/generation/simulators/robosuite.py` | Scene handling for robosuite. It reads a scene chain from `configs/scene/`, validates the placement bounds, and registers an environment class with the configured bounds and reach constraint. |
| `tmg/generation/generators/mimicgen/` | The MimicGen adapter. `graft.py` edits a function's syntax tree. `runtime.py` holds the callbacks the grafted code calls. `adapter.py` builds the chunk config. `entrypoint.py` is the chunk process. `sources.py` downloads and prepares the source demonstrations. |
| `configs/` | All YAML config groups. See section 2. |
| `docker/robosuite.Dockerfile`, `requirements/` | The image for the MimicGen generator. All packages are pinned in `requirements/robosuite.lock`. |
| `tests/unit/` | Unit tests that run without a simulator. |
| `Makefile` | The commands of section 3. Each target runs inside the docker image. |

The two method hooks are the only places where the method changes the generator. The mixer changes the source
segment of a subtask before the generator moves it to the current scene. The follower chooses the command of
each control step while the trajectory of a subtask executes. When both hooks are off, the inserted calls draw no
random numbers and the run is identical to the official run.

## 2. Configuration

`configs/README.md` describes every folder, group and command file in detail. In short, the config folder has three layers:

| Folder | Layer | Content |
|---|---|---|
| `configs/command/` | What command runs | One file per command (`generate`, `sources`, `verify_baseline`, `check_scene`, `rank_sources`, `dataset`, `train`, `select_checkpoint`, `resets`, `serve`, `eval`). It names a default for each group and holds the values of that command only. |
| `configs/machine/` | Where it runs | `cluster/` (container runtime, worker count, data root per machine) and `paths/` (data layout). |
| `configs/experiment/` | What experiment | Generator-independent groups `method/`, `generation/`, `policy/`, `training/`, `evaluation/`, and one folder per generator. |

Everything that depends on the generator lives under it:

```
configs/experiment/generator/mimicgen.yaml        the generator package: pinned commit, graft hashes, source repo, method_overrides
configs/experiment/generator/mimicgen/task/        one file per task: environment class, subtasks, cameras, source file
configs/experiment/generator/mimicgen/scene/       placement ranges per task for each scene condition (official_d1, ours)
configs/experiment/generator/mimicgen/ranking/     measured source rankings per task, written by `tmg rank_sources`
```

You select a file of a group on the command line with `group=name`. The generator's folder joins the search path,
so its groups keep short names:

```bash
tmg generate generator=mimicgen task=square scene=ours method=trackmix generation=pool cluster=isaac
```

| Group | Files | Meaning |
|---|---|---|
| `paths` | `local` | The data root. `TMG_DATA_ROOT` overrides it. Inside the images it is `/data`. |
| `cluster` | `local`, `isaac`, `a100`, `ada` | Container runtime, worker count, data root of one machine. `ada` is the single-GPU hand-off machine of the Isaac Lab baseline (`docs/ISAACLAB_BASELINE_RUNBOOK.md`). |
| `generator` | `mimicgen` | The generator adapter, its pinned commit, and the source hashes of the grafted functions. |
| `task` | MimicGen: `square`, `coffee`, `threading`, `stack_three`, `hammer_cleanup`, `mug_cleanup`, `three_piece_assembly`; SoftMimicGen: `rope`, `towel`, `jenga`; DexMimicGen: `pick_place`; CP-Gen: `coffee`, `threading`, `three_piece_assembly`, `hammer_cleanup`, `mug_cleanup` | The task of the chosen generator. |
| `scene` | `official_d1`, `ours` | Object placement. `official_d1` runs the generator's own distribution unchanged and lists its ranges. `ours` keeps each D1 centre, doubles the x and y widths, clips to 0.35 m of the table centre, pulls back where the 2026-10-26 scene check found robot contacts, and redraws until every object is within 0.85 m of the robot base. |
| `method` | `none`, `trackmix`, `mixup_only`, `tracking_only` | Which hooks are on. `none` runs the official generator unchanged. A generator may override a few values through `method_overrides` in its own file. |
| `generation` | `pilot20`, `pilot20_images`, `pool`, `pool_lowdim`, `pool_lowdim_onesrc`, `pool_isaac`, `rank3000`, `check200`, `official` | The attempt count, the chunk size, the seed offset and the generator protocol (noise, interpolation steps, whether cameras are rendered). `pool` stores images, `pool_lowdim` is the same pool without them. `pool_isaac` is the Isaac Lab pool (1,000 successes in 4 chunks; the protocol block does not apply to those generators, which keep their official settings). |
| `observation` | `image`, `low_dim` | What the policy sees. `image`: the task's cameras (84x84) and the robot state, the paper setting. `low_dim`: the robot state and the task's object state (`task.object_state`, no images), for method exploration. It selects the dataset variant, the policy's `by_observation` values and the training batch size. |
| `policy` | `bc_rnn`, `diffusion_policy`, `pi0` | Model values, with one override block per observation (`by_observation`). `pi0` has no low-dim form. |
| `training` | `default` | Batch size and update budget per policy and observation, epoch length, checkpoint interval, gradient clip, selection rule. |
| `evaluation` | `default` | Fixed scene counts (validation 25, test 300 per training seed), the training seeds 202601, 202602, 202603, the scene seed 202600 (robosuite: stored states; Isaac Lab: per-scene reset seeds), the policy server address. |

Every command writes exactly one output folder. For `generate`, `dataset` and `train` they are:

```
<paths.pools>/<generator>/<task>/<scene>/<method>/<generation>/
<paths.datasets>/<generator>/<task>/<scene>/<method>/<generation>/<observation>/n<size>_seed<seed>.hdf5
<paths.runs>/<generator>/<task>/<scene>/<method>/<generation>/<observation>/<policy>/seed<seed>/
```

A low-dim dataset can be written from an image pool or from a `pool_lowdim` pool; an image dataset needs a pool
that stored images. `make dataset`, `make train`, `make validate` and `make evaluate` take `OBSERVATION=image|low_dim`
(default `image`); evaluation must use the observation the checkpoint was trained with, and the policy server
reports its inputs so a mismatch stops with an error.

The config is strict. A key that does not exist in the group file causes an error. Use `tmg show generate ...`
to see the composed config before you start a long run.

## 3. How to run an experiment

### 3.1 Requirements

- Docker 24 or later on a Linux machine with a GPU. The image contains Python 3.10, MuJoCo 2.3.2, robosuite
  1.4.1, robomimic, and MimicGen at commit `72bd767`.
- About 20 GB of free disk for the image and the source demonstrations.
- For the unit tests on your own machine, Python 3.10 or later and `pip install -e .[dev]`.

### 3.2 Build the image

Run this command once per machine. A code or config change does not need a rebuild, because the repository is
mounted at `/opt/tmg` inside the container.

```bash
make build IMAGE=robosuite
```

### 3.3 Get the source demonstrations

This command downloads the released MimicGen source file of the task and prepares it with MimicGen's
`prepare_src_dataset`. The result is `<paths.sources>/mimicgen/<task>/source.hdf5` and a manifest next to it.

```bash
make sources TASK=square
```

### 3.4 Check that the adapter is identical to the official generator

Do this check after each change of the generator adapter or of the pinned packages. The command runs the same
chunk twice and writes `VERIFY.json`. The exit code is 0 only if the outputs are identical.

```bash
make verify-baseline TASK=square
```

### 3.5 Run a pilot pool of 20 attempts

A pilot pool uses the same code path as a full pool. Run it before each full pool.

```bash
make generate TASK=square SCENE=official_d1 METHOD=none GENERATION=pilot20
```

To run the path follower on the same 20 scenes, change the method:

```bash
make generate TASK=square SCENE=official_d1 METHOD=tracking_only GENERATION=pilot20
```

### 3.6 Run a full pool

A full pool makes chunks of 250 attempts. Set the attempt count on the command line through `OVERRIDES`.

```bash
make generate TASK=square SCENE=ours METHOD=none GENERATION=pool OVERRIDES="generation.attempts=3000"
```

If the run stops, run the same command again. Finished chunks are kept. If a finished chunk was made under a
different config, the run stops with an error. Add `generation.on_config_change=rebuild` to `OVERRIDES` to
discard those chunks.

### 3.7 Read the results

The pool folder contains:

| File | Content |
|---|---|
| `resolved_config.yaml` | The composed config of the run. |
| `provenance.json` | The code revision, the package versions, the image and the host. |
| `plan.json` | The chunk plan: index, seed and attempt count of each chunk. |
| `logs/seed<N>.log` | The output of the chunk process. |
| `chunks/seed<N>/chunk_spec.json`, `mg_config.json` | The chunk spec of the adapter and the MimicGen config of the chunk. |
| `chunks/seed<N>/<experiment>/demo.hdf5`, `demo_failed.hdf5`, `important_stats.json` | The MimicGen outputs of the chunk. |
| `chunks/seed<N>/records/attempts.jsonl` | 1 line per attempt with the success, the source indices, the subtask metadata and the mix record. |
| `chunks/seed<N>/records/attempt_%06d.npz` | The per-step arrays of the attempt. |
| `chunks/seed<N>/CHUNK_DONE.json` | The chunk is complete. |
| `DONE.json` | The pool is complete. It lists the attempts, the successes and the result of each chunk. |

If a chunk fails, the run stops with an error that names the log file of each failed chunk, and `DONE.json` is
not written. If the failure was in the recorder, the chunk's `records/` folder also keeps `RECORD_ERROR.json`
with the traceback.

### 3.8 Run a method variant

The method is the `METHOD` axis. To compare a variant with the baseline and with `trackmix`, give it a name of
its own and run the same five commands with that name. The name is part of every output path, so nothing is
overwritten.

1. Make the variant. For other values of the same algorithm, add a file to `configs/experiment/method/` that
   inherits `trackmix` and changes the values:

   ```yaml
   # configs/experiment/method/trackmix_w07.yaml
   defaults:
     - trackmix
     - _self_
   name: trackmix_w07
   mixing: {weight_a: 0.7}
   ```

   For another algorithm, write a class in `tmg/generation/method/`, register its name in `interfaces.py` and
   name it in the method file with `mixing.type` or `tracking.type`. `tmg/generation/method/README.md` gives the
   interfaces, the record keys and a skeleton.

2. Check the composed config, then run the pool. The baseline pool (`METHOD=none`) and the ranking of the task
   must exist, because the mixer reads the ranking file:

   ```bash
   tmg show generate task=square scene=ours method=trackmix_w07 generation=pool_lowdim
   make generate TASK=square SCENE=ours METHOD=trackmix_w07 GENERATION=pool_lowdim CLUSTER=a100 OVERRIDES="generation.attempts=5000"
   ```

3. Read the attempt records before you train anything. `chunks/seed<N>/records/attempts.jsonl` has the success
   of every attempt and the mix record of every subtask (`mixed`, `donor`, `identity_reason`), and the `.npz`
   files have the per-step follower records (`track_k`, `track_region`, ...). The generation success rate and
   these records already show whether the variant did what it should.

4. Train and evaluate with the same `METHOD`:

   ```bash
   make dataset  TASK=square SCENE=ours METHOD=trackmix_w07 GENERATION=pool_lowdim OBSERVATION=low_dim CLUSTER=a100 OVERRIDES="size=300"
   make train    TASK=square SCENE=ours METHOD=trackmix_w07 GENERATION=pool_lowdim OBSERVATION=low_dim POLICY=diffusion_policy SEED=1 CLUSTER=a100 OVERRIDES="dataset.size=300"
   make validate RUN=<data root>/runs/mimicgen/square/ours/trackmix_w07/pool_lowdim/low_dim/diffusion_policy/seed1 TASK=square SCENE=ours OBSERVATION=low_dim CLUSTER=a100
   make evaluate RUN=<the same run> TASK=square SCENE=ours OBSERVATION=low_dim CLUSTER=a100
   ```

   `validate` plays the 25 validation scenes of every checkpoint and appends to the run's `validation.jsonl`;
   `tmg select_checkpoint` picks the checkpoint; `evaluate` plays the 300 test scenes of that checkpoint.

5. Compare with the baseline (`METHOD=none`) and with `trackmix` run through the same commands. Keep the
   generation config, the observation, the policy and the seeds equal; only `METHOD` differs.

### 3.9 Run the tests

```bash
make test-local
```

This runs the unit tests without a simulator. `make test` runs them inside the image.

## 4. Rules for changes

- Do not copy or modify the official generator code. Add a graft anchor in `runtime.py` and record it in
  `docs/GENERATORS.md`.
- When the pinned commit of a generator changes, update `graft_hashes` in `configs/experiment/generator/<name>.yaml` and run
  `make verify-baseline`.
- Keep the method hooks generator-independent. Put step-level logic in `tmg/generation/execution.py`, not in
  the adapter.
- Add a unit test for each rule that must not drift, for example the random draw count or the label kind.
