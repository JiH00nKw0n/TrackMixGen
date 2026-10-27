# Configuration layout

This folder has three layers. `machine/` says on which computer a command runs, `experiment/` says what the
experiment is, and `command/` says which command runs. One command reads one file of `command/` as its base.
The `defaults` list of that file picks one file from each group it needs in `machine/` and `experiment/`. A
command-line choice such as `task=square` replaces the default of that group.

Rule for where a value belongs:

- A value that changes when the computer changes goes to `machine/` (container runtime, worker count, data root).
- A value that defines or compares an experiment condition goes to the matching group of `experiment/` (placement
  ranges, mixing weight, policy size, update budget).
- A value that only describes how one command runs goes to that command's file in `command/` (output folder rule,
  number of resets to inspect, which checkpoint to serve).

Every group file is strict: a key that the group file does not define causes an error at composition. Use
`tmg show <command> ...` to print the composed config before a long run.

## 1. `machine/`: values that differ per computer

| Group | Files | Content |
|---|---|---|
| `paths/` | `local.yaml` | The data layout. The root is `TMG_DATA_ROOT`, or `~/tmg_data` when the variable is not set; inside the containers it is `/data`. Under the root: `sources` (human source demonstrations), `pools` (generation pools), `datasets` (training files), `resets` (fixed evaluation scenes), `runs` (training runs), `reports` (scene checks, rankings). One file is enough because every machine uses the same relative layout. |
| `cluster/` | `local.yaml`, `isaac.yaml`, `a100.yaml` | The execution environment of one machine: container runtime (`docker` or `singularity`), image name, Singularity image path, data root on that machine, the upper limit of parallel workers (`max_workers`: Isaac 16, A100 48), threads per worker, and the minimum free disk space. `make CLUSTER=<name>` reads the runtime and the data root from this file. |

## 2. `experiment/`: the axes of an experiment

Each group is one folder; each file in the folder is one choice. The three groups that belong to a generator
(task, scene, ranking) live in that generator's own folder, `experiment/generator/<generator>/`, which joins the
search path when that generator is chosen.

| Group | Files | Content |
|---|---|---|
| `generator/` | `mimicgen.yaml`, `softmimicgen.yaml`, `dexmimicgen.yaml`, `cpgen.yaml` | Fixed facts of one generator: its simulator, container image, pinned commit of the official code, source hashes of the grafted official functions, the repository of the source demonstrations, and the few method values that must differ for this generator (`method_overrides`, for example DexMimicGen's 40-degree anchor bound and CP-Gen's mixing switched off). |
| `generator/<g>/task/` | one file per task | One task: environment class prefix, subtask names, reference objects, source file location and count, step limit (`horizon`), camera names and image size, robot state keys, object state keys (`object_state`, for low-dim observations), and the language instruction for π0. |
| `generator/<g>/scene/` | `official_*.yaml`, `ours.yaml` | Object placement. `official_*` keeps the generator's own distribution and lists its ranges for reading. `ours` is our range (D1 centres, doubled widths, clipped to 0.35 m of the table centre, every object within 0.85 m of the robot base). The file also names the registered environment class suffix (`D1`, `OURS`). |
| `generator/<g>/ranking/` | one file per task | The measured order of the source demonstrations per subtask (success rate, attempts, successes), written by `tmg rank_sources` from a baseline pool. The file only orders the sources; the good/bad split is a method value. |
| `method/` | `none.yaml`, `trackmix.yaml`, `mixup_only.yaml`, `tracking_only.yaml` | The switches and values of our method: the mixing block (primary weight 0.8, top fraction 0.5, 20-degree anchor bound, which ranking file), the tracking block (minimum advance, gains per region, approach window, extra-step cap, gripper event detector) and the label kind (`pre_noise` or `executed`). `mixup_only` and `tracking_only` inherit `trackmix` and switch one part off. |
| `generation/` | `pilot20`, `pilot20_images`, `pool`, `pool_lowdim`, `rank3000`, `check200` | Size and protocol of a pool: attempt count, chunk size, seed offset, noise 0.05, interpolation steps 5, and whether cameras are rendered. `pilot20` is a 20-attempt smoke run, `pool` the full pool with images, `pool_lowdim` the same pool without images, `rank3000` the 3,000-attempt ranking pool, `check200` a 200-attempt pool for code checks. The source-selection protocol (one source demo per subtask, chosen at random) is our standard and differs from MimicGen's templates on purpose; `docs/GENERATORS.md` section 1.4 records the difference and its measured effect. |
| `observation/` | `image.yaml`, `low_dim.yaml` | What the policy sees. `image`: the task's cameras and the robot state. `low_dim`: the robot state and the object state, no images. The choice selects the dataset variant, the policy's `by_observation` block and the training batch size. |
| `policy/` | `bc_rnn.yaml`, `diffusion_policy.yaml`, `pi0.yaml` | Model values: common `params`, one override block per observation (`by_observation`), and whether EMA weights are kept. Batch size and update budget are not here. |
| `training/` | `default.yaml` | The training budget and bookkeeping: batch size and update count per policy and observation, epoch length, checkpoint interval, gradient clip, which action label the policy learns, validation scene count, checkpoint selection rule. |
| `evaluation/` | `default.yaml` | The fixed evaluation scenes (25 validation, 300 test per training seed; 3 seeds per condition) and their seed, and the policy server address. One file, so every condition uses the same scenes. |

## 3. `command/`: one file per `tmg` command

Each file lists only the groups its command uses and holds the values that belong to that command alone.

| File | What the command does | Groups it uses | Values of its own |
|---|---|---|---|
| `sources.yaml` | downloads and prepares the source demonstrations | generator, task | none |
| `generate.yaml` | generates one pool | generator, task, scene, method, generation | output `pools/<generator>/<task>/<scene>/<method>/<generation>` |
| `verify_baseline.yaml` | compares one chunk run by the official generator and by the adapter with the method off, byte for byte | generator, task, scene, generation | output under `pools/_verify_baseline/` |
| `check_scene.yaml` | resets a scene many times and reports contacts, overhang, falls and reach | generator, task, scene | 200 resets, 20 settle steps, fall and drift thresholds |
| `rank_sources.yaml` | orders the sources from a baseline pool and writes the ranking file | generator, task, scene, generation | the pool to read (method fixed to `none`), minimum 100 attempts |
| `dataset.yaml` | writes a training file from a finished pool | generator, task, scene, method, generation, observation | 1,000 demonstrations, selection seed, output `datasets/.../<observation>/n1000_seed0.hdf5` |
| `train.yaml` | trains one policy | the six above plus policy, training | training seed, update override, output `runs/.../<observation>/<policy>/seed<k>` |
| `select_checkpoint.yaml` | chooses one checkpoint of a run | same as `train` | same output rule |
| `resets.yaml` | draws and stores the fixed evaluation scenes | generator, task, scene, evaluation | robot name |
| `serve.yaml` | serves one checkpoint over a websocket | evaluation | checkpoint path, device, server host and port |
| `eval.yaml` | plays the fixed scenes against the served policy and measures success | generator, task, scene, observation, evaluation | checkpoint, split (val or test), scene limit, output folder |

Output folders of the three main commands:

```
<paths.pools>/<generator>/<task>/<scene>/<method>/<generation>/
<paths.datasets>/<generator>/<task>/<scene>/<method>/<generation>/<observation>/n<size>_seed<seed>.hdf5
<paths.runs>/<generator>/<task>/<scene>/<method>/<generation>/<observation>/<policy>/seed<seed>/
```
