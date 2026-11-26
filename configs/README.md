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
| `paths/` | `local.yaml` | The data layout. The root is `TMG_DATA_ROOT`, or `~/tmg_data` when the variable is not set; inside the containers it is `/data`. Under the root: `sources` (human source demonstrations), `pools` (generation pools), `datasets` (training files), `resets` (fixed evaluation scenes), `runs` (training runs), `reports` (scene checks, coefficient fits). One file is enough because every machine uses the same relative layout. |
| `cluster/` | `local.yaml`, `isaac.yaml`, `a100.yaml`, `ada.yaml` | The execution environment of one machine (`ada`: the single-GPU hand-off machine of the Isaac Lab baseline, with `device`, the GPU Isaac Sim uses): container runtime (`docker` or `singularity`), image name, Singularity image path, data root on that machine, the upper limit of parallel workers (`max_workers`: Isaac 16, A100 48; `isaaclab_max_workers`, default 3: the cap on concurrent Isaac Sim chunk processes), threads per worker, and the minimum free disk space. `make CLUSTER=<name>` reads the runtime and the data root from this file. |

## 2. `experiment/`: the axes of an experiment

Each group is one folder; each file in the folder is one choice. The groups that belong to a generator
(task, scene, coefficients) live in that generator's own folder, `experiment/generator/<generator>/`, which joins the
search path when that generator is chosen.

| Group | Files | Content |
|---|---|---|
| `generator/` | `mimicgen.yaml`, `softmimicgen.yaml`, `dexmimicgen.yaml` (AutoData stack: Pick Place, Microwave, Bottle Door), `cpgen.yaml` | Fixed facts of one generator: its simulator, container image, pinned commit of the official code, source hashes of the grafted official functions, the repository of the source demonstrations, and the few method values that must differ for this generator (`method_overrides`, for example DexMimicGen's hand-closure events and CP-Gen's pre-noise labels). |
| `generator/<g>/task/` | one file per task | One task: environment class prefix, subtask names, reference objects, source file location and count, step limit (`horizon`), camera names and image size, robot state keys, object state keys (`object_state`, for low-dim observations), and the language instruction for π0. |
| `generator/<g>/scene/` | `official_*.yaml`, `ours.yaml` | Object placement (CP-Gen: `official_wide` and `ours` derive from CP-Gen's Wide classes, which also draw object sizes per reset; `env_class` / `base_env_class` name such classes). `official_*` keeps the generator's own distribution and lists its ranges for reading. `ours` is our range (D1 centres, doubled widths, clipped to 0.35 m of the table centre, every object within 0.85 m of the robot base). The file also names the registered environment class suffix (`D1`, `OURS`). |
| `selection/` | `official.yaml`, `ours.yaml` | The source-demonstration selection rule, independent of the pool's size and recording. `official`: the generator's own rule. For MimicGen that is what its `scripts/generate_core_configs.py` writes on top of the templates: one source per attempt chosen at random, nearest-neighbour (`nn_k` 3) for Square, and for Stack Three nearest-neighbour picked again at every subtask. For CP-Gen it is source 0 only. `ours`: MimicGen picks a source at random for every subtask. CP-Gen uses every source, each with its own constraint windows written by `tmg sources`. Named in the pool, dataset and run folders after the scene. |
| `method/` | `none.yaml`, `skillmix.yaml` | `none`: the generator as released. `skillmix`: SkillMix source mixing (its rule, `softmax` or `randmix`; the rule file it reads; the object symmetries) and the label kind (`pre_noise` or `executed`). |
| `generation/` | `pool_robosuite_image`, `pool_robosuite_lowdim`, `pool_isaac`, `pool_unmix`, `imported_official`, and `test/` (`pilot20`, `pilot20_images`, `pilot50`, `pool_robosuite_lowdim_onesrc`) | Size and protocol of a pool (`pool_isaac`: the Isaac Lab generators' pool of 1,000 successes in 4 chunks. Their official source selection, noise and interpolation apply. From the protocol block they read only `guarantee` and, for DexMimicGen, `camera_names`): attempt count, chunk size, seed offset, noise 0.05, interpolation steps 5, and whether cameras are rendered. `pool_robosuite_image` is the full robosuite pool with images and holds the protocol block the others inherit, `pool_robosuite_lowdim` the same pool without images, `pool_unmix` the 2,000-attempt pools the SkillMix fit reads (control, and randmix as `pool_randmix`), `imported_official` the name an imported official dataset is stored under. `test/pilot20` (20 attempts, no images), `test/pilot20_images`, `test/pilot50` and the diagnostic `test/pool_robosuite_lowdim_onesrc` are selected as `generation=test/<name>`; the pool folder is named after `name`, without `test/`. The source selection is not part of this group: it comes from `selection/` (`official` or `ours`), see `docs/GENERATORS.md` section 1.4. |
| `observation/` | `image.yaml`, `low_dim.yaml` | What the policy sees. `image`: the task's cameras and the robot state. `low_dim`: the robot state and the object state, no images. The choice selects the dataset variant, the policy's `by_observation` block and the training batch size. |
| `policy/` | `bc_rnn.yaml`, `diffusion_policy.yaml`, `pi0.yaml`, `pi05.yaml` | Model values: common `params`, one override block per observation (`by_observation`), and whether EMA weights are kept. Batch size and update budget are not here. |
| `training/` | `default.yaml` | The training budget and bookkeeping: batch size and update count per policy and observation, epoch length, checkpoint interval, gradient clip, which action label the policy learns, validation scene count, checkpoint selection rule. |
| `evaluation/` | `default.yaml` | The fixed evaluation scenes (25 validation, 300 test per training seed; 3 seeds per condition) and their seed, and the policy server address. One file, so every condition uses the same scenes. |

## 3. `command/`: one file per `tmg` command

Each file lists only the groups its command uses and holds the values that belong to that command alone.

| File | What the command does | Groups it uses | Values of its own |
|---|---|---|---|
| `sources.yaml` | downloads and prepares the source demonstrations | generator, task | none |
| `generate.yaml` | generates one pool | generator, task, scene, selection, method, generation | output `pools/<generator>/<task>/<scene>/<selection>/<method>/<generation>` |
| `verify_baseline.yaml` | compares one chunk run by the official generator and by the adapter with the method off, byte for byte | generator, task, selection, scene, generation | output under `pools/_verify_baseline/` |
| `check_scene.yaml` | resets a scene many times and reports contacts, overhang, falls and reach | generator, task, scene | 200 resets, 20 settle steps, fall and drift thresholds |
| `dataset.yaml` | writes a training file from a finished pool | generator, task, scene, selection, method, generation, observation | 1,000 demonstrations, selection seed, output `datasets/.../<observation>/n1000_seed0.hdf5` |
| `train.yaml` | trains one policy | the seven above plus policy, training | training seed, update override, output `runs/.../<observation>/<policy>/seed<k>` |
| `select_checkpoint.yaml` | chooses one checkpoint of a run | same as `train` | same output rule |
| `resets.yaml` | draws and stores the fixed evaluation scenes | generator, task, scene, evaluation | robot name |
| `serve.yaml` | serves one checkpoint over a websocket | evaluation | checkpoint path, device, server host and port |
| `eval.yaml` | plays the fixed scenes against the served policy and measures success | generator, task, scene, observation, evaluation | checkpoint, split (val or test), scene limit, output folder |

Output folders of the three main commands:

```
<paths.pools>/<generator>/<task>/<scene>/<selection>/<method>/<generation>/
<paths.datasets>/<generator>/<task>/<scene>/<selection>/<method>/<generation>/<observation>/n<size>_seed<seed>.hdf5
<paths.runs>/<generator>/<task>/<scene>/<selection>/<method>/<generation>/<observation>/<policy>/seed<seed>/
```
