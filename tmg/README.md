# The `tmg` package

`tmg` is one Python package that runs inside several container images. The images have different Python and
torch versions (the simulators pin old ones, the policies need new ones), so the package is split by what each
image needs. `tmg/core/` imports only light libraries and is used everywhere; `tmg/generation/` and
`tmg/evaluation/` run in the simulator images; `tmg/data/` runs in both; `tmg/policy/` runs in the train image.

```
tmg/
  cli.py            `tmg <command> key=value ...`: composes the config and calls the module of the command
  config/           Hydra composition over configs/ (strict keys, generator folder on the search path)
  core/             shared, light: chunk plan, attempt records, poses, action layouts, provenance, wire format
  generation/       synthetic demonstration generation: runner, method hooks, grafting, one adapter per generator
  data/             the common training hdf5 written from a pool, and the in-memory loader the policies read
  policy/           the policies, the training loop, the policy server (train image)
  evaluation/       fixed evaluation scenes and the rollout that talks to the policy server (simulator image)
```

## `tmg/core/`: shared by every image

| Module | Content |
|---|---|
| `chunks.py` | The chunk plan of a pool (attempt count, chunk size, seeds) and the on-disk markers (`CHUNK_DONE.json`, `DONE.json`) that make a pool resumable. |
| `records.py` | `AttemptRecorder`: the generator-independent record of one attempt (target and measured end-effector poses per step, pre-noise and executed actions, noise, segment table, success flags, source indices), written as `attempt_%06d.npz` and one line of `attempts.jsonl`; `validate_records` and `read_attempts`. |
| `poses.py` | Pose arithmetic on 4x4 matrices (numpy only). |
| `actions.py` | Action vector layouts of the adapters (arm blocks, gripper columns, hand joint prefixes). |
| `workers.py` | Worker count for chunk processes from the cluster config and the CPU count. |
| `provenance.py` | Writes what produced a result folder: code revision, installed packages, machine, image, time. |
| `wire.py` | msgpack encoding of messages with numpy arrays between the policy server and the evaluation client. |

## `tmg/generation/`: making demonstrations

| Module | Content |
|---|---|
| `base.py` | `SyntheticDataGenerator`, the contract every adapter implements: `fetch_sources`, `prepare_chunk`, `chunk_command`, `chunk_result`, `demo_layout`; `ChunkSpec`; `apply_method_overrides` (the generator's `method_overrides` merged into the method config). |
| `runner.py` | `run_pool`: plans the chunks, skips finished ones, runs the others as parallel processes, validates each result, writes `DONE.json`. Generator-independent. |
| `verify.py` | `verify_baseline`: one chunk by the official generator and one through the adapter with `method=none`, compared byte for byte. |
| `graft.py` | `Graft`: reads an official function's source, checks its hash, inserts our calls at named anchors in the syntax tree, recompiles. The official source is never copied or edited on disk. |
| `execution.py` | `TrajectoryExecution`: the step bookkeeping shared by the robosuite adapters (follower driving, noise accounting, pre-noise labels, feeding the recorder). |
| `scene_check.py` | `tmg check_scene`: resets a (task, scene) many times and reports contacts, overhang, falls, drift and reach. |
| `rank_sources.py` | `tmg rank_sources`: per-source success rates from a baseline pool, written as the task's ranking file. |
| `method/` | The TrackMix method behind two hook interfaces. `interfaces.py` (`SegmentMixer`, `PathFollower`, `Method`, `build_method`), `guided_mix.py` (source mixing guided by the ranking), `event_mix.py` (event-aligned blending of two segments), `tracking.py` (`ErrorAdaptiveFollower`, `EndEffectorPath`), `grippers.py` (gripper event detectors: sign rule, hand closure), `ranking.py` (reading rankings and splitting them). No simulator imports. |
| `simulators/` | Simulator-level helpers shared by adapters on the same simulator. `robosuite.py` registers environment classes with the scene ranges and the reach constraint on MimicGen's classes; `isaaclab.py` applies scene ranges to Isaac Lab environment configs, reads hand layouts and measured link poses; `isaaclab_tasks/` registers a GR1 pick-and-place environment with two cameras. |
| `generators/` | One adapter per generator. `make_generator(cfg)` picks it by `generator.name`. |

Each adapter folder has the same four files:

| File | Role |
|---|---|
| `adapter.py` | The `SyntheticDataGenerator`: builds the official generator's configuration for one chunk, names the chunk command, validates the chunk's outputs, tells `tmg dataset` where the demonstrations are (`demo_layout`). |
| `entrypoint.py` | The chunk process: registers the scene's environment, installs the grafts, runs the official generation script or function. |
| `runtime.py` | The `_tmg` object the grafted official code calls: begins and ends attempts, mixes segments, drives the follower, records steps. |
| `sources.py` | Downloads and prepares the official source demonstrations. |

`mimicgen/` also has `config_builder.py` (the official json config from the task template and our protocol).
`isaaclab/` is the Isaac Lab base (SoftMimicGen on its Isaac Lab fork; the chunk entrypoint and the mixing/tracking runtime), `autodata/` the DexMimicGen generator (NVIDIA AutoData's `autodata_core` on Isaac Lab 3.0 with Arena: Pick Place, Microwave, Bottle Door). `cpgen/` is CP-Gen.

### Adding a method variant

See `tmg/generation/method/README.md`: the hook interfaces, the record keys a follower must return, the current
values, and the three steps of a new variant (class, registry name, method file).

## `tmg/data/`: from pools to training files

| Module | Content |
|---|---|
| `dataset.py` | `tmg dataset`: selects successful attempts of a finished pool and writes one robomimic-compatible hdf5 with the observation chosen (`image`: cameras and robot state; `low_dim`: robot and object state), the pool's action label, the executed and pre-noise actions, states, segment indices and source indices. `DemoLayout` says where each generator keeps its outputs. |
| `loader.py` | `TrajectoryStore` (the whole file in RAM, images as uint8) and `FrameDataset` (one item per step with LeRobot delta indices for temporal stacking), `collate`, `to_device`, `compute_stats`. |

## `tmg/policy/`: training and serving (train image)

| Module | Content |
|---|---|
| `lerobot_policies.py` | Builds a LeRobot policy from our `policy` config and the dataset's features (cameras, state, object state, action); `by_observation` blocks; `load_policy` from a checkpoint. |
| `bc_rnn.py`, `nets.py` | BC-RNN in LeRobot's `PreTrainedPolicy` format (robomimic `BC_RNN_GMM` port: image encoders, LSTM, Gaussian mixture head; low-dim form without encoders). |
| `normalize.py`, `io.py` | Per-key normalization outside the policies (`Normalizer`) and the `PolicyIO` that prepares batches and post-processes actions: our normalizer for BC-RNN and Diffusion Policy, LeRobot's processor pipelines for π0. |
| `ema.py` | Exponential moving average of the weights (Diffusion Policy evaluates the EMA copy). |
| `trainer.py` | `tmg train`: the one loop for every policy (budget per policy and observation, gradient clip, checkpoints with resume, validation command hook) and `select_checkpoint`. |
| `server.py` | `tmg serve`: websocket server around one checkpoint (`info`, `reset`, `act`). |

## `tmg/evaluation/`: measuring success (simulator image)

| Module | Content |
|---|---|
| `resets.py` | `tmg resets`: draws and stores the fixed validation and test scenes of a (task, scene) with their model xml. |
| `rollout.py` | `tmg eval`: restores each stored scene, asks the policy server for an action every step, records success per episode and a Wilson interval; appends the validation row the checkpoint selection reads. |
| `client.py` | Synchronous websocket client of the policy server. |

Rules kept across the package: official generator code is grafted at run time and never copied; everything an
experiment writes goes to exactly one output folder with a `resolved_config.yaml` and a provenance file; the
method hooks (`tmg/generation/method/`) never import a simulator, so they are unit-tested without one.
