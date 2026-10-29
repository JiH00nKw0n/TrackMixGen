# The `tmg` package

`tmg` is one Python package that runs inside several container images. The images have different Python and
torch versions (the simulators pin old ones, the policies need new ones), so the package is split by what each
image needs. `tmg/core/` imports only light libraries and is used everywhere; `tmg/generation/` and
`tmg/evaluation/` run in the simulator images; `tmg/data/` runs in both; `tmg/policy/` runs in the train image.

```
tmg/
  cli.py            `tmg <command> key=value ...`: composes the config and calls the command's function from the table
  config/           Hydra composition over configs/ (strict keys, generator folder on the search path), the command table
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
| `io_json.py` | JSON and JSON-lines helpers (`write_json`, `read_json`, `read_jsonl`, `append_jsonl`); `chunks.py` re-exports the first two. |
| `layout.py` | Every result path and file name in one place: the pool, dataset, run and resets folders of a condition, the run-folder file names (`checkpoint.json`, `validation.jsonl`, `SELECTED.json`, ...), sidecar names, and `indexed_keys` for `demo_<i>`-style hdf5 keys. Standard library only. |
| `records.py` | `AttemptRecorder`: the generator-independent record of one attempt (target and measured end-effector poses per step, pre-noise and executed actions, noise, segment table, success flags, source indices), written as `attempt_%06d.npz` and one line of `attempts.jsonl`; `validate_records` and `read_attempts`. |
| `poses.py` | Pose arithmetic on 4x4 matrices (numpy only). |
| `features.py` | The policy feature keys (`observation.images.<camera>`, `observation.state`, `observation.environment_state`, `action`). |
| `seeding.py` | `seed_all`: seeds the environment (when given), torch, numpy and `random` from one value. It imports torch, so only the train and simulator images call it. |
| `actions.py` | Action vector layouts of the adapters (arm blocks, gripper columns, hand joint prefixes). |
| `workers.py` | Worker count for chunk processes from the cluster config and the CPU count. |
| `provenance.py` | Writes what produced a result folder: code revision, installed packages, machine, image, time. |
| `wire.py` | msgpack encoding of messages with numpy arrays between the policy server and the evaluation client. |

## `tmg/generation/`: making demonstrations

| Module | Content |
|---|---|
| `base.py` | `SyntheticDataGenerator`, the contract every adapter implements: `fetch_sources`, `prepare_chunk`, `chunk_result`, `demo_layout`; a default `chunk_command` that runs the adapter's `entrypoint_module`; `_spec`, which builds the `ChunkSpec` the same way for every adapter; `apply_method_overrides` (the generator's `method_overrides` merged into the method config). |
| `runner.py` | `run_pool`: plans the chunks, skips finished ones, runs the others as parallel processes, validates each result, writes `DONE.json`. Generator-independent. |
| `verify.py` | `verify_baseline`: one MimicGen chunk run two times with the same `mg_config.json`, without the grafts and with the grafts and `method=none`, compared byte for byte. It shows that the grafts change nothing when the method is off, not that our settings equal the authors' settings. MimicGen only (CP-Gen and the Isaac Lab generators are not deterministic from run to run on the GPU). |
| `graft.py` | `Graft`: reads an official function's source, checks its hash, inserts our calls at named anchors in the syntax tree, recompiles. The official source is never copied or edited on disk. |
| `execution.py` | `TrajectoryExecution`: the step bookkeeping of the MimicGen adapter (follower driving, noise accounting, pre-noise labels, feeding the recorder), and `SegmentTable`, the waypoint-to-record-segment map that the CP-Gen runtime also uses. |
| `seeds.py` | The random streams we derive from a chunk seed (private noise stream: seed + 7919; per-attempt source pick of CP-Gen), named once so every adapter derives them the same way. |
| `scene_check.py` | `tmg check_scene`: resets a (task, scene) many times and reports contacts, overhang, falls, drift and reach. |
| `rank_sources.py` | `tmg rank_sources`: per-source success rates from a baseline pool, written as the task's ranking file. |
| `method/` | The TrackMix method behind two hook interfaces. `interfaces.py` (`SegmentMixer`, `PathFollower`, `Method`, `build_method`), `guided_mix.py` (source mixing guided by the ranking), `event_mix.py` (event-aligned blending of two segments), `tracking.py` (`ErrorAdaptiveFollower`, `EndEffectorPath`), `grippers.py` (gripper event detectors: sign rule, hand closure), `ranking.py` (reading rankings and splitting them). No simulator imports. |
| `simulators/` | Simulator-level helpers shared by adapters on the same simulator. `robosuite.py` registers environment classes with the scene ranges and the reach constraint on MimicGen's classes; `isaaclab.py` applies scene ranges to Isaac Lab environment configs, reads hand layouts and measured link poses; `isaaclab_tasks/` registers a GR1 pick-and-place environment with two cameras. |
| `generators/` | One adapter per generator. `make_generator(cfg)` picks it from the `ADAPTERS` table by `generator.name` and imports only that adapter. Shared code: `robosuite_base.py` (base of the two robosuite adapters), `_sources.py` (download, md5, manifest), `_shared.py` (record summary, `RECORD_ERROR.json`), `_entry.py` (spec loading and graft installation in the chunk process). |

Each adapter folder has the same four files:

| File | Role |
|---|---|
| `adapter.py` | The `SyntheticDataGenerator`: builds the official generator's configuration for one chunk, names the chunk command, validates the chunk's outputs, tells `tmg dataset` where the demonstrations are (`demo_layout`). |
| `entrypoint.py` | The chunk process: registers the scene's environment, installs the grafts, runs the official generation script or function. |
| `runtime.py` | The `_tmg` object the grafted official code calls: begins and ends attempts, mixes segments, drives the follower, records steps. |
| `sources.py` | Downloads and prepares the official source demonstrations. |

`mimicgen/` also has `config_builder.py` (the official json config from the task template and our protocol). `cpgen/` also has
`windows.py` (per-demonstration constraint windows of the source demonstrations) and `checkout.py` (puts CP-Gen's checkout on the path).
`isaaclab/` is the Isaac Lab base (SoftMimicGen on its Isaac Lab fork; the chunk entrypoint, the mixing runtime and `tracking_loop.py`, the one tracking loop whose package-specific steps are hooks), `autodata/` the DexMimicGen generator (NVIDIA AutoData's `autodata_core` on Isaac Lab 3.0 with Arena: Pick Place, Microwave, Bottle Door). `cpgen/` is CP-Gen.

### Adding a method variant

See `tmg/generation/method/README.md`: the hook interfaces, the record keys a follower must return, the current
values, and the three steps of a new variant (class, registry name, method file).

## `tmg/data/`: from pools to training files

| Module | Content |
|---|---|
| `dataset.py` | `tmg dataset`: selects successful attempts of a finished pool and writes one robomimic-compatible hdf5 with the observation chosen (`image`: cameras and robot state; `low_dim`: robot and object state), the pool's action label, the executed and pre-noise actions, states, segment indices and source indices. `DemoLayout` says where each generator keeps its outputs. |
| `loader.py` | `TrajectoryStore` (the whole file in RAM, images as uint8) and `FrameDataset` (one item per step with LeRobot delta indices for temporal stacking), `collate`, `to_device`, `compute_stats`. |
| `images.py` | Camera frames: resizing, `to_uint8_frame`, and `camera_obs_key` (where a camera's frames sit in a pool or a dataset). |

## `tmg/policy/`: training and serving (train image)

| Module | Content |
|---|---|
| `lerobot_policies.py` | `POLICY_SPECS`: one `PolicySpec` per policy (stored config type, LeRobot config and policy classes, batch IO). Builds a LeRobot policy from our `policy` config and the dataset's features; `by_observation` blocks; `load_policy` from a checkpoint. |
| `bc_rnn.py`, `nets.py` | BC-RNN in LeRobot's `PreTrainedPolicy` format (robomimic `BC_RNN_GMM` port: image encoders, LSTM, Gaussian mixture head; low-dim form without encoders). |
| `normalize.py`, `io.py` | Per-key normalization outside the policies (`Normalizer`) and the `PolicyIO` that prepares batches and post-processes actions: our normalizer for BC-RNN and Diffusion Policy, LeRobot's processor pipelines for π0. |
| `ema.py` | Exponential moving average of the weights (Diffusion Policy evaluates the EMA copy). |
| `trainer.py` | `tmg train`: the one loop for every policy (budget per policy and observation, gradient clip, checkpoints with resume, validation command hook) and `tmg select_checkpoint` (`select_checkpoint_cmd`). |
| `server.py` | `tmg serve`: websocket server around one checkpoint (`info`, `reset`, `act`). |

## `tmg/evaluation/`: measuring success (simulator image)

| Module | Content |
|---|---|
| `resets.py` | `tmg resets`: draws and stores the fixed validation and test scenes of a (task, scene) with their model xml. |
| `rollout.py` | `tmg eval` for robosuite: restores each stored scene, asks the policy server for an action every step, records success per episode; appends the validation row the checkpoint selection reads. |
| `isaaclab_rollout.py` | `tmg eval` for the Isaac Lab tasks, with the same session code and output files as `rollout.py`. |
| `session.py` | What both rollouts share: checking that the policy server matches the evaluation (cameras, state size), the per-scene loop, episode seeds, the Wilson interval and the summary. |
| `envs.py` | Environment construction details shared by resets and rollouts (robosuite evaluation keyword arguments, Isaac Lab task id). |
| `client.py` | Synchronous websocket client of the policy server. |

Rules kept across the package: official generator code is grafted at run time and never copied; everything an
experiment writes goes to exactly one output folder with a `resolved_config.yaml` and a provenance file; the
method hooks (`tmg/generation/method/`) never import a simulator, so they are unit-tested without one.
