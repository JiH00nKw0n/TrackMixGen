# `tmg/generation/method/`: SkillMix

This folder holds the one hook that every generator adapter calls (the segment mixer), its implementation, SkillMix,
and the pieces it shares. Nothing here imports a simulator, so every class is unit-tested with numpy arrays alone. Three
methods exist (`configs/experiment/method/`): `none`, the generator as released, and SkillMix with its two rules,
`skillmix` (softmax) and `skillmix_sparsemax` (sparsemax).

```
interfaces.py    MixContext, MixedSegment, SegmentMixer, Method, build_method and the mixer registry _MIXERS (MixerType)
skillmix.py      SkillMixer (mixing.type: skillmix), the rule files (SkillMixRule, load_rule), the fitted rules (FITTED,
                 fitted_weights),
                 RandmixDraws / randmix_weights (mixing.rule: randmix), symmetry alignment
success_model.py the per-source features (scene, in-hand) and the rows of the fit (task-success label)
event_mix.py     event-aligned blending of segments (mix_many; align_segments for CP-Gen)
grippers.py      gripper state and event rules: gripper_states / change_indices (scalar sign commands),
                 SignGripper and ClosureGripper detectors (scalar +/- commands, many-joint hands), build_detector
```

## SkillMix

Per subtask k of an attempt, every source j gets five features against the current scene, `phi_j` = (|dx|, |dy|, rotation
of the reference object; in-hand position and rotation distances from the second subtask on). A rule file
(`configs/experiment/generator/<generator>/coefficients/<rule>/<task>.yaml`, written by `tmg fit_coefficients`) holds, per
subtask, the fitted mixed-effects logistic model of the task success given the mixing weights `w`:

    P(task succeeds | reached k, w) = sigmoid( c0 + sum_j w_j s_j + d spread(w) ),   s_j = a_j + g . phi_j

`a_j` is source j's own effect, `g` the feature slopes, `d` the effect of spreading the weight. The two rules measure the
spread differently and are fitted on the same rows, one file each:

| rule (`mixing.rule`, method) | spread | weights when `d > 0` |
|---|---|---|
| `softmax` (`skillmix`) | entropy `H(w) = -sum w log w` | `softmax(s / d)`, weights below 1e-3 dropped |
| `sparsemax` (`skillmix_sparsemax`) | Gini `1 - sum w^2` | `sparsemax(s / (2 d))`: sources below a threshold get exactly 0 |

Over the sources compatible with the selected one (same gripper events). When `d <= 0` one source takes all the weight: the best-scored one (`mixing.single_source: best`, method `skillmix`) or one drawn with probability `softmax(s)` (`sample`, method `skillmix_sample`; with `best` every scene can get the same source). The retained segments are aligned
by the object and gripper symmetries and blended in their reference-object frames (`skillmix.py` module docstring). CP-Gen
blends its re-solved constraints itself (`tmg/generation/generators/cpgen/mixing.py`) and takes the weights and their record
from the same call, `SkillMixer.decide`, so both generators apply one rule.

**Rule files, from scratch.** `make coefficients GENERATOR=<g> TASK=<t>` (`scripts/ensure_coefficients.sh`) builds two
2,000-attempt pools of the same settings and chunk seeds (selection `ours`, generation `pool_unmix`; chunks of 50, CP-Gen
25 from its generator config, the sizes of the committed rules' pools): an unmix pool (method `none`, `none/pool_unmix`) and a randmix pool
(`skillmix_randmix/pool_randmix`, method `skillmix` with `mixing.rule=randmix` and `method.name=skillmix_randmix`: random
Dirichlet weights), then fits the model on both (`tmg fit_coefficients`) for the rule files that are missing; a rule file
that exists is never refitted (`make fit-coefficients` refits both, or the rules of `OVERRIDES="rules=[...]"`).
`make generate METHOD=skillmix` (or `skillmix_sparsemax`) does this by itself when its rule file is missing. The committed files (MimicGen: seven tasks, CP-Gen: five) were fitted on 2026-11-20;
each file's `basis` names its pools. Pools made on the servers before 2026-11-23 keep their folder names (`<method>/pool_calib`, `none/pool_ctrl`); to fit on them, name the folders: `make fit-coefficients GENERATOR=<g> TASK=<t> SCENE=ours SELECTION=ours GENERATION=pool_unmix OVERRIDES="randmix_pools=[/data/pools/<g>/<t>/ours/ours/mixup_only/pool_calib] unmix_pools=[/data/pools/<g>/<t>/ours/ours/none/pool_ctrl]"`.

The rows of the fit are the attempts that reached subtask k, labelled with the attempt's task success (not subtask k's own
success; `success_model.py` explains why). The fit standardizes the columns and picks an L2 penalty by chunk-grouped
cross-validation; each block of the rule file also keeps the cross-validated log-loss gain over the base rate.

## 1. The hook

A generator adapter (`tmg/generation/generators/<name>/runtime.py`) calls the mixer once per subtask: after the official
code has selected a source demonstration for the subtask and cut its segment (end-effector poses, target poses, gripper
commands) and before it transforms that segment to the current object pose, the adapter calls `mixer.mix(context)`. The
runtimes also call `mixer.begin_attempt()` when an attempt starts (the randmix rule draws new weights per attempt).

| `MixContext` field | Meaning |
|---|---|
| `subtask_ind` | index of the subtask |
| `primary` | index of the source the official code selected (A) |
| `primary_eef_start`, `primary_targets`, `primary_gripper` | A's segment: first end-effector pose (4x4), target poses (T, 4, 4), gripper commands (T, G) |
| `src_object_poses` | per source: the reference object's pose (4x4) at that source's segment start, so a mixer can express every segment in the object frame |
| `src_segments` | per source: (eef_start, targets, gripper) of its segment |
| `num_sources`, `has_reference_object` | how many sources exist; whether the subtask has a reference object (no object: no frame to mix in) |
| `rng` | optional (defaults to `default_rng(0)`); SkillMix does not use it (its randmix draws come from their own stream seeded with the chunk seed) |
| `extra` | `current_object_pose`, `reference_object`, and from the second subtask on `current_in_hand` and `source_in_hand` |

The mixer returns a `MixedSegment`: the segment the generator should transform and execute (`eef_start`, `targets`,
`gripper`) and a `record` dict. Returning the inputs unchanged is the identity (the official behaviour). The record goes
into the attempt's `subtasks[i]["mix"]` entry of `attempts.jsonl`; SkillMix writes `call`, `subtask`, `source_a`, `rule`,
`mixed`, `weights`, `incompatible`, `scene_features` (and `inhand_features`), with a fitted rule `scores` and `d`, with
the randmix rule `dirichlet_c`, on a blend `symmetry` and `alignment`, and when it did nothing `identity_reason`. The fit
reads `weights`, `scene_features` and `inhand_features` of the randmix pools.

## 2. Current values

`configs/experiment/method/skillmix.yaml` holds every value (`skillmix_sparsemax.yaml` takes them all and sets the rule
and the name):

| Block | Key | Value | Meaning |
|---|---|---|---|
| `mixing` | `type` | `skillmix` | implementation (`SkillMixer`) |
| | `rule` | `softmax` | `softmax` or `sparsemax` (`skillmix_sparsemax.yaml`): the SkillMix weights; `randmix`: random Dirichlet weights for the randmix pools only |
| | `coefficients` | `<generator>/<task>` | which rule file to read |
| | `symmetries` | `task.mixing_symmetries` | object symmetries the sources may be aligned with before blending |
| | `gripper_events` | `{type: sign}` (default) | DexMimicGen: `{type: closure, columns: [0, 1, 2, 3]}` (`method_overrides` of its generator file) |
| `labels` | | `executed` | the dataset stores what the simulator received (CP-Gen: `pre_noise`, its saved actions are noise-free) |

A generator overrides values under `method_overrides: {<method name>: {...}}` in its own file; every rule has its own
method name (`skillmix`, `skillmix_sparsemax`, `skillmix_randmix` for the randmix pools), so a generator that overrides
`skillmix` lists all three. Those names run their own rule only; a variant of a fitted rule (another rule file) takes a
name of its own, and the randmix rule runs under `skillmix_randmix` alone.

## 3. Adding a mixer variant

1. Write the class next to `skillmix.py`: a `mix(context) -> MixedSegment` method, a class method `from_config(cfg, *,
   chunk_seed)`, and `begin_attempt()` if it keeps per-attempt state. Seed every random draw from the chunk seed.
2. Register it under a name in `interfaces._mixer_types()`.
3. Add a method file that names it (`mixing: {type: <name>, ...}`) under its own `name`, so its pools, datasets and runs
   get their own folders.

Write its unit test next to `tests/unit/test_skillmix.py`. The identity check (`make verify-baseline`, `method=none`) does
not depend on the mixer.
