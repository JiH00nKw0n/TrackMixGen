# `tmg/generation/method/`: SkillMix

This folder holds the one hook that every generator adapter calls (the segment mixer), its implementation, SkillMix,
and the pieces it shares. Nothing here imports a simulator, so every class is unit-tested with numpy arrays alone. Two
methods exist (`configs/experiment/method/`): `none`, the generator as released, and `skillmix`.

```
interfaces.py    MixContext, MixedSegment, SegmentMixer, Method, build_method and the mixer registry _MIXERS (MixerType);
                 the value objects every runtime, the mixer, the records and the fit share: Arms and SubtaskId (which end
                 effectors number their own subtasks; the id of one subtask), SubtaskScene (the scene of a subtask as
                 SkillMix scores it), HeldRule / HeldTest (when a hand counts as holding the carried object)
skillmix.py      SkillMixer (mixing.type: skillmix), the rule file (SkillMixRule, load_rule), softmax_weights,
                 RandmixDraws / randmix_weights (mixing.rule: randmix), symmetry alignment
success_model.py the per-source features of a SubtaskScene (scene, in-hand, hold mismatch), the same function for the mixer
                 and for the fit's rows (scene_of_record rebuilds the scene from a subtask record)
fit_layout.py    FitLayout: what a generator adapter declares about its records (arms, reached rule, where the source
                 poses come from, features, sanity limits); the one fit path of tmg fit_coefficients reads it
fit_records.py   the records as the fit reads them: record versions, checks, the reached rules (REACHED), the sanity report
soft_frame.py    the object frame of a deformable object from its nodes (SoftMimicGen rope, towel), for the features
event_mix.py     event-aligned blending of segments (mix_many; align_segments for CP-Gen)
grippers.py      gripper state and event rules: gripper_states / change_indices (scalar sign commands),
                 SignGripper and ClosureGripper detectors (scalar +/- commands, many-joint hands), build_detector
```

## SkillMix

Per subtask k of an attempt, every source j gets five features against the current scene, `phi_j` = (|dx|, |dy|, rotation
of the reference object; in-hand position and rotation distances from the second subtask on). The rule file
(`configs/experiment/generator/<generator>/coefficients/<task>.yaml`, written by `tmg fit_coefficients`) holds, per subtask,
the fitted mixed-effects logistic model of the task success given the mixing weights `w`:

    P(task succeeds | reached k, w) = sigmoid( c0 + sum_j w_j s_j + d H(w) ),   s_j = a_j + g . phi_j

Two-armed generators (DexMimicGen's GR1) number the subtasks of every arm from 0, so their rule files key the blocks
`<eef>/<k>` (`SubtaskId.key`); single-arm files (MimicGen, CP-Gen, SoftMimicGen) keep the keys `0, 1, ...`. The arms come
from the generator's layout (the generator config's `eefs`): the runtime builds the mixer with them
(`build_method(..., arms=...)`) and the adapter's fit layout names the same ones, so the mixer and the fit cannot disagree.
A generator with a hold test (the Isaac Lab runtimes: is the previous subtask's reference object in the hand?) has a sixth
feature, `inhand_mismatch` (1 when exactly one of the two hands, now or source j, holds the object), so "source j does not
hold it" is not read as "source j holds it exactly as now"; its rule file lists its `features` and records the `held_rule`
(and, for a deformable object, the `object_frame`) its records were measured with, and the mixer refuses a scene measured
otherwise.
`a_j` is source j's own effect, `g` the feature slopes, `d` the effect of spreading the weight (`H` the entropy of `w`). The
weights that maximize it over the sources compatible with the selected one (same gripper events) are `softmax(s / d)` when
`d > 0`; weights below 1e-3 are dropped. When `d <= 0` one source takes all the weight: the best-scored one (`mixing.single_source: best`, method `skillmix`) or one drawn with probability `softmax(s)` (`sample`, method `skillmix_sample`; with `best` every scene can get the same source). The retained segments are aligned
by the object and gripper symmetries and blended in their reference-object frames (`skillmix.py` module docstring). CP-Gen
blends its re-solved constraints itself (`tmg/generation/generators/cpgen/mixing.py`) and takes the weights and their record
from the same call, `SkillMixer.decide`, so both generators apply one rule.

**Rule files, from scratch.** `make coefficients GENERATOR=<g> TASK=<t>` (`scripts/ensure_coefficients.sh`) builds two
2,000-attempt pools of the same settings and chunk seeds (selection `ours`, generation `pool_unmix`; chunks of 50, CP-Gen
25 from its generator config, the sizes of the committed rules' pools): an unmix pool (method `none`, `none/pool_unmix`) and a randmix pool
(`skillmix_randmix/pool_randmix`, method `skillmix` with `mixing.rule=randmix` and `method.name=skillmix_randmix`: random
Dirichlet weights), then fits the model on both (`tmg fit_coefficients`). `make generate METHOD=skillmix` does this by
itself when the rule file is missing. The committed files (MimicGen: seven tasks, CP-Gen: five) were fitted on 2026-11-20;
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
| `scene` | the `SubtaskScene`: the reference object now and in every source, and from the second subtask on the end effector and the carried object (the previous subtask's reference object) now and in every source; with a hold test, its rule and distances. `None`: the mixer leaves the segment as it is |

The mixer returns a `MixedSegment`: the segment the generator should transform and execute (`eef_start`, `targets`,
`gripper`) and a `record` dict. Returning the inputs unchanged is the identity (the official behaviour). The record goes
into the attempt's `subtasks[i]["mix"]` entry of `attempts.jsonl`; SkillMix writes `call`, `subtask`, `source_a`, `rule`,
`mixed`, `weights`, `incompatible`, `scene_features` (and from the second subtask on `inhand_features`; with a hold test
`held`: who holds the carried object, now and per source), with the softmax rule `scores` and `d`, with the randmix rule
`dirichlet_c`, on a blend `symmetry` and `alignment`, and when it did nothing `identity_reason`. The fit reads `weights`,
`scene_features`, `inhand_features` and `held` of the randmix pools.

## 1b. A new generator

Nothing in this folder or in `tmg/generation/fit_coefficients.py` names a generator. A new generator plugs in through its
adapter and config: its runtime fills a `SubtaskScene` per subtask and builds the method with its arms; its adapter
declares `fit_layout()`; and its generator config says what shared code asks about it (`simulator`, `byte_verify`,
`gpu_variable`, `default_max_workers`, and for a hold test `scene_reading` and `record_checks`). What it needs beyond the
existing pieces it registers from its own modules: a reached rule (`fit_records.register_reached_rule`), a feature group
(`success_model.register_features`: names, units, how to compute it from a scene and read it from a mix record), a mixer
(`interfaces.register_mixer`).

Who holds the carried object is asked in one place, `SubtaskScene.holds_now()` / `source_holds(j)`: the hold test's
verdict where the runtime makes one (rebuilt from the recorded distances and rule for a scene read back from a record),
else whether the pose is known. The in-hand features, the mismatch, the records and the sanity report all use it.

## 2. Current values

`configs/experiment/method/skillmix.yaml` holds every value:

| Block | Key | Value | Meaning |
|---|---|---|---|
| `mixing` | `type` | `skillmix` | implementation (`SkillMixer`) |
| | `rule` | `softmax` | `softmax`: the SkillMix weights; `randmix`: random Dirichlet weights for the randmix pools only |
| | `coefficients` | `<generator>/<task>` | which rule file to read |
| | `symmetries` | `task.mixing_symmetries` | object symmetries the sources may be aligned with before blending |
| | `gripper_events` | `{type: sign}` (default) | DexMimicGen: `{type: closure, columns: [0, 1, 2, 3]}` (`method_overrides` of its generator file) |
| `labels` | | `executed` | the dataset stores what the simulator received (CP-Gen: `pre_noise`, its saved actions are noise-free) |

A generator overrides values under `method_overrides: {<method name>: {...}}` in its own file; the randmix pools have
the method name `skillmix_randmix`, so a generator that overrides `skillmix` lists that name too.

## 3. Adding a mixer variant

1. Write the class next to `skillmix.py`: a `mix(context) -> MixedSegment` method, a class method `from_config(cfg, *,
   chunk_seed, method_name, arms)`, and `begin_attempt()` if it keeps per-attempt state. Seed every random draw from the chunk seed.
2. Register it under a name at the end of its module (`interfaces.register_mixer(name, cls)`); a mixer outside this
   package is found through the method config's `mixing.module`.
3. Add a method file that names it (`mixing: {type: <name>, ...}`) under its own `name`, so its pools, datasets and runs
   get their own folders.

Write its unit test next to `tests/unit/test_skillmix.py`. The identity check (`make verify-baseline`, `method=none`) does
not depend on the mixer.
