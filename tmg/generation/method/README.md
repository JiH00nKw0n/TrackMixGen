# `tmg/generation/method/`: the TrackMix method and its variants

This folder is where method variants are written and compared. It holds the two hooks that every generator
adapter calls, the current implementations behind them, and the pieces they share. Nothing here imports a
simulator, so every class is unit-tested with numpy arrays alone.

```
interfaces.py   MixContext, MixedSegment, SegmentMixer, PathFollower, Method, build_method and the type
                registries _MIXERS / _FOLLOWERS (MixerType, FollowerType protocols)
guided_mix.py   GuidedMixer and its GuidedMixConfig: source mixing guided by the measured ranking (mixing.type: guided);
                its flow is split into steps (_identity_reason, _donor_pool, _choose, _blend_schedule) that variants override
geometric_mix.py TransformMixer (mixing.type: transform) and JunctionMixer (mixing.type: junction): donor and weight
                chosen from the current scene's geometry instead of the ranking (mixup exploration axes 2 and 3)
source_select.py TopKSources (method.sources.top_k): the generator's source choice restricted to the k best-ranked sources
event_mix.py    event-aligned blending of two segments (used by GuidedMixer)
tracking.py     ErrorAdaptiveFollower, EndEffectorPath: error-adaptive following  (tracking.type: error_adaptive)
grippers.py     gripper state and event rules: gripper_states / change_indices (scalar sign commands),
                SignGripper and ClosureGripper detectors (scalar +/- commands, many-joint hands), build_detector
ranking.py      reading the measured source rankings and splitting them into high- and low-success groups
```

## 1. The two hooks

A generator adapter (`tmg/generation/generators/<name>/runtime.py`) calls the method at two points of the
official generation loop. The method never sees the simulator; it sees arrays.

**Segment mixer, once per subtask.** After the official code has selected a source demonstration for the
subtask and cut its segment (end-effector poses, target poses, gripper commands) and before it transforms that
segment to the current object pose, the adapter calls `mixer.mix(context)`.

| `MixContext` field | Meaning |
|---|---|
| `subtask_ind` | index of the subtask |
| `primary` | index of the source the official code selected (A) |
| `primary_eef_start`, `primary_targets`, `primary_gripper` | A's segment: first end-effector pose (4x4), target poses (T, 4, 4), gripper commands (T, G) |
| `src_object_poses` | per source: the reference object's pose (4x4) at that source's segment start, so a mixer can express every segment in the object frame |
| `src_segments` | per source: (eef_start, targets, gripper) of its segment |
| `num_sources`, `has_reference_object` | how many sources exist; whether the subtask has a reference object (no object: no frame to mix in) |
| `rng` | optional (defaults to `default_rng(0)`); `GuidedMixer` ignores it and draws its donor choice from its own generator seeded with `(chunk_seed, call index, seed_offset)`, so a run is reproducible without the adapter passing a stream. A new mixer may use `rng` only if the adapter seeds it deliberately |

The mixer returns a `MixedSegment`: the segment the generator should transform and execute (`eef_start`,
`targets`, `gripper`) and a `record` dict. Returning the inputs unchanged is the identity (the official
behaviour). The record goes into the attempt's `subtasks[i]["mix"]` entry of `attempts.jsonl`; `GuidedMixer`
writes `call`, `subtask`, `mixed`, `source_a`, `weight_a`, `good`, `eligible_donors`, on a mix `source_b`,
`anchor_rotation_degrees` and `alignment`, and when it did nothing `identity_reason`.

**Path follower, once per trajectory.** When the official code starts executing a transformed trajectory (all
waypoint sequences of a subtask, all end effectors), the adapter makes one follower with `method.follower_factory()`
and drives the control loop from it instead of stepping one waypoint per control step.

| Call | What the follower does |
|---|---|
| `start_path(eefs, noises, kinds, previous_states)` | receives the flattened path: one `EndEffectorPath` per end effector (poses (n, 4, 4) and gripper commands (n, G), all of equal length n), the official noise scale per waypoint, the segment kind per waypoint (`SEG_BRIDGE` 0 for interpolation to the segment, `SEG_SKILL` 1 for the segment itself), and the gripper state at the end of the previous path (for event detection across paths) |
| `next_target(measured)` | receives the measured pose (4x4) per end effector and returns `(targets, grippers, noise, record)`: the commanded pose and gripper per end effector, the noise scale of the waypoint being commanded, and a record |
| `finished()` | True once the path variable reached the last waypoint; the adapter then stops the loop |
| `final_gripper_states()` | gripper state per end effector at the end, handed to the next path |

The record of `next_target` must contain these keys, because the execution bookkeeping
(`tmg/generation/execution.py`) and the recorder read them: `k` (index of the base waypoint being commanded),
`new_slots` (base waypoint indices passed for the first time by this step, each of which consumes one official
noise draw so the official random stream stays aligned), `s_before`, `s_after`, `ds_proposed`, `ds_actual`
(the path variable before and after, the proposed and applied advance), `region` (`rest`, `approach` or
`bridge`), `capped` (whether the extra-step cap forced base speed), `kind`, `e_pos`, `e_rot` (the errors used).
A follower that does not have a notion for one of them still fills the key (for example `ds_proposed =
ds_actual`).

## 2. The current implementations and their values

`configs/experiment/method/trackmix.yaml` holds every value; the code only has fallbacks for missing keys.

| Block | Key | Value | Meaning |
|---|---|---|---|
| `mixing` | `type` | `guided` | implementation (`GuidedMixer`) |
| | `weight_a` | 0.8 | share of the primary source in the blend; the donor gets the rest |
| | `good_fraction` | 0.5 | top share of the ranking that counts as high-success (donor candidates) |
| | `good_count` | null | an explicit count instead of the fraction |
| | `max_anchor_rotation_deg` | 20.0 | a donor is compatible only if its hand orientation is within this angle of the primary's at the segment anchors |
| | `ranking` | `<generator>/<task>` | which measured ranking file to read |
| | `seed_offset` | 817263 | seed material of the private mixing RNG |
| `tracking` | `type` | `error_adaptive` | implementation (`ErrorAdaptiveFollower`) |
| | `ds_min` | 0.25 | smallest advance of the path variable per step |
| | `regions` | gains per region | `alpha_p`, `alpha_r`: how strongly position and rotation error slow the advance; `default` and `approach` (near gripper events) |
| | `approach_window` | 10 | waypoints before a gripper event that use the approach gains |
| | `extra_steps_cap` | `segment` | after as many extra steps as the path has waypoints, fall back to base speed |
| | `gripper_events` | `{type: sign}` | event detector: `sign` for scalar open/close commands, `closure` for many-joint hands (columns given) |
| `labels` | | `pre_noise` | the dataset stores the clipped pre-noise action (`executed` stores what the simulator received) |

`GuidedMixer` does the following for a subtask: if the primary source is in the high-success group, or the
subtask has no reference object, or no donor is compatible, it returns the identity and records why. Otherwise
it picks a compatible donor from the high-success group, expresses both segments in the reference-object frame,
aligns them at gripper events (`event_mix.mix_segments`), blends poses with `weight_a` (`blend_poses`), and
returns the blend, plus the primary's first end-effector pose blended the same way.

`ErrorAdaptiveFollower` advances a path variable s along the flattened waypoints. Each step it measures the
position and rotation error between the measured pose and the commanded pose and sets the advance to
`max(ds_min, 1 / (1 + alpha_p e_pos + alpha_r e_rot))` with the gains of the current region, never crossing a
gripper event before the gripper reached its state. Steps beyond the path's own length are capped
(`extra_steps_cap`). Multi-arm paths share one s, so both hands stay aligned in source time.

## 3. Adding a variant

**Same algorithm, other values.** Add one file to `configs/experiment/method/`:

```yaml
# configs/experiment/method/trackmix_w07.yaml
defaults:
  - trackmix
  - _self_
name: trackmix_w07
mixing: {weight_a: 0.7}
```

The `name` is the `<method>` part of every output path (pools, datasets, runs), so the variant's results never
mix with those of `trackmix`. A command-line override (`method.mixing.weight_a=0.7`) also works but writes into
the `trackmix` folders; use it only for a quick look.

**Another algorithm.** Three steps, no adapter changes:

1. Write the class in this folder. A mixer implements `mix(context) -> MixedSegment` and a class method
   `from_config(cfg, chunk_seed=...)` that reads its own values from the `mixing` block (`MixerType` in
   `interfaces.py`). A follower implements the four `PathFollower` methods and a class method `factory(cfg)` that
   returns a zero-argument constructor, one follower per trajectory (`FollowerType`). Seed every random draw from
   the chunk seed yourself, as `GuidedMixer` does with `SeedSequence([chunk_seed, call, seed_offset])`; the
   adapters do not seed `context.rng`. Put everything you want to analyse later into the returned `record`.
2. Register the class under a name in `interfaces.py`: followers in the `_FOLLOWERS` dict, mixers in
   `_mixer_types()` (filled lazily because `guided_mix` imports `MixContext` from `interfaces`).
3. Add a method file that names it (`mixing: {type: <name>, ...}` or `tracking: {type: <name>, ...}`) and lists
   its values. An unknown type stops `build_method` with the list of known names.

Skeleton of a follower variant:

```python
class ConstantSpeedFollower:
    """Advances one waypoint per step (the official schedule), but through the follower interface."""

    def __init__(self, *, detector):
        self.detector = detector

    @classmethod
    def factory(cls, cfg: dict):
        from tmg.generation.method.grippers import build_detector
        return lambda: cls(detector=build_detector(cfg.get("gripper_events")))

    def start_path(self, eefs, noises, kinds, previous_states=None):
        self.eefs, self.noises, self.kinds, self.k = eefs, list(noises), list(kinds), -1

    def finished(self):
        return self.k >= len(self.eefs[0].poses) - 1

    def final_gripper_states(self):
        return {e.name: bool(e.grippers[-1][0] > 0) for e in self.eefs}

    def next_target(self, measured):
        self.k += 1
        targets = {e.name: e.poses[self.k] for e in self.eefs}
        grippers = {e.name: e.grippers[self.k] for e in self.eefs}
        record = {"k": self.k, "new_slots": [self.k], "s_before": self.k - 1.0, "s_after": float(self.k),
                  "ds_proposed": 1.0, "ds_actual": 1.0, "region": "rest", "capped": False, "kind": self.kinds[self.k],
                  "e_pos": 0.0, "e_rot": 0.0}
        return targets, grippers, self.noises[self.k], record
```

**Testing.** Write the unit test next to the existing ones (`tests/unit/test_tracking.py`, `test_event_mix.py`):
build the class from a config dict, feed it arrays, check the targets and the record. `build_method` is tested
in `test_tracking.py` (`test_build_method_picks_implementations_by_type`). The identity check
(`tmg verify_baseline`) must still pass with `method=none`; a variant must not change that path.

**Comparing.** Run the same pipeline with the variant's name: `make generate ... METHOD=<name>`, then `make
dataset`, `make train`, `make validate`, `make evaluate` with the same `METHOD`. The resolved config of every
output folder records the exact values used, and the attempt records keep the per-step and per-subtask records
the classes wrote, so a variant can be analysed from the pool alone before any policy is trained.

## 4. Mixup exploration conditions (`configs/experiment/method/mx_*.yaml`)

Every `mx_*` condition turns path following off and labels the actions before noise, so conditions differ only in
how sources are chosen and mixed. Mixing replaces the selected source's segment, in its reference-object frame,
before the generator transforms it to the current scene; because the transform is the same rigid map for both
segments, this equals transforming each source first and mixing the transformed segments.

| Condition | What changes |
|---|---|
| `mx_none` | nothing: the generator as released |
| `mx_top1`, `mx_top3`, `mx_top5` | every source choice is a uniform draw among the k best-ranked sources (`TopKSources`); the generator's own selection still runs first so its random draws are unchanged |
| `mx_success_a20`, `_a40`, `_a90`, `_anone` | axis 1, `GuidedMixer`: a low-ranked primary is blended 0.8 : 0.2 with a high-ranked donor whose hand orientation at the anchors is within the bound (180 = no bound) |
| `mx_transform` | axis 2, `TransformMixer`: when the primary's reference-object pose is far from the current one (above the median of all sources), blend with the compatible source of the smallest transform; primary share `clip(d_b / (d_a + d_b), 0.2, 0.8)` |
| `mx_junction` | axis 3, `JunctionMixer`: when the primary's transformed start is far from the pose the generator interpolates from, blend the first `head_steps` steps with the source whose transformed start is nearest; the primary's share rises linearly from `clip(c_b / (c_a + c_b), 0.2, 0.8)` to 1 |

The geometric mixers read two poses of the current scene from `MixContext.extra` (`current_object_pose`,
`junction_pose`); the MimicGen runtime provides them (`MimicGenRuntime.scene_poses`). Adapters that do not provide
them get the identity with the reason recorded. CP-Gen refuses any mixer for now (it re-solves constraints instead
of replaying segments; its transform-then-mix variant is not wired yet); the top-K rule works on CP-Gen with
`selection=ours`, one source per attempt from subtask 0's ranking.

