"""Convex source mixing (``mixing.type: convex``): every subtask segment becomes a convex combination of the source
segments, weighted by the success the task's success model predicts for each source in the current scene.

For one subtask of one attempt:

1. Every source k gets its transform size to the current scene (the displacement and rotation between its reference
   object at the segment start and the current reference object) and a predicted full-task success ``p_k`` from the
   task's ``TransformSuccessModel`` (``tmg.generation.method.transform_model``: one fit per task on the baseline ranking
   pool, the same procedure for every task; ``mixing.model_dir`` picks the model family, for example
   ``transform_model_decay`` for the per-source decay model).
2. The anchor is the source with the largest ``p_k``. Only sources whose gripper open/close events have the anchor's
   initial state and order can be blended with it; the others get weight 0.
3. The weights follow ``mixing.rule``:
   - ``proportional``: ``w_k = p_k / sum p`` over the compatible sources;
   - ``above_mean``: the same, over the compatible sources whose ``p_k`` lies above the mean ``p`` of the compatible
     sources (the anchor always stays);
   - ``best``: the anchor alone (no blend), the single-source control of the rule.
4. The segments are blended with ``event_mix.mix_many`` in their own reference-object frames (events at the weighted
   mean of the event steps, weighted mean positions, weighted rotation mean) and the result is expressed in the frame of
   the source the generator selected, so the generator's own transform moves it to the current scene unchanged.

No value is set by hand: the model is fitted, and the rule has no threshold or temperature.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from tmg.core.poses import invert, transform_all
from tmg.generation.method.event_mix import _matching_events, mix_many
from tmg.generation.method.guided_mix import GuidedMixConfig, GuidedMixer
from tmg.generation.method.interfaces import MixContext, MixedSegment
from tmg.generation.method.transform_model import TransformSuccessModel, transform_size
from tmg.generation.simulators.scenes import generator_config_dir

RULES = ("proportional", "above_mean", "best")
MODEL_DIR = "transform_model"


def convex_weights(p: np.ndarray, compatible: list[int], rule: str) -> np.ndarray:
    """The weights of step 3 of the module docstring (one per source, non-negative, summing to 1)."""
    if rule not in RULES:
        raise ValueError(f"mixing.rule must be one of {RULES}, got {rule!r}")
    p = np.asarray(p, dtype=float)
    w = np.zeros(len(p))
    anchor = max(compatible, key=lambda k: (p[k], -k))
    if rule == "best":
        w[anchor] = 1.0
        return w
    keep = list(compatible)
    if rule == "above_mean":
        mean = float(np.mean(p[keep]))
        keep = [k for k in keep if p[k] > mean or k == anchor]
    total = float(p[keep].sum())
    if total <= 0:                                  # every prediction clipped to 0: the anchor alone
        w[anchor] = 1.0
        return w
    w[keep] = p[keep] / total
    return w


class ConvexMixer(GuidedMixer):
    """The ``SegmentMixer`` of the module docstring."""

    def __init__(self, *, model: TransformSuccessModel, rule: str, chunk_seed: int = 0, gripper_events: Mapping | None = None):
        if rule not in RULES:
            raise ValueError(f"mixing.rule must be one of {RULES}, got {rule!r}")
        super().__init__(ranking=None, chunk_seed=chunk_seed,
                         config=GuidedMixConfig(weight_a=0.5, max_anchor_rotation_deg=1e9, gripper_events=gripper_events))
        self.model, self.rule = model, rule

    @classmethod
    def from_config(cls, cfg: Mapping, *, chunk_seed: int) -> ConvexMixer:
        """``mixing.model``: ``<generator>/<task>`` (``configs/experiment/generator/<generator>/<model_dir>/<task>.yaml``,
        ``mixing.model_dir`` default ``transform_model``) or a file path; ``mixing.rule``: one of ``RULES``."""
        ref = str(cfg["model"])
        path = Path(ref)
        if not path.suffix:
            generator, task = ref.split("/", 1)
            path = generator_config_dir(generator) / str(cfg.get("model_dir", MODEL_DIR)) / f"{task}.yaml"
        return cls(model=TransformSuccessModel.load(path), rule=str(cfg.get("rule", "proportional")), chunk_seed=chunk_seed,
                   gripper_events=cfg.get("gripper_events"))

    def describe(self) -> dict:
        return {"type": "convex", "rule": self.rule, "num_sources": self.model.num_sources, "gripper_events": self.detector.name}

    def mix(self, context: MixContext) -> MixedSegment:
        primary = int(context.primary)
        record = {"call": self.calls, "subtask": int(context.subtask_ind), "source_a": primary, "mixed": False, "rule": self.rule}
        self.calls += 1
        identity = MixedSegment(context.primary_eef_start, context.primary_targets, context.primary_gripper, record)
        current = context.extra.get("current_object_pose")
        if not context.has_reference_object or current is None:
            record["identity_reason"] = "subtask has no reference object" if not context.has_reference_object else "no current object pose"
            return identity
        n = context.num_sources
        sizes = [transform_size(current, context.src_object_poses[k]) for k in range(n)]
        gaps = context.extra.get("inhand_gaps")
        p = self.model.predict(int(context.subtask_ind), [s[0] for s in sizes], [s[1] for s in sizes], current_pose=current,
                               inhand_gaps=gaps)
        series = [self._event_series(np.asarray(context.src_segments[k][2], dtype=float)) for k in range(n)]
        anchor = max(range(n), key=lambda k: (p[k], -k))
        compatible = [k for k in range(n) if k == anchor or _matching_events(series[anchor], series[k]) is not None]
        w = convex_weights(p, compatible, self.rule)
        record.update(predicted=[round(float(v), 4) for v in p], weights=[round(float(v), 4) for v in w], anchor=int(anchor),
                      compatible=compatible, displacement=[round(s[0], 4) for s in sizes], rotation=[round(s[1], 4) for s in sizes])
        if gaps is not None:
            record["inhand_gaps"] = [round(float(v), 4) for v in gaps]
        keep = [k for k in range(n) if w[k] > 0]
        obj = [np.asarray(context.src_object_poses[k], dtype=float) for k in keep]
        poses = [transform_all(invert(o), np.asarray(context.src_segments[k][1], dtype=float)) for o, k in zip(obj, keep)]
        mixed_obj, grip, meta = mix_many(poses, [series[k] for k in keep], w[keep])
        eef_obj = [invert(o) @ np.asarray(context.src_segments[k][0], dtype=float) for o, k in zip(obj, keep)]
        eef_mixed = mix_many([e[None] for e in eef_obj], [np.full(1, series[k][0]) for k in keep], w[keep])[0][0]
        obj_primary = np.asarray(context.src_object_poses[primary], dtype=float)
        gripper = self._grippers(context, keep, w[keep], grip, meta)
        record.update(mixed=len(keep) > 1 or keep[0] != primary, alignment={k: meta[k] for k in ("samples", "segments", "actual_event_steps") if k in meta})
        return MixedSegment(obj_primary @ eef_mixed, transform_all(obj_primary, mixed_obj), gripper, record)

    def _grippers(self, context: MixContext, keep: list[int], w: np.ndarray, grip: np.ndarray, meta: dict) -> np.ndarray:
        """Mixed gripper commands in the selected source's layout: the blended scalar for sign commands, otherwise each
        source's joint commands time-warped to the mixed timeline and blended with the pose weights."""
        g0 = np.asarray(context.src_segments[keep[0]][2], dtype=float)
        if self._scalar_commands(g0.reshape(len(g0), -1)):
            return grip.reshape(-1) if np.asarray(context.primary_gripper).ndim == 1 else grip.reshape(-1, 1)
        if len(keep) == 1:
            return g0
        index = meta["source_index"]           # keyed by the position of each segment in keep
        width = g0.reshape(len(g0), -1).shape[1]
        out = np.zeros((meta["samples"], width))
        for i, (wk, k) in enumerate(zip(w, keep)):
            out += wk * np.asarray(context.src_segments[k][2], dtype=float).reshape(-1, width)[np.asarray(index[str(i)])]
        return out.reshape(-1) if g0.ndim == 1 else out
