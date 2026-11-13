"""Logistic source mixing (``mixing.type: logistic``): at the start of every subtask, predict each source's success in the
current scene with the task's per-subtask logistic model and blend the sources whose prediction is at least the mean.

For subtask k of one attempt:

1. Every source j gets its features against the current scene (``tmg.generation.method.success_model``): the x and y
   distances and the rotation angle between the current reference object and source j's at its segment start, and from
   the second subtask on the position distance and rotation angle between the in-hand pose now and source j's. The
   coefficient file (``configs/experiment/generator/<generator>/coefficients/<task>.yaml``, ``mixing.coefficients``) turns
   them into predicted probabilities ``p_j``.
2. The retained set is ``A = {j : p_j >= mean_i p_i}`` (the mean over all J sources). Sources whose gripper open/close
   events do not match those of the best-predicted source cannot be time-aligned with it and are dropped from A (the
   record lists them under ``incompatible``); the best-predicted source always stays.
3. The weights are ``w_j = p_j / sum_{i in A} p_i`` on A and 0 elsewhere.
4. Symmetry alignment. A source can hold the object in a different but equivalent way: the parallel gripper is the same
   after 180 degrees about its approach axis (the end-effector z axis; only for scalar open/close commands), and an
   object may be symmetric (``mixing.symmetries``: per reference object a list of ``{rot_axis, rot_param, translation}``
   transforms in CP-Gen's annotation format, from the task config's ``mixing_symmetries``; Hammer Cleanup's hammer turned
   180 degrees about its handle axis). Every retained source other than the selected one is re-expressed with the
   combination ``inv(S_obj) @ pose @ S_eef`` closest to the selected source at the key step (the first gripper event of
   each segment, else its last step), with CP-Gen's ``find_best_transform`` score (5 per metre plus 1 per radian). The
   record keeps per source the chosen combination and its distance to the selected source after alignment and without
   it. Without this step two sources that grasp the hammer from opposite sides are averaged into
   a pose between them, rotated about 180 degrees from both (CP-Gen Hammer Cleanup pilot 0/20, 2026-11-13).
5. The retained segments are blended in their own reference-object frames (``event_mix.mix_many``: events at the weighted
   mean event steps, weighted mean positions, weighted rotation mean) and the result is expressed in the frame of the
   source the generator selected, so the generator's own transform moves it to the current scene unchanged.

No threshold or temperature is set by hand. The record of every call keeps the predictions, the features and the weights.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tmg.core.poses import invert, transform_all
from scipy.spatial.transform import Rotation

from tmg.generation.method.event_mix import _matching_events, events, mix_many
from tmg.generation.method.guided_mix import GuidedMixConfig, GuidedMixer
from tmg.generation.method.interfaces import MixContext, MixedSegment
from tmg.generation.method.success_model import SuccessModel, inhand_features, load_coefficients, scene_features


def above_mean_weights(p: np.ndarray, compatible: list[int]) -> np.ndarray:
    """Step 2 and 3 of the module docstring: weights proportional to ``p`` over the compatible sources whose prediction is
    at least the mean over all sources (the best-predicted compatible source always stays)."""
    p = np.asarray(p, dtype=float)
    w = np.zeros(len(p))
    best = max(compatible, key=lambda k: (p[k], -k))
    mean = float(p.mean())
    keep = [k for k in compatible if p[k] >= mean or k == best]
    total = float(p[keep].sum())
    if total <= 0:
        w[best] = 1.0
        return w
    w[keep] = p[keep] / total
    return w


W_TRANS, W_ROT = 5.0, 1.0                    # CP-Gen find_best_transform's score: 5 per metre, 1 per radian


def symmetry_matrix(params: Mapping) -> np.ndarray:
    """4x4 transform of one ``{rot_axis, rot_param, translation}`` entry (CP-Gen's ``generate_se3_transforms``)."""
    out = np.eye(4)
    if "rot_axis" in params and "rot_param" in params:
        out[:3, :3] = Rotation.from_euler(str(params["rot_axis"]), float(params["rot_param"])).as_matrix()
    if params.get("translation") is not None:
        shift = np.eye(4)
        shift[:3, 3] = np.asarray(params["translation"], dtype=float)
        out = shift @ out
    return out


def _symmetry_label(kind: str, params: Mapping, k: int) -> str:
    if "rot_axis" in params and "rot_param" in params:
        return f"{kind}:{params['rot_axis']}{int(round(np.degrees(float(params['rot_param']))))}"
    return f"{kind}:{k}"


def symmetry_candidates(object_symmetries, parallel_gripper: bool) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """``(label, S_obj, S_eef)`` combinations (module docstring, step 4), identity first."""
    objs = [("", np.eye(4))] + [(_symmetry_label("obj", p, k), symmetry_matrix(p)) for k, p in enumerate(object_symmetries or [])]
    eefs = [("", np.eye(4))]
    if parallel_gripper:
        flip = np.eye(4)
        flip[:3, :3] = Rotation.from_euler("z", np.pi).as_matrix()
        eefs.append(("eef:z180", flip))
    return [("+".join(x for x in (lo, le) if x) or "identity", so, se) for le, se in eefs for lo, so in objs]


def key_step(series) -> int:
    """The first gripper event of a segment, else its last step (module docstring, step 4)."""
    change = events(np.asarray(series, dtype=float).reshape(-1))[1]
    return int(change[0]) if len(change) else len(series) - 1


def pose_distance(a, b) -> tuple[float, float]:
    """``(position distance m, rotation angle rad)`` between two 4x4 poses."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())


def align_to_reference(poses, eef, key: int, reference, candidates) -> tuple[np.ndarray, np.ndarray, dict]:
    """Re-express one source's object-frame ``poses`` (T,4,4) and start pose ``eef`` (4x4) with the candidate closest to
    ``reference`` at step ``key``: ``(poses, eef, record)``."""
    poses = np.asarray(poses, dtype=float)
    best = None
    for label, so, se in candidates:
        dp, dr = pose_distance(invert(so) @ poses[key] @ se, reference)
        score = W_TRANS * dp + W_ROT * dr
        if best is None or score < best[0]:
            best = (score, label, so, se, dp, dr)
    _score, label, so, se, dp, dr = best
    dp0, dr0 = pose_distance(poses[key], reference)
    record = {"symmetry": label, "pos_cm": round(100 * dp, 2), "rot_deg": round(float(np.degrees(dr)), 1),
              "identity_pos_cm": round(100 * dp0, 2), "identity_rot_deg": round(float(np.degrees(dr0)), 1)}
    if label == "identity":
        return poses, np.asarray(eef, dtype=float), record
    return transform_all(invert(so), poses) @ se, invert(so) @ np.asarray(eef, dtype=float) @ se, record


class LogisticMixer(GuidedMixer):
    """The ``SegmentMixer`` of the module docstring."""

    def __init__(self, *, model: SuccessModel, chunk_seed: int = 0, gripper_events: Mapping | None = None,
                 symmetries: Mapping | None = None):
        super().__init__(ranking=None, chunk_seed=chunk_seed,
                         config=GuidedMixConfig(weight_a=0.5, max_anchor_rotation_deg=1e9, gripper_events=gripper_events))
        self.model = model
        self.symmetries = {str(k): [dict(p) for p in v] for k, v in (symmetries or {}).items()}   # object -> transforms

    @classmethod
    def from_config(cls, cfg: Mapping, *, chunk_seed: int) -> LogisticMixer:
        """``mixing.coefficients``: ``<generator>/<task>`` or a path to a coefficient file."""
        return cls(model=load_coefficients(str(cfg["coefficients"])), chunk_seed=chunk_seed, gripper_events=cfg.get("gripper_events"),
                   symmetries=cfg.get("symmetries"))

    def describe(self) -> dict:
        return {"type": "logistic", "rule": "above_mean", "coefficients": self.model.describe(), "gripper_events": self.detector.name,
                "symmetries": self.symmetries}

    def predict(self, context: MixContext) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """``(p, scene features (J,3), in-hand features (J,2) or None)`` of every source for the context's subtask."""
        k = int(context.subtask_ind)
        sub = self.model.for_subtask(k)
        current = context.extra["current_object_pose"]
        scene = np.array([scene_features(current, context.src_object_poses[j]) for j in range(context.num_sources)])
        inhand = None
        if k >= 1 and "current_in_hand" in context.extra:
            now = context.extra["current_in_hand"]
            inhand = np.array([inhand_features(now, src) for src in context.extra["source_in_hand"]])
        return sub.predict(scene, inhand), scene, inhand

    def mix(self, context: MixContext) -> MixedSegment:
        primary = int(context.primary)
        record = {"call": self.calls, "subtask": int(context.subtask_ind), "source_a": primary, "mixed": False, "rule": "above_mean"}
        self.calls += 1
        identity = MixedSegment(context.primary_eef_start, context.primary_targets, context.primary_gripper, record)
        if not context.has_reference_object or context.extra.get("current_object_pose") is None:
            record["identity_reason"] = "subtask has no reference object" if not context.has_reference_object else "no current object pose"
            return identity
        n = context.num_sources
        p, scene, inhand = self.predict(context)
        series = [self._event_series(np.asarray(context.src_segments[j][2], dtype=float)) for j in range(n)]
        best = max(range(n), key=lambda j: (p[j], -j))
        compatible = [j for j in range(n) if j == best or _matching_events(series[best], series[j]) is not None]
        w = above_mean_weights(p, compatible)
        record.update(predicted=[round(float(v), 4) for v in p], weights=[round(float(v), 4) for v in w], best=int(best),
                      incompatible=[j for j in range(n) if j not in compatible],
                      scene_features=np.round(scene, 4).tolist())
        if inhand is not None:
            record["inhand_features"] = np.round(inhand, 4).tolist()
        keep = [j for j in range(n) if w[j] > 0]
        obj = [np.asarray(context.src_object_poses[j], dtype=float) for j in keep]
        poses = [transform_all(invert(o), np.asarray(context.src_segments[j][1], dtype=float)) for o, j in zip(obj, keep)]
        eef_obj = [invert(o) @ np.asarray(context.src_segments[j][0], dtype=float) for o, j in zip(obj, keep)]
        symmetry = self._align(context, keep, series, poses, eef_obj)
        if symmetry:
            record["symmetry"] = symmetry
        mixed_obj, grip, meta = mix_many(poses, [series[j] for j in keep], w[keep])
        eef_mixed = mix_many([e[None] for e in eef_obj], [np.full(1, series[j][0]) for j in keep], w[keep])[0][0]
        obj_primary = np.asarray(context.src_object_poses[primary], dtype=float)
        gripper = self._grippers(context, keep, w[keep], grip, meta)
        record.update(mixed=len(keep) > 1 or keep[0] != primary,
                      alignment={key: meta[key] for key in ("samples", "segments", "actual_event_steps") if key in meta})
        return MixedSegment(obj_primary @ eef_mixed, transform_all(obj_primary, mixed_obj), gripper, record)

    def _align(self, context: MixContext, keep: list[int], series: list, poses: list, eef_obj: list) -> dict:
        """Step 4 of the module docstring, in place on ``poses`` and ``eef_obj`` (object frames of the ``keep`` sources)."""
        primary = int(context.primary)
        g = np.asarray(context.src_segments[primary][2], dtype=float)
        candidates = symmetry_candidates(self.symmetries.get(str(context.extra.get("reference_object"))),
                                         self._scalar_commands(g.reshape(len(g), -1)))
        if len(candidates) == 1 or keep == [primary]:
            return {}
        frame = invert(np.asarray(context.src_object_poses[primary], dtype=float))
        reference = frame @ np.asarray(context.src_segments[primary][1], dtype=float)[key_step(series[primary])]
        out = {}
        for i, j in enumerate(keep):
            if j == primary:
                continue
            poses[i], eef_obj[i], out[str(j)] = align_to_reference(poses[i], eef_obj[i], key_step(series[j]), reference, candidates)
        return out

    def _grippers(self, context: MixContext, keep: list[int], w: np.ndarray, grip: np.ndarray, meta: dict) -> np.ndarray:
        """Mixed gripper commands in the selected source's layout: the blended scalar for sign commands, otherwise each
        source's joint commands time-warped to the mixed timeline and blended with the pose weights."""
        g0 = np.asarray(context.src_segments[keep[0]][2], dtype=float)
        if self._scalar_commands(g0.reshape(len(g0), -1)):
            return grip.reshape(-1) if np.asarray(context.primary_gripper).ndim == 1 else grip.reshape(-1, 1)
        if len(keep) == 1:
            return g0
        index = meta["source_index"]
        width = g0.reshape(len(g0), -1).shape[1]
        out = np.zeros((meta["samples"], width))
        for i, (wk, k) in enumerate(zip(w, keep)):
            out += wk * np.asarray(context.src_segments[k][2], dtype=float).reshape(-1, width)[np.asarray(index[str(i)])]
        return out.reshape(-1) if g0.ndim == 1 else out
