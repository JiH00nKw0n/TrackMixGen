"""SkillMix (``mixing.type: skillmix``): at the start of every subtask, blend the source demonstrations with the weights
that maximize the task success predicted by a mixed-effects logistic model of the mixing weights.

The model (fitted by ``tmg fit_coefficients``, one block per subtask k in the rule file
``configs/experiment/generator/<generator>/coefficients/<task>.yaml``, ``mixing.coefficients``):

    P(task succeeds | reached k, weights w) = sigmoid( c0 + sum_j w_j s_j + d * H(w) ),   s_j = a_j + g . phi_j

- ``phi_j`` are source j's five features against the current scene (``tmg.generation.method.success_model``): |dx|, |dy|
  and the rotation angle of the reference object, and from the second subtask on the in-hand position and rotation
  distances (0 for subtask 0);
- ``a_j`` is source j's own effect, ``g`` the five feature slopes shared by all sources, ``d`` the effect of spreading the
  weight (``H(w) = -sum_j w_j log w_j``, 0 for a single source) and ``c0`` the intercept;
- the weights are a point of the simplex over the sources compatible with the selected source.

Maximizing ``sum_j w_j s_j + d H(w)`` over the simplex gives the softmax rule (``mixing.rule: softmax``):

    w_j = exp(s_j / d) / sum_i exp(s_i / d)   over the compatible sources, when d > 0;
    w = one-hot on one compatible source, when d <= 0 (spreading does not help).

With ``d <= 0`` which source gets all the weight is ``mixing.single_source``:

- ``best``: the compatible source with the largest ``s_j``. When the sources' own effects ``a_j`` differ by more than the
  scene terms move them, every scene gets the same source (CP-Gen Threading 2026-11-23: all 1,254 successes from the
  same two sources, one per subtask);
- ``sample``: a source drawn with probability ``softmax(s)`` over the compatible sources, i.e. proportional to
  ``exp(s_j)``, the factor by which source j multiplies the predicted odds of task success. Better sources are drawn
  more often and every source keeps a share. One uniform number per subtask and attempt comes from a stream of its own,
  seeded with ``(chunk seed, SOURCE_SEED)`` (``SourceDraws``), so a pool is reproducible and a re-solved constraint keeps
  its source.

Weights below ``MIN_WEIGHT`` are dropped and the rest renormalized. Nothing is set by hand: the scores, the spread effect
(and with it the temperature) come from the fit. Sources whose gripper open/close events do not match those of the
selected source cannot be time-aligned with it and get weight 0 (the record lists them under ``incompatible``).

``mixing.rule: randmix`` makes the pools the fit reads: the weights are drawn at random,
``w ~ Dirichlet(c, ..., c)`` over the compatible sources with ``c ~ LogUniform(0.05, 20)`` per subtask and attempt (small c:
nearly one source, large c: nearly uniform), so the fit sees how task success changes with the weights. The draws come
from a stream of their own seeded with ``(chunk seed, 727272)`` (``RandmixDraws``). Randmix pools are not
training data; they are written under the method name ``skillmix_randmix``.

The blend itself (both rules), for subtask k of one attempt:

1. Symmetry alignment. A source can hold the object in a different but equivalent way: the parallel gripper is the same
   after 180 degrees about its approach axis (the end-effector z axis; only for scalar open/close commands), and an
   object may be symmetric (``mixing.symmetries``: per reference object a list of ``{rot_axis, rot_param, translation}``
   transforms in CP-Gen's annotation format, from the task config's ``mixing_symmetries``; Hammer Cleanup's hammer turned
   180 degrees about its handle axis). Every retained source other than the selected one is re-expressed with the
   combination ``inv(S_obj) @ pose @ S_eef`` closest to the selected source at the key step (the first gripper event of
   each segment, else its last step), with CP-Gen's ``find_best_transform`` score (5 per metre plus 1 per radian). The
   record keeps per source the chosen combination and its distance to the selected source after alignment and without
   it. Without this step two sources that grasp the hammer from opposite sides are averaged into a pose between them,
   rotated about 180 degrees from both (CP-Gen Hammer Cleanup pilot 0/20, 2026-11-13).
2. The retained segments are blended in their own reference-object frames (``event_mix.mix_many``: events at the weighted
   mean event steps, weighted mean positions, weighted rotation mean) and the result is expressed in the frame of the
   source the generator selected, so the generator's own transform moves it to the current scene unchanged.

The record of every call keeps the rule, the scores, the spread effect, the features and the weights
(``SkillMixer.decide``). CP-Gen re-solves constraints instead of replaying segments: it blends solved poses in its own
module (``tmg.generation.generators.cpgen.mixing``) but takes the weights, the record and the symmetry score from here,
so both generators apply one rule.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from tmg.core.poses import invert, transform_all
from tmg.generation.method.event_mix import _matching_events, events, mix_many
from tmg.generation.method.grippers import SignGripper, build_detector
from tmg.generation.method.interfaces import MixContext, MixedSegment
from tmg.generation.method.success_model import FEATURES, inhand_features, scene_features
from tmg.generation.simulators.scenes import generator_config_dir

RULES = ("softmax", "randmix")           # mixing.rule
RANDMIX_METHOD = "skillmix_randmix"  # method name (pool, dataset and run folder) of the randmix rule
COEFFICIENTS_DIR = "coefficients"
MIN_WEIGHT = 1e-3                            # smaller weights are dropped and the rest renormalized
RANDMIX_SEED = 727272                    # second seed word of the randmix draws (the chunk seed is the first)
SOURCE_SEED = 737373                     # second seed word of the single-source draws of mixing.single_source=sample
SINGLE_SOURCE = ("best", "sample")       # mixing.single_source: the one source of the softmax rule when d <= 0
RANDMIX_LOG_C = (np.log(0.05), np.log(20.0))   # c ~ LogUniform(0.05, 20) of the Dirichlet(c, ..., c) weights


# rule file ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SubtaskRule:
    """The fitted block of one subtask: source effects ``a`` (J,), feature slopes ``g`` (5,), spread effect ``d``, intercept
    ``c0``."""
    a: np.ndarray
    g: np.ndarray
    d: float
    c0: float

    def scores(self, scene, inhand=None) -> np.ndarray:
        """``s_j = a_j + g . phi_j`` for every source. ``scene`` (J, 3); ``inhand`` (J, 2) or None (zeros)."""
        scene = np.asarray(scene, dtype=float)
        inhand = np.zeros((len(scene), 2)) if inhand is None else np.asarray(inhand, dtype=float)
        return self.a + np.hstack([scene, inhand]) @ self.g


@dataclass(frozen=True)
class SkillMixRule:
    """A task's rule file."""
    name: str                    # "<generator>/<task>"
    num_sources: int
    subtasks: dict               # subtask index -> SubtaskRule
    path: str = ""

    def for_subtask(self, k: int) -> SubtaskRule:
        if int(k) not in self.subtasks:
            raise KeyError(f"SkillMix rule {self.name}: no subtask {k}")
        return self.subtasks[int(k)]

    @classmethod
    def from_mapping(cls, data: Mapping, *, name: str = "", path: str = "") -> SkillMixRule:
        if data.get("rule") != "softmax":
            raise ValueError(f"{path or name}: not a SkillMix softmax rule file (rule: {data.get('rule')!r}); refit it with "
                             f"`make fit-coefficients`")
        n = int(data["num_sources"])
        subtasks = {}
        for k, block in (data.get("subtasks") or {}).items():
            a = np.array([float(block["a"][j]) for j in range(n)])
            g = np.array([float(block["g"][f]) for f in FEATURES])
            subtasks[int(k)] = SubtaskRule(a=a, g=g, d=float(block["d"]), c0=float(block["c0"]))
        if not subtasks:
            raise KeyError(f"{path or name}: rule file has no subtasks")
        return cls(name=name or f"{data.get('generator', '')}/{data.get('task', '')}", num_sources=n, subtasks=subtasks,
                   path=path)

    def describe(self) -> dict:
        return {"name": self.name, "path": self.path, "num_sources": self.num_sources, "subtasks": sorted(self.subtasks)}


def rule_path(name: str, root: Path | None = None) -> Path:
    """``configs/experiment/generator/<generator>/coefficients/<task>.yaml`` for a ``<generator>/<task>`` name."""
    generator, _, task = str(name).partition("/")
    if not task:
        raise ValueError(f"rule name must be <generator>/<task>, got {name!r}")
    return generator_config_dir(generator, root) / COEFFICIENTS_DIR / f"{task}.yaml"


def load_rule(name: str, root: Path | None = None) -> SkillMixRule:
    """Read a rule file by its ``<generator>/<task>`` name (or a path ending in .yaml)."""
    path = Path(name) if str(name).endswith(".yaml") else rule_path(name, root)
    if not path.exists():
        raise FileNotFoundError(
            f"SkillMix rule file {path} not found. Build it with `make coefficients GENERATOR=<gen> TASK=<task>` (a "
            f"randmix pool and an unmix pool, then tmg fit_coefficients); `make generate` does this by itself for "
            f"method skillmix.")
    return SkillMixRule.from_mapping(yaml.safe_load(path.read_text()) or {}, name=str(name), path=str(path))


# weights --------------------------------------------------------------------------------------------------------------
def _drop_small(w: np.ndarray) -> np.ndarray:
    w = np.where(w < MIN_WEIGHT, 0.0, w)
    return w / w.sum()


def single_source(scores, compatible: list[int], u: float | None = None) -> int:
    """The one source of the softmax rule with ``d <= 0`` (module docstring): the compatible source with the largest score
    when ``u`` is None (``best``), else the one that a uniform number ``u`` in [0, 1) picks from ``softmax(s)`` over the
    compatible sources (``sample``, inverse of the cumulative distribution)."""
    s = np.asarray(scores, dtype=float)
    if u is None:
        return max(compatible, key=lambda j: s[j])
    z = s[compatible]
    p = np.exp(z - z.max())
    cumulative = np.cumsum(p / p.sum())
    return compatible[min(int(np.searchsorted(cumulative, float(u), side="right")), len(compatible) - 1)]


def softmax_weights(scores, d: float, compatible: list[int], u: float | None = None) -> np.ndarray:
    """The softmax rule of the module docstring: ``softmax(s / d)`` over ``compatible`` when ``d > 0``, else one source
    (``single_source``; ``u`` None: the best one, else drawn); weights below ``MIN_WEIGHT`` dropped."""
    s = np.asarray(scores, dtype=float)
    w = np.zeros(len(s))
    if not compatible:
        raise ValueError("no compatible source")
    if d <= 0:
        w[single_source(s, compatible, u)] = 1.0
        return w
    z = s[compatible] / float(d)
    e = np.exp(z - z.max())
    w[compatible] = e / e.sum()
    return _drop_small(w)


class RandmixDraws:
    """The random weights of ``mixing.rule: randmix``: one draw ``(c, gamma variates (J,), fallback index)`` per
    subtask of an attempt, kept until the next ``begin_attempt`` (CP-Gen may solve a constraint more than once)."""

    def __init__(self, chunk_seed: int):
        self._rng = np.random.default_rng(np.random.SeedSequence([int(chunk_seed), RANDMIX_SEED]))
        self._drawn: dict[int, tuple[float, np.ndarray, int]] = {}

    def begin_attempt(self) -> None:
        self._drawn = {}

    def draw(self, subtask: int, num_sources: int) -> tuple[float, np.ndarray, int]:
        if int(subtask) not in self._drawn:
            c = float(np.exp(self._rng.uniform(*RANDMIX_LOG_C)))
            gamma = self._rng.gamma(c, size=int(num_sources))
            fallback = int(self._rng.integers(int(num_sources)))
            self._drawn[int(subtask)] = (c, gamma, fallback)
        return self._drawn[int(subtask)]


class SourceDraws:
    """The uniform numbers of ``mixing.single_source: sample``: one per subtask of an attempt, kept until the next
    ``begin_attempt`` (CP-Gen may solve a constraint more than once, and keeps the drawn source)."""

    def __init__(self, chunk_seed: int):
        self._rng = np.random.default_rng(np.random.SeedSequence([int(chunk_seed), SOURCE_SEED]))
        self._drawn: dict[int, float] = {}

    def begin_attempt(self) -> None:
        self._drawn = {}

    def draw(self, subtask: int) -> float:
        if int(subtask) not in self._drawn:
            self._drawn[int(subtask)] = float(self._rng.random())
        return self._drawn[int(subtask)]


def randmix_weights(draw: tuple[float, np.ndarray, int], compatible: list[int]) -> np.ndarray:
    """Dirichlet(c, ..., c) weights over ``compatible`` from one ``RandmixDraws.draw`` (normalized gamma variates)."""
    _c, gamma, fallback = draw
    w = np.zeros(len(gamma))
    gc = np.asarray(gamma, dtype=float)[compatible]
    if gc.sum() <= 0:
        w[compatible[fallback % len(compatible)]] = 1.0
        return w
    w[compatible] = gc / gc.sum()
    return _drop_small(w)


def compatible_sources(series: list, primary: int) -> list[int]:
    """The sources whose gripper events match the selected source's (the selected source always)."""
    return [j for j in range(len(series)) if j == primary or _matching_events(series[primary], series[j]) is not None]


# blending ---------------------------------------------------------------------------------------------------------------
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


def symmetry_label(kind: str, params: Mapping, k: int) -> str:
    """Record label of the ``k``-th annotated symmetry of ``kind`` (obj, obj0, obj1, eef), e.g. ``"obj:z180"``."""
    if "rot_axis" in params and "rot_param" in params:
        return f"{kind}:{params['rot_axis']}{int(round(np.degrees(float(params['rot_param']))))}"
    return f"{kind}:{k}"


def gripper_flip() -> np.ndarray:
    """The parallel gripper's own symmetry: 180 degrees about its approach axis (the end-effector z axis)."""
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("z", np.pi).as_matrix()
    return out


def symmetry_candidates(object_symmetries, parallel_gripper: bool) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """``(label, S_obj, S_eef)`` combinations (module docstring, step 1), identity first."""
    objs = [("", np.eye(4))] + [(symmetry_label("obj", p, k), symmetry_matrix(p)) for k, p in enumerate(object_symmetries or [])]
    eefs = [("", np.eye(4))]
    if parallel_gripper:
        eefs.append(("eef:z180", gripper_flip()))
    return [("+".join(x for x in (lo, le) if x) or "identity", so, se) for le, se in eefs for lo, so in objs]


def key_step(series) -> int:
    """The first gripper event of a segment, else its last step (module docstring, step 1)."""
    change = events(np.asarray(series, dtype=float).reshape(-1))[1]
    return int(change[0]) if len(change) else len(series) - 1


def pose_distance(a, b) -> tuple[float, float]:
    """``(position distance m, rotation angle rad)`` between two 4x4 poses."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())


def symmetry_score(position_m: float, rotation_rad: float) -> float:
    """How far a candidate pose is from the reference (smaller is closer): CP-Gen ``find_best_transform``'s score."""
    return W_TRANS * position_m + W_ROT * rotation_rad


def align_to_reference(poses, eef, key: int, reference, candidates) -> tuple[np.ndarray, np.ndarray, dict]:
    """Re-express one source's object-frame ``poses`` (T,4,4) and start pose ``eef`` (4x4) with the candidate closest to
    ``reference`` at step ``key``: ``(poses, eef, record)``."""
    poses = np.asarray(poses, dtype=float)
    best = None
    for label, so, se in candidates:
        dp, dr = pose_distance(invert(so) @ poses[key] @ se, reference)
        score = symmetry_score(dp, dr)
        if best is None or score < best[0]:
            best = (score, label, so, se, dp, dr)
    _score, label, so, se, dp, dr = best
    dp0, dr0 = pose_distance(poses[key], reference)
    record = {"symmetry": label, "pos_cm": round(100 * dp, 2), "rot_deg": round(float(np.degrees(dr)), 1),
              "identity_pos_cm": round(100 * dp0, 2), "identity_rot_deg": round(float(np.degrees(dr0)), 1)}
    if label == "identity":
        return poses, np.asarray(eef, dtype=float), record
    return transform_all(invert(so), poses) @ se, invert(so) @ np.asarray(eef, dtype=float) @ se, record


class SkillMixer:
    """The ``SegmentMixer`` of the module docstring. MimicGen and the Isaac Lab generators call ``mix``; CP-Gen, which blends
    solved poses itself, calls ``decide`` (``tmg.generation.generators.cpgen.mixing``). ``rule`` and ``coefficients`` are the
    ``mixing.rule`` and ``mixing.coefficients`` of the method config.

    ``gripper_events`` is the open/closed rule of the gripper commands (``{type: sign}`` for one scalar command per step,
    the default; ``{type: closure, columns: [...]}`` for many-joint hands such as the GR1's). Events found by this rule
    align the segments; with the closure rule the mixed hand commands are the time-warped blend of the sources' joint
    commands (same phase at every step).
    """

    def __init__(self, *, rule: str = "softmax", coefficients: SkillMixRule | None = None, chunk_seed: int = 0,
                 gripper_events: Mapping | None = None, symmetries: Mapping | None = None, single_source: str = "best"):
        if rule not in RULES:
            raise ValueError(f"mixing.rule must be one of {RULES}, got {rule!r}")
        if rule == "softmax" and coefficients is None:
            raise ValueError("mixing.rule=softmax needs the task's rule file (mixing.coefficients)")
        if single_source not in SINGLE_SOURCE:
            raise ValueError(f"mixing.single_source must be one of {SINGLE_SOURCE}, got {single_source!r}")
        self.rule = rule
        self.coefficients = coefficients                   # the fitted rule file; None for the randmix rule
        self.single_source = single_source
        self.draws = RandmixDraws(chunk_seed)
        self.sources = SourceDraws(chunk_seed)
        self.detector = build_detector(gripper_events)
        self.chunk_seed = int(chunk_seed)
        self.calls = 0
        self.symmetries = {str(k): [dict(p) for p in v] for k, v in (symmetries or {}).items()}   # object -> transforms

    @classmethod
    def from_config(cls, cfg: Mapping, *, chunk_seed: int, method_name: str) -> SkillMixer:
        """``mixing.coefficients``: ``<generator>/<task>`` or a path to a rule file; ``mixing.rule``: softmax | randmix.
        The randmix rule needs no rule file (it draws the weights). Random weights are randmix data, not SkillMix,
        so that rule runs under the method name ``RANDMIX_METHOD`` and no other rule does: the name is the pool,
        dataset and run folder and the key of a generator's ``method_overrides``."""
        rule = str(cfg.get("rule", "softmax"))
        if (rule == "randmix") != (method_name == RANDMIX_METHOD):
            raise ValueError(f"method {method_name!r} with mixing.rule={rule}: the randmix rule and the method name "
                             f"{RANDMIX_METHOD} go together (method.mixing.rule=randmix method.name={RANDMIX_METHOD})")
        coefficients = None if rule == "randmix" else load_rule(str(cfg["coefficients"]))
        return cls(rule=rule, coefficients=coefficients, chunk_seed=chunk_seed, gripper_events=cfg.get("gripper_events"),
                   symmetries=cfg.get("symmetries"), single_source=str(cfg.get("single_source", "best")))

    def describe(self) -> dict:
        return {"type": "skillmix", "rule": self.rule, "single_source": self.single_source,
                "coefficients": None if self.coefficients is None else self.coefficients.describe(),
                "gripper_events": self.detector.name, "symmetries": self.symmetries}

    def begin_attempt(self) -> None:
        """Called by the runtime when an attempt starts (forgets the draws of the previous attempt)."""
        self.draws.begin_attempt()
        self.sources.begin_attempt()

    def weights(self, subtask: int, scene, inhand, compatible: list[int]) -> tuple[np.ndarray, dict]:
        """The weights of one subtask under the configured rule, and what the record keeps about them."""
        if self.rule == "randmix":
            draw = self.draws.draw(subtask, len(scene))
            return randmix_weights(draw, compatible), {"dirichlet_c": round(draw[0], 5)}
        block = self.coefficients.for_subtask(subtask)
        s = block.scores(scene, inhand)
        about = {"scores": [round(float(v), 4) for v in s], "d": round(block.d, 4)}
        u = None
        if block.d <= 0 and self.single_source == "sample":
            u = self.sources.draw(subtask)
            about["source_draw"] = round(u, 6)
        return softmax_weights(s, block.d, compatible, u), about

    def decide(self, subtask: int, primary: int, scene, inhand, series: list) -> tuple[np.ndarray, dict]:
        """The decision of one subtask, the same for every generator: ``(weights (J,), record)`` from the sources'
        features (``scene`` (J,3), ``inhand`` (J,2) or None) and gripper event series (``series``, one per source; only
        sources whose events match the selected source ``primary`` can be blended with it)."""
        compatible = compatible_sources(series, primary)
        w, about = self.weights(subtask, scene, inhand, compatible)
        record = {"rule": self.rule, **about, "weights": [round(float(v), 4) for v in w],
                  "incompatible": [j for j in range(len(series)) if j not in compatible],
                  "scene_features": np.round(scene, 4).tolist()}
        if inhand is not None:
            record["inhand_features"] = np.round(inhand, 4).tolist()
        return w, record

    # events ------------------------------------------------------------------------------------------
    def _scalar_commands(self, commands: np.ndarray) -> bool:
        """True when the commands are one scalar sign command per step, the official robosuite convention, so the raw
        column itself is the event series and the blended scalar is the mixed command."""
        return isinstance(self.detector, SignGripper) and commands.reshape(len(commands), -1).shape[1] == 1

    def _event_series(self, commands) -> np.ndarray:
        """The commands as the scalar series the event functions read: the raw first column for the sign rule (the
        official robosuite convention, zeros carry the state), otherwise +1 closed / -1 open from the detector."""
        g = np.asarray(commands, dtype=float).reshape(len(commands), -1)
        if self._scalar_commands(g):
            return g[:, 0]
        return np.where(self.detector.states(g), 1.0, -1.0)

    def features(self, context: MixContext) -> tuple[np.ndarray, np.ndarray | None]:
        """``(scene features (J,3), in-hand features (J,2) or None)`` of every source for the context's subtask."""
        k = int(context.subtask_ind)
        current = context.extra["current_object_pose"]
        scene = np.array([scene_features(current, context.src_object_poses[j]) for j in range(context.num_sources)])
        inhand = None
        if k >= 1 and "current_in_hand" in context.extra:
            now = context.extra["current_in_hand"]
            inhand = np.array([inhand_features(now, src) for src in context.extra["source_in_hand"]])
        return scene, inhand

    def mix(self, context: MixContext) -> MixedSegment:
        primary = int(context.primary)
        record = {"call": self.calls, "subtask": int(context.subtask_ind), "source_a": primary, "mixed": False, "rule": self.rule}
        self.calls += 1
        identity = MixedSegment(context.primary_eef_start, context.primary_targets, context.primary_gripper, record)
        if not context.has_reference_object or context.extra.get("current_object_pose") is None:
            record["identity_reason"] = "subtask has no reference object" if not context.has_reference_object else "no current object pose"
            return identity
        n = context.num_sources
        scene, inhand = self.features(context)
        series = [self._event_series(np.asarray(context.src_segments[j][2], dtype=float)) for j in range(n)]
        w, decided = self.decide(int(context.subtask_ind), primary, scene, inhand, series)
        record.update(decided)
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
        """Step 1 of the module docstring, in place on ``poses`` and ``eef_obj`` (object frames of the ``keep`` sources)."""
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
