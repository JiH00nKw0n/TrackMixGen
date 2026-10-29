"""Per-demonstration constraint windows for CP-Gen's source demonstrations (decision 2026-10-29).

CP-Gen re-solves annotated constraints instead of replaying source segments. A constraint is the pair of bodies that
interact (hand and object, or object and object), their keypoints, symmetries, attachment and post-constraint behaviour,
and the ``timesteps`` of the source demonstration during which the interaction holds. The released annotation
(``constraints.json``, one file per task) carries the time windows of demonstration 0 only, and CP-Gen applies that file
to every demonstration it loads, so demonstrations 1 to 9 would get windows that do not match their motion. Everything
but the windows is a property of the task's objects and is the same for every demonstration.

This module derives each demonstration's windows with CP-Gen's own segmentation rule (``demo_aug.utils.
demo_segmentation_utils``: an interaction starts at the first step where the distance between its two bodies drops under a
threshold and ends by CP-Gen's contact rules) and writes them into the prepared source file as the per-demonstration
``constraint_data`` groups that CP-Gen's ``load_demos`` reads first. CP-Gen uses one threshold (3 cm) for everything; here
the threshold of every interaction is chosen on demonstration 0 so that its window matches the released window best (grid
``GRID``), and the same thresholds are applied to the other demonstrations. Demonstration 0 keeps the released
constraints verbatim. The derived windows were reviewed on video on 2026-10-29 (docs/GENERATORS.md 3.1); Threading's
second interaction (needle into the tripod) starts earlier than the released window in every demonstration because the
needle is near the tripod well before the insertion the annotator marked.

The simulator is reset the way CP-Gen's own ``demo_aug.generate.reset_to`` resets it (that function replaces robomimic's
``EnvRobosuite.reset_to`` when ``demo_aug.generate`` is imported, which this module does not do): the stored model xml of a
demonstration is tried first and, when the installed robosuite cannot load it, the freshly reset model is kept and the
states are restored into it. The released sources were recorded with robosuite 1.4.1 and their xml cannot be loaded by the
robosuite 1.5 fork of the CP-Gen image (renamed asset folders, missing robot sites), so every demonstration takes that
fall-back. Object placement lives in the state, but object *geometry* does not: MimicGen builds some objects with random
dimensions at every reset (the hammer of Hammer Cleanup), so a fresh model would put a differently sized object under the
recorded states and the body distances that decide the windows would change from reset to reset. :func:`reset_model`
therefore copies the stored xml's body and geom placement and sizes into the fresh model by name (robot, gripper and
mount bodies excluded), which makes the segmentation deterministic and the scene the recorded one. CP-Gen itself does not
do this when it loads the sources; it does not need to, because it only reads the states and the constraint windows.

Runs inside the CP-Gen image (robosuite 1.5, robomimic, CP-Gen's checkout on ``sys.path``); ``tmg sources`` calls
:func:`annotate_source` for the CP-Gen tasks. The pure helpers (:func:`windowed_constraints`, :func:`best_threshold`) have
unit tests; the segmentation needs the simulator.
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np

GRID = (0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08)   # metres; 0.001 = contact
GAP = 10          # CP-Gen: steps skipped after a completed interaction before the next one may start
GROUP = "constraint_data"


def interactions_of(constraints: Sequence[Mapping]) -> list[tuple[str, str]]:
    """The interaction pairs, in order, from the released constraints' names (``"entity1-entity2"``)."""
    return [tuple(str(c["name"]).split("-", 1)) for c in constraints]


def released_windows(constraints: Sequence[Mapping]) -> list[tuple[int, int]]:
    """``[start, end)`` of every released constraint (demonstration 0)."""
    return [(int(c["timesteps"][0]), int(c["timesteps"][-1]) + 1) for c in constraints]


def windowed_constraints(constraints: Sequence[Mapping], windows: Sequence[tuple[int, int]]) -> list[dict]:
    """The released constraints with ``timesteps`` replaced by ``windows`` (one ``[start, end)`` per constraint)."""
    if len(windows) != len(constraints):
        raise ValueError(f"{len(windows)} windows for {len(constraints)} constraints")
    out = []
    for c, (start, end) in zip(constraints, windows):
        if not 0 <= start < end:
            raise ValueError(f"bad window {(start, end)} for constraint {c.get('name')}")
        d = json.loads(json.dumps(c))
        d["timesteps"] = list(range(int(start), int(end)))
        out.append(d)
    return out


def best_threshold(candidates: Mapping[float, tuple[int, int] | None], released: tuple[int, int]) -> tuple[float, dict]:
    """The threshold whose window is nearest the released one (sum of start and end differences in steps; ties keep the
    smaller threshold). ``candidates`` maps threshold -> window or None (interaction not found)."""
    best = None
    for thr in sorted(candidates):
        w = candidates[thr]
        if w is None:
            continue
        err = abs(w[0] - released[0]) + abs(w[1] - released[1])
        if best is None or err < best[0]:
            best = (err, float(thr), (int(w[0]), int(w[1])))
    if best is None:
        return 0.03, {"threshold": 0.03, "found": False, "released": list(released)}
    return best[1], {"threshold": best[1], "found": True, "window": list(best[2]), "released": list(released), "error_steps": int(best[0])}


# --- simulator side (CP-Gen image) ------------------------------------------------------------------------------------
def _cpgen_modules(source_root: str):
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    cwd = os.getcwd()
    os.chdir(source_root)                     # CP-Gen opens demo_aug/configs/... relative to its checkout
    try:
        import cpgen_envs  # noqa: F401  registers CP-Gen's robosuite 1.5 environments (Coffee_D0, ...; robosuite.make needs them)
        from demo_aug.utils import demo_segmentation_utils as seg
    finally:
        os.chdir(cwd)
    return seg


def make_env(demo_path: str):
    """A robosuite environment of the dataset's env_args without rendering, with robosuite 1.5's composite controller."""
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    from robosuite.controllers import load_composite_controller_config

    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=["agentview_image"])))
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=demo_path)
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    env_meta["env_kwargs"]["use_camera_obs"] = False
    env_meta["env_kwargs"]["has_offscreen_renderer"] = False
    env_meta["env_kwargs"].pop("env_name", None)
    cc = load_composite_controller_config(robot="Panda")
    cc["body_parts"]["right"]["input_type"] = "absolute"
    cc["body_parts"]["right"]["input_ref_frame"] = "world"
    env_meta["env_kwargs"]["controller_configs"] = cc
    return EnvUtils.create_env_from_metadata(env_meta=env_meta, use_image_obs=False, render_offscreen=False, render=False)


SKIP_BODY_PREFIXES = ("robot", "gripper", "mount")   # robot-side bodies: the fresh robosuite 1.5 model is authoritative for them


def _floats(text: str | None, n: int, default: Sequence[float]) -> np.ndarray:
    out = np.array(default, dtype=float)
    if text:
        vals = [float(x) for x in text.split()]
        out[: len(vals)] = vals[:n]
    return out


def apply_stored_geometry(env, model_xml: str) -> int:
    """Copy body placement (pos, quat) and geom placement and size from the stored model xml into the current model of
    ``env`` for every body and geom of the same name, except robot-side bodies. Mesh geoms keep their mesh and size.
    Returns the number of bodies and geoms updated. Used when the stored xml itself cannot be loaded."""
    import xml.etree.ElementTree as ET

    m = env.env.sim.model
    n = 0
    for body in ET.fromstring(model_xml).iter("body"):
        name = body.get("name")
        if not name or name.startswith(SKIP_BODY_PREFIXES):
            continue
        try:
            bid = m.body_name2id(name)
        except Exception:      # noqa: BLE001  body only in the stored model
            continue
        m.body_pos[bid] = _floats(body.get("pos"), 3, (0.0, 0.0, 0.0))
        if not (body.get("euler") or body.get("axisangle")):
            m.body_quat[bid] = _floats(body.get("quat"), 4, (1.0, 0.0, 0.0, 0.0))
        n += 1
        for geom in body.findall("geom"):
            gname = geom.get("name")
            if not gname or geom.get("fromto"):
                continue
            try:
                gid = m.geom_name2id(gname)
            except Exception:      # noqa: BLE001
                continue
            m.geom_pos[gid] = _floats(geom.get("pos"), 3, (0.0, 0.0, 0.0))
            if not (geom.get("euler") or geom.get("axisangle")):
                m.geom_quat[gid] = _floats(geom.get("quat"), 4, (1.0, 0.0, 0.0, 0.0))
            if geom.get("type") != "mesh" and not geom.get("mesh") and geom.get("size"):
                m.geom_size[gid] = _floats(geom.get("size"), 3, (0.0, 0.0, 0.0))
            n += 1
    return n


def reset_model(env, model_xml: str) -> bool:
    """Reset ``env`` (a robomimic ``EnvRobosuite``) and try to load the demonstration's model xml, as CP-Gen's ``reset_to``
    does. Returns False when the installed robosuite cannot load the xml; the freshly reset model is then kept and the
    stored geometry is copied into it (:func:`apply_stored_geometry`)."""
    env.reset()
    try:
        from robosuite.utils.binding_utils import MjSim
        xml = env.env.edit_model_xml(model_xml)
        MjSim.from_xml_string(xml)
        env.env.reset_from_xml_string(xml)
        ok = True
    except Exception:      # noqa: BLE001  CP-Gen logs and continues with the reset model
        ok = False
        apply_stored_geometry(env, model_xml)
    env.env.sim.reset()
    return ok


def set_state(env, state) -> None:
    """Restore one flattened MuJoCo state without computing an observation."""
    env.env.sim.set_state_from_flattened(state)
    env.env.sim.forward()


def close_env(env) -> None:
    inner = getattr(env, "env", env)
    if hasattr(inner, "close"):
        inner.close()


def segment(seg, env, states, model_file: str, interactions, thresholds, upto: int | None = None) -> list[tuple[int, int]]:
    """CP-Gen's sequential segmentation with a threshold per interaction: ``[start, end)`` per interaction found."""
    def ended(k, thr):
        e1, e2 = interactions[k]
        nxt = interactions[k + 1] if k + 1 < len(interactions) else None
        if e1.startswith("gripper0_") and not e2.startswith("gripper0_"):
            return seg.is_robot_object_interaction_end(env, e1, e2, next_interaction=nxt, interaction_threshold=thr)
        if e2.startswith("gripper0_") and not e1.startswith("gripper0_"):
            return seg.is_robot_object_interaction_end(env, e2, e1, next_interaction=nxt, interaction_threshold=thr)
        return seg.is_obj_obj_interaction_end(env, e1, e2, next_interaction=nxt, interaction_threshold=thr)

    reset_model(env, model_file)
    windows, k, active, start, last_end = [], 0, False, None, -GAP
    n = len(interactions) if upto is None else upto
    for t, state in enumerate(states):
        if k >= n:
            break
        set_state(env, state)
        thr = float(thresholds[k])
        if not active:
            if t < last_end + GAP:
                continue
            e1, e2 = interactions[k]
            if seg.compute_body_distance(env, e1, e2, collision_distance=thr, mj_dist=True) < thr:
                active, start = True, t
        elif ended(k, thr):
            windows.append((start, t)); last_end = t; active = False; k += 1
    if active:
        windows.append((start, len(states)))
    return windows


def derive_windows(demo_path: str, constraints: Sequence[Mapping], *, source_root: str, grid=GRID) -> dict:
    """Thresholds calibrated on demonstration 0 and the windows of every demonstration (see the module docstring)."""
    seg = _cpgen_modules(source_root)
    interactions = interactions_of(constraints)
    released = released_windows(constraints)
    with h5py.File(demo_path, "r") as f:
        demos = [(int(k.split("_")[1]), np.array(f["data"][k]["states"]), str(f["data"][k].attrs["model_file"]))
                 for k in sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))]
    env = make_env(demo_path)
    try:
        _, states0, model0 = demos[0]
        thresholds, calibration = [], []
        for k in range(len(interactions)):
            cands = {}
            for thr in grid:
                w = segment(seg, env, states0, model0, interactions, thresholds + [thr], upto=k + 1)
                cands[thr] = w[k] if len(w) > k else None
            thr, info = best_threshold(cands, released[k])
            thresholds.append(thr); calibration.append(info)
        windows = {}
        for idx, states, model in demos:
            windows[idx] = [list(w) for w in segment(seg, env, states, model, interactions, thresholds)]
    finally:
        close_env(env)
    return {"interactions": interactions, "thresholds": thresholds, "calibration": calibration, "released_windows_demo0": released,
            "windows": windows}


def write_constraint_data(demo_path: str, constraints: Sequence[Mapping], windows: Mapping[int, Sequence]) -> dict:
    """Write every demonstration's constraints into ``data/demo_<i>/constraint_data/c<k>`` (JSON strings, the form CP-Gen's
    ``load_demos`` reads) and the interaction order as the ``interactions`` attribute. Demonstration 0 gets the released
    constraints verbatim. Returns ``{demo index: number of constraints}``; demonstrations whose window count differs from
    the constraint count are left without constraint data and reported with 0."""
    interactions = ",".join(f"{a}:{b}" for a, b in interactions_of(constraints))
    written = {}
    with h5py.File(demo_path, "a") as f:
        for key in f["data"].keys():
            idx = int(key.split("_")[1])
            g = f["data"][key]
            if GROUP in g:
                del g[GROUP]
            if idx == 0:
                items = [json.loads(json.dumps(c)) for c in constraints]
            else:
                w = [tuple(x) for x in windows.get(idx, [])]
                if len(w) != len(constraints):
                    written[idx] = 0
                    continue
                items = windowed_constraints(constraints, w)
            grp = g.create_group(GROUP)
            for k, item in enumerate(items):
                grp.create_dataset(f"c{k}", data=json.dumps(item))
            g.attrs["interactions"] = interactions
            written[idx] = len(items)
    return written


def has_constraint_data(demo_path: str) -> bool:
    """True when every demonstration of the file carries a ``constraint_data`` group."""
    with h5py.File(demo_path, "r") as f:
        return all(GROUP in f["data"][k] for k in f["data"].keys())


def annotate_source(demo_path: str, constraints_path: str, *, source_root: str, report_path: str | None = None) -> dict:
    """Derive and write the per-demonstration windows of one task's source file; returns the derivation report."""
    constraints = json.loads(Path(constraints_path).read_text())
    report = derive_windows(demo_path, constraints, source_root=source_root)
    report["written"] = write_constraint_data(demo_path, constraints, report["windows"])
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=1))
    return report
