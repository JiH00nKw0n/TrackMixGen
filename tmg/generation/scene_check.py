"""Scene placement check: do the configured ranges of a (task, scene) produce usable initial states?

The MimicGen environments draw object placements with robosuite's ``UniformRandomSampler`` with
``ensure_valid_placement=True`` (no overlap between sampled objects, by horizontal radius) and
``ensure_object_boundary_in_range=False`` (the range bounds the object centre, not its footprint). Widening the
ranges therefore cannot be checked from the config alone: a large object such as a drawer may hang over the
table edge, be drawn in contact with the robot, or leave no room for the other objects so that the sampler
gives up (``RandomizationError``). ``check_scene`` resets the real environment ``resets`` times and measures
exactly those things:

- placement failures: resets that raise ``RandomizationError`` (sampler or reach constraint);
- table edge: per object, the fraction of resets whose centre lies outside the table top and whose footprint
  (centre plus horizontal radius) crosses the edge;
- contacts at reset: penetrating contacts between two task objects, or a task object and the robot;
- settling: after ``settle_steps`` zero-action steps, objects whose centre ended below the table top (fell off),
  objects that dropped more than ``fall_threshold_m`` (information: the sampler spawns objects slightly above the
  surface) and objects that drifted in xy;
- listed ranges: the scene file's ``bounds`` compared with what the environment class actually samples from
  (``bounds_match_class``), which is how ``official_d1``'s listed D1 values are kept honest;
- reach: distance of every object to the robot base, the reject count of a reach-constrained scene, and the
  share of resets a 0.85 m radius would reject when the scene has no constraint.

Runs inside the simulator image (``make check-scene``). Output: ``<output>/SCENE_CHECK.json`` and a printed
summary. The stage never changes the environment classes it inspects.
"""
from __future__ import annotations

import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tmg.config import dump_yaml, resolved
from tmg.core.chunks import RESOLVED_CONFIG, write_json
from tmg.core.provenance import write_provenance
from tmg.generation.simulators.robosuite import (
    INTERFACE_TYPE,
    SCENE_META_ATTR,
    TABLE_OFFSET,
    effective_bounds,
    effective_reach_constraint,
    interface_name,
    load_scene_chain,
    register_env_class,
    robot_base_xy,
)

REPORT = "SCENE_CHECK.json"
DEFAULT_REACH_M = 0.85
MAX_CONTACT_DETAILS = 50            # contact details kept in the report (the pair counts cover every reset)
MAX_FAILURES = 20                   # placement failures kept in the report
PER_OBJECT_COUNTS = ("centre_outside", "footprint_over_edge", "beyond_reach", "dropped", "off_table", "drifted")


def make_env(env_name: str, robot: str):
    """Create the registered environment headless with an OSC pose controller (no rendering, no cameras)."""
    import robosuite
    from robosuite.controllers import load_controller_config

    return robosuite.make(env_name, robots=robot, controller_configs=load_controller_config(default_controller="OSC_POSE"),
                          has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=True,
                          ignore_done=True, control_freq=20, reward_shaping=False)


def object_geometry(env, object_bodies: dict[str, str]) -> dict:
    """Per task object: root body id, geom ids of its body subtree, and its horizontal radius.

    Args:
        env: the robosuite environment after reset.
        object_bodies: ``{object name: root body name}`` from the MimicGen interface.

    Returns:
        ``{object name: {"body_id", "geoms", "radius"}}``; radius comes from the matching ``MujocoObject`` when
        the model lists one, else from the largest geom bounding sphere of the subtree.
    """
    model = env.sim.model
    root_of_body = {}
    for b in range(model.nbody):
        node = b
        while model.body_parentid[node] != 0 and node != 0:
            node = model.body_parentid[node]
        root_of_body[b] = node
    by_root: dict[int, list[int]] = defaultdict(list)
    for g in range(model.ngeom):
        by_root[root_of_body[model.geom_bodyid[g]]].append(g)
    radius_by_body = {}
    for obj in getattr(env.model, "mujoco_objects", []) or []:
        radius = getattr(obj, "horizontal_radius", None)
        if radius is not None:
            radius_by_body[obj.root_body] = float(radius)
    out = {}
    for name, body in object_bodies.items():
        bid = model.body_name2id(body)
        geoms = by_root.get(root_of_body[bid], [])
        radius = radius_by_body.get(body)
        if radius is None:
            radius = float(max((model.geom_rbound[g] for g in geoms), default=0.0))
        out[name] = {"body_id": bid, "body": body, "geoms": geoms, "radius": radius}
    return out


def penetrating_contacts(env, geometry: dict, robot_root: int) -> list[tuple[str, str, str]]:
    """Triples (object, other, robot geom name or "") in penetration after ``sim.forward()``; ``other`` is an
    object name or ``robot``."""
    model, data = env.sim.model, env.sim.data
    owner = {}
    for name, geo in geometry.items():
        for g in geo["geoms"]:
            owner[g] = name
    robot_geoms = set()
    for g in range(model.ngeom):
        node = model.geom_bodyid[g]
        while node != 0:
            if node == robot_root:
                robot_geoms.add(g)
                break
            node = model.body_parentid[node]
    pairs = set()
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist >= -1e-4:
            continue
        a, b = owner.get(c.geom1), owner.get(c.geom2)
        if a and b and a != b:
            pairs.add((*sorted((a, b)), ""))
        elif a and c.geom2 in robot_geoms:
            pairs.add((a, "robot", model.geom_id2name(c.geom2) or str(c.geom2)))
        elif b and c.geom1 in robot_geoms:
            pairs.add((b, "robot", model.geom_id2name(c.geom1) or str(c.geom1)))
    return sorted(pairs)


def _r3(value) -> float:
    """``round(float(value), 3)``: the precision of the positions and distances in the report."""
    return round(float(value), 3)


def _table_facts(env) -> dict:
    """Centre, top height and half extents of the table top (``table_offset`` and ``table_full_size`` of the env)."""
    full = np.asarray(env.table_full_size, dtype=float)
    off = np.asarray(env.table_offset, dtype=float)
    return {"centre_xy": [float(off[0]), float(off[1])], "top_z": float(off[2]),
            "half_x": float(full[0] / 2), "half_y": float(full[1] / 2)}


@dataclass
class _Tally:
    """Everything ``check_scene`` accumulates over the resets."""
    counts: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))   # object -> count name -> n
    dists: dict = field(default_factory=lambda: defaultdict(list))                      # object -> base distances (m)
    xy_min: dict = field(default_factory=dict)                                          # object -> smallest x, y seen
    xy_max: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)                                        # {reset, error}
    contact_pairs: dict = field(default_factory=lambda: defaultdict(int))               # "a & b" -> resets with contact
    contact_details: list = field(default_factory=list)                                 # first MAX_CONTACT_DETAILS hits
    rejects: list = field(default_factory=list)                                         # reach-constraint rejects per reset
    times: list = field(default_factory=list)                                           # reset wall times (s)
    done: int = 0                                                                       # resets that succeeded

    def add_contacts(self, reset: int, hits: list, before: dict) -> None:
        """Count the penetrating contacts of one reset and keep the first few as details."""
        for obj, other, geom in hits:
            self.contact_pairs[f"{obj} & {other}"] += 1
            if len(self.contact_details) < MAX_CONTACT_DETAILS:
                self.contact_details.append({"reset": reset, "object": obj, "other": other, "robot_geom": geom,
                                             "object_xy": [_r3(v) for v in before[obj][:2]],
                                             "other_xy": [_r3(v) for v in before[other][:2]] if other in before else None})

    def add_placement(self, name: str, p: np.ndarray, radius: float, table: dict, base_xy, margin: float, reach_m: float) -> None:
        """Classify one object's reset position against the table edge and the reach radius."""
        c = self.counts[name]
        cx, cy = table["centre_xy"]
        dx, dy = abs(p[0] - cx), abs(p[1] - cy)
        c["n"] += 1
        if dx > table["half_x"] or dy > table["half_y"]:
            c["centre_outside"] += 1
        if dx + radius > table["half_x"] - margin or dy + radius > table["half_y"] - margin:
            c["footprint_over_edge"] += 1
        d = float(np.hypot(p[0] - base_xy[0], p[1] - base_xy[1]))
        self.dists[name].append(d)
        if d > reach_m:
            c["beyond_reach"] += 1
        self.xy_min[name] = np.minimum(self.xy_min.get(name, p[:2]), p[:2])
        self.xy_max[name] = np.maximum(self.xy_max.get(name, p[:2]), p[:2])

    def add_settling(self, name: str, before: np.ndarray, after: np.ndarray, table: dict, fall_thr: float,
                     drift_thr: float) -> None:
        """Compare one object's position after the settle steps with its reset position."""
        c = self.counts[name]
        if before[2] - after[2] > fall_thr:
            c["dropped"] += 1
        if after[2] < table["top_z"]:
            c["off_table"] += 1
        if np.hypot(*(after[:2] - before[:2])) > drift_thr:
            c["drifted"] += 1

    def objects(self, geometry: dict) -> dict:
        """The per-object block of the report: radius, position range, base distance and the count fractions."""
        out = {}
        for k, c in self.counts.items():
            m = max(c["n"], 1)
            out[k] = {"radius_m": round(geometry[k]["radius"], 4),
                      "xy_min": [_r3(v) for v in self.xy_min[k]], "xy_max": [_r3(v) for v in self.xy_max[k]],
                      "base_dist_m": {"min": _r3(min(self.dists[k])), "mean": _r3(np.mean(self.dists[k])),
                                      "max": _r3(max(self.dists[k]))},
                      **{key: {"count": c[key], "fraction": round(c[key] / m, 4)} for key in PER_OBJECT_COUNTS}}
        return out


def _build_report(tally: _Tally, *, env_name: str, task: dict, scene: dict, interface: str, resets: int, table: dict | None,
                  base_xy, bounds_match: dict | None, constraint: dict | None, reach_m: float, geometry: dict | None) -> dict:
    """Assemble the report of the module docstring from the tally and the facts read once, with the ``ok`` verdict."""
    objects = tally.objects(geometry) if tally.counts else {}
    any_contact = sum(tally.contact_pairs.values())
    rejects = tally.rejects
    report = {
        "env": env_name, "task": task["name"], "scene": scene["name"], "interface": interface,
        "resets_requested": resets, "resets_done": tally.done, "placement_failures": len(tally.failures),
        "table": table, "robot_base_xy": list(base_xy) if base_xy else None, "bounds_match_class": bounds_match,
        "reach": {"constraint": constraint, "radius_used_m": reach_m,
                  "rejects": {"mean": round(float(np.mean(rejects)), 2), "max": int(max(rejects))} if rejects else None,
                  "resets_with_any_object_beyond": _resets_beyond(tally.dists, reach_m)},
        "contacts_at_reset": {"resets_with_penetration": any_contact, "pairs": dict(tally.contact_pairs),
                              "details": tally.contact_details},
        "objects": objects, "failures": tally.failures[:MAX_FAILURES],
        "reset_time_s": {"mean": _r3(np.mean(tally.times)) if tally.times else None},
    }
    report["ok"] = (not tally.failures and any_contact == 0 and (bounds_match is None or bounds_match["match"])
                    and all(o["off_table"]["count"] == 0 and o["drifted"]["count"] == 0 for o in objects.values()))
    return report


def check_scene(cfg) -> dict:
    """Run the check described in the module docstring and write ``SCENE_CHECK.json``.

    Args:
        cfg: the composed ``check_scene`` config.

    Returns:
        The report: environment and table facts, per-object fractions, failure list and an ``ok`` verdict
        (no placement failure, no penetrating contact, no object off the table after settling, no drift). The
        footprint-over-edge and drop fractions are reported as information only.
    """
    plain = resolved(cfg)
    task, scene = plain["task"], load_scene_chain(plain["scene"], plain["generator"]["name"])
    output = Path(plain["output"])
    output.mkdir(parents=True, exist_ok=True)
    dump_yaml(plain, output / RESOLVED_CONFIG)
    write_provenance(output, extra={"command": "check_scene", "output": str(output)})

    env_name = register_env_class(task, scene)
    interface = interface_name(task)
    constraint = effective_reach_constraint(scene)
    reach_m = float(constraint["radius_m"]) if constraint else DEFAULT_REACH_M

    env = make_env(env_name, str(plain["robot"]))
    np.random.seed(int(plain["seed"]))
    from mimicgen.env_interfaces.base import REGISTERED_ENV_INTERFACES
    from robosuite.utils import RandomizationError

    iface = REGISTERED_ENV_INTERFACES[INTERFACE_TYPE][interface](env)
    n = int(plain["resets"])
    settle = int(plain["settle_steps"])
    margin = float(plain["edge_margin_m"])
    fall_thr, drift_thr = float(plain["fall_threshold_m"]), float(plain["drift_threshold_m"])

    tally = _Tally()
    table = None             # read once, after the first successful reset
    base_xy = None
    bounds_match: dict | None = None
    geometry = None
    for i in range(n):
        t0 = time.time()
        try:
            env.reset()
        except RandomizationError as exc:
            tally.failures.append({"reset": i, "error": str(exc)})
            continue
        except Exception:  # noqa: BLE001  report, do not hide
            tally.failures.append({"reset": i, "error": traceback.format_exc()})
            continue
        tally.times.append(time.time() - t0)
        tally.done += 1
        if table is None:
            bounds_match = compare_bounds(env, effective_bounds(scene, str(task["name"])))
            table = _table_facts(env)
            base_xy = robot_base_xy(env)
        poses = iface.get_object_poses()
        if geometry is None:
            bodies = {k: _root_body_of(env, k, iface) for k in poses}
            geometry = object_geometry(env, bodies)
        meta = getattr(env, SCENE_META_ATTR, None)
        if meta is not None:
            tally.rejects.append(int(meta.get("n_reject", 0)))
        env.sim.forward()
        before = {k: np.asarray(p)[:3, 3].copy() for k, p in poses.items()}
        hits = penetrating_contacts(env, geometry, env.sim.model.body_name2id(env.robots[0].robot_model.root_body))
        tally.add_contacts(i, hits, before)
        for k, p in before.items():
            tally.add_placement(k, p, geometry[k]["radius"], table, base_xy, margin, reach_m)
        if settle > 0:
            zero = np.zeros(env.action_dim)
            for _ in range(settle):
                env.step(zero)
            after = {k: np.asarray(p)[:3, 3] for k, p in iface.get_object_poses().items()}
            for k in before:
                tally.add_settling(k, before[k], after[k], table, fall_thr, drift_thr)
    env.close()

    report = _build_report(tally, env_name=env_name, task=task, scene=scene, interface=interface, resets=n, table=table,
                           base_xy=base_xy, bounds_match=bounds_match, constraint=constraint, reach_m=reach_m,
                           geometry=geometry)
    write_json(output / REPORT, report)
    _print_summary(report)
    return report


def compare_bounds(env, listed: dict | None) -> dict | None:
    """Compare the scene file's listed ranges with ``env._get_initial_placement_bounds()``.

    Returns:
        ``{"match": bool, "differences": [...]}`` or None when the scene lists no ranges for the task.
    """
    if listed is None:
        return None
    actual = env._get_initial_placement_bounds()
    diffs = []
    for name, e in listed.items():
        a = actual.get(name)
        if a is None:
            diffs.append(f"{name}: listed but not sampled by the class")
            continue
        for key in ("x", "y", "z_rot"):
            if not np.allclose(np.asarray(a[key], dtype=float), np.asarray(e[key], dtype=float), atol=1e-6):
                diffs.append(f"{name}.{key}: class {tuple(float(v) for v in a[key])} vs listed {e[key]}")
        ref = np.asarray(env.table_offset if e["reference"] == TABLE_OFFSET else e["reference"], dtype=float)
        if not np.allclose(np.asarray(a["reference"], dtype=float), ref, atol=1e-6):
            diffs.append(f"{name}.reference: class {np.asarray(a['reference']).tolist()} vs listed {ref.tolist()}")
        if a.get("rotation_axis", "z") != e.get("rotation_axis", "z"):
            diffs.append(f"{name}.rotation_axis: class {a.get('rotation_axis', 'z')} vs listed {e.get('rotation_axis', 'z')}")
    for name in actual:
        if name not in listed:
            diffs.append(f"{name}: sampled by the class but not listed")
    return {"match": not diffs, "differences": diffs}


def _resets_beyond(dists: dict[str, list[float]], radius: float) -> int:
    """Number of resets in which at least one object lies farther than ``radius`` from the base."""
    if not dists:
        return 0
    n = min(len(v) for v in dists.values())
    return int(sum(any(dists[k][i] > radius for k in dists) for i in range(n)))


def _root_body_of(env, name: str, iface) -> str:
    """Root body name of interface object ``name`` by matching its reported pose against the model bodies."""
    pose = np.asarray(iface.get_object_poses()[name])[:3, 3]
    model, data = env.sim.model, env.sim.data
    best, best_d = None, 1e9
    for b in range(1, model.nbody):
        if model.body_parentid[b] != 0:
            continue
        d = float(np.linalg.norm(data.body_xpos[b] - pose))
        if d < best_d:
            best, best_d = b, d
    if best is None or best_d > 1e-6:
        raise RuntimeError(f"cannot match object {name!r} to a root body (nearest {best_d:.4f} m)")
    return model.body_id2name(best)


def _print_summary(r: dict) -> None:
    """Print the report as a short table."""
    bm = r.get("bounds_match_class")
    if bm is not None:
        print(f"[tmg] listed ranges match the environment class: {bm['match']}" + ("; " + "; ".join(bm["differences"]) if bm["differences"] else ""))
    print(f"[tmg] scene check {r['env']}: {r['resets_done']}/{r['resets_requested']} resets, "
          f"{r['placement_failures']} placement failures, {r['contacts_at_reset']['resets_with_penetration']} resets with "
          f"penetrating contacts, ok={r['ok']}")
    t = r["table"]
    if t:
        print(f"[tmg] table centre {t['centre_xy']} half {t['half_x']}x{t['half_y']} m, robot base xy {r['robot_base_xy']}, "
              f"reach radius {r['reach']['radius_used_m']} m, resets with an object beyond it: {r['reach']['resets_with_any_object_beyond']}"
              + (f", rejects mean {r['reach']['rejects']['mean']} max {r['reach']['rejects']['max']}" if r['reach']['rejects'] else ""))
    print(f"[tmg] {'object':<16}{'radius':>8}{'centre_out':>12}{'over_edge':>11}{'beyond':>8}{'dropped':>9}{'off_tbl':>9}{'drift':>7}  base_dist min/mean/max")
    for k, o in r["objects"].items():
        bd = o["base_dist_m"]
        print(f"[tmg] {k:<16}{o['radius_m']:>8.3f}{o['centre_outside']['fraction']:>12.3f}{o['footprint_over_edge']['fraction']:>11.3f}"
              f"{o['beyond_reach']['fraction']:>8.3f}{o['dropped']['fraction']:>9.2f}{o['off_table']['fraction']:>9.2f}"
              f"{o['drifted']['fraction']:>7.2f}  {bd['min']}/{bd['mean']}/{bd['max']}")
    for pair, c in r["contacts_at_reset"]["pairs"].items():
        print(f"[tmg] contact {pair}: {c} resets")
    for d in r["contacts_at_reset"]["details"][:12]:
        print(f"[tmg]   reset {d['reset']}: {d['object']} at xy {d['object_xy']} with {d['other']}"
              + (f" geom {d['robot_geom']}" if d["robot_geom"] else f" at xy {d['other_xy']}"))
    for f in r["failures"][:3]:
        print(f"[tmg] failure at reset {f['reset']}: {f['error'].strip().splitlines()[-1]}")
