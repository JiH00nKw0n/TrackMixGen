"""The evaluation loop shared by the robosuite evaluator (``rollout.py``) and the Isaac Lab evaluator
(``isaaclab_rollout.py``): connect to the policy server, check that it serves the requested checkpoint with the
inputs the observation config describes, then play every scene with a per-episode policy seed and hand each
episode row to the caller. Neither simulator is imported here; the per-episode work is a callable the evaluator
passes in (``play_scenes``), so this module is unit-tested with fakes.

Per-episode policy seed: ``SeedSequence([evaluation.seed, training_seed, reset_index])`` (``episode_seed``), sent
to the server's ``reset`` so a policy that samples (BC-RNN, Diffusion Policy) is reproducible per scene.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import TypedDict

import numpy as np

from tmg.core.features import ENV_STATE, STATE, image_feature
from tmg.data.images import to_uint8_frame
from tmg.evaluation.client import PolicyClient


class Scene(TypedDict, total=False):
    """One fixed evaluation scene (``resets.read_resets``): a stored robosuite state with its model xml, or, for the
    Isaac Lab generators, the seed the reset events draw the placement from."""

    index: int
    split: str
    states: np.ndarray
    model: str
    seed: int


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def episode_seed(eval_seed: int, training_seed: int, reset_index: int) -> int:
    """The policy seed of one episode: ``SeedSequence([evaluation.seed, training_seed, reset_index])``."""
    return int(np.random.SeedSequence([eval_seed, training_seed, reset_index]).generate_state(1)[0])


def select_scenes(scenes: list[Scene], *, limit=None, shard=None) -> list[Scene]:
    """The scenes one ``tmg eval`` process plays: the first ``limit`` (check runs), then shard ``k`` of ``n``
    (``shard=[k, n]``: the scenes whose reset index modulo n equals k). Shards of one checkpoint append to the
    same ``episodes.jsonl`` (one line per episode); a final run without ``shard`` finds every scene done and
    writes the summary."""
    if limit is not None:
        scenes = scenes[:int(limit)]
    if shard is not None:
        k, n = (int(v) for v in shard)
        if not 0 <= k < n:
            raise ValueError(f"shard must be [k, n] with 0 <= k < n, got {shard}")
        scenes = [s for s in scenes if int(s["index"]) % n == k]
    return scenes


def check_server(info: dict, checkpoint: str) -> None:
    """The server must serve exactly the requested checkpoint. A server that failed to bind its port (another
    server still holds it) leaves the client talking to that other server; without this check the episodes
    would be recorded under the wrong checkpoint, as happened on 2026-10-27 when validation ports were reused."""
    served = str(info.get("checkpoint", "")).rstrip("/")
    if served != str(checkpoint).rstrip("/"):
        raise RuntimeError(f"the policy server serves {served!r}, not the requested {checkpoint!r}; "
                           "another server probably holds the port")


def check_policy_matches(info: dict, obs_cfg: dict) -> None:
    """The served policy must take the cameras of the observation config and the object state exactly when the
    config has object state keys (a low-dim checkpoint cannot be evaluated as an image one, nor the reverse)."""
    if info["cameras"] != list(obs_cfg.get("cameras") or []):
        raise RuntimeError(f"policy cameras {info['cameras']} differ from observation={obs_cfg['name']} "
                           f"({obs_cfg.get('cameras')})")
    if bool(info.get("env_state_dim", 0)) != bool(obs_cfg.get("object_state")):
        raise RuntimeError(f"policy env_state_dim {info.get('env_state_dim', 0)} does not match observation={obs_cfg['name']}; "
                           "evaluate with the observation the checkpoint was trained with")


def policy_input(obs: dict, observation: dict, *, image_key: str = "{cam}_image") -> dict:
    """Policy input from an observation dict (images -> CHW uint8; state keys concatenated).

    Images arrive in one of two forms: raw frames, (H, W, 3) uint8 (robosuite ``make_env``, Isaac Lab observation
    terms), or robomimic-processed frames, (3, H, W) float32 in [0, 1] (``make_env_from_meta``: robomimic's
    ``EnvRobosuite.get_observation`` runs ``ObsUtils.process_obs``, which scales and transposes). Both become the
    (3, H, W) uint8 the datasets store. ``image_key`` is the observation key pattern of a camera: ``{cam}_image`` for
    the robosuite generators, ``{cam}`` for the Isaac Lab generators (same pattern as the dataset layout).
    """
    out = {}
    for cam in observation.get("cameras") or []:
        img = to_uint8_frame(obs[image_key.format(cam=cam)])
        if not (img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3):     # (H, W, 3) -> (3, H, W)
            img = img.transpose(2, 0, 1)
        out[image_feature(cam)] = np.ascontiguousarray(img)
    out[STATE] = np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in observation["robot_state"]])
    object_keys = list(observation.get("object_state") or [])
    if object_keys:
        out[ENV_STATE] = np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in object_keys])
    return out


def summarize(rows: list[dict]) -> dict:
    """Success rate with Wilson interval overall and per split."""
    def block(rs):
        n, k = len(rs), sum(int(r["success"]) for r in rs)
        lo, hi = wilson(k, n)
        return {"n": n, "n_success": k, "success_rate": (k / n if n else None), "ci_low": lo, "ci_high": hi,
                "mean_steps": (float(np.mean([r["steps"] for r in rs])) if rs else None),
                "errors": sum(1 for r in rs if r.get("error"))}
    out = {"all": block(rows), "by_split": {}}
    for split in sorted({r["split"] for r in rows}):
        out["by_split"][split] = block([r for r in rows if r["split"] == split])
    return out


def server_url(server: dict) -> str:
    """``ws://<host>:<port>`` of ``evaluation.server``."""
    return f"ws://{server['host']}:{server['port']}"


def play_scenes(todo: list[Scene], *, server: dict, checkpoint: str, obs_cfg: dict, eval_seed: int, training_seed: int,
                run_one: Callable[[PolicyClient, Scene, int], dict], record: Callable[[dict], None], client=None) -> None:
    """Connect to the policy server, check it, and play ``todo`` scene by scene.

    Args:
        todo: the scenes still to play, in order.
        server: ``evaluation.server`` (host, port).
        checkpoint: the checkpoint the server must serve (``check_server``).
        obs_cfg: the observation config the policy must match (``check_policy_matches``).
        eval_seed, training_seed: enter the per-episode policy seed (``episode_seed``).
        run_one: ``run_one(policy, scene, seed) -> {"success", "steps", "error", ...}`` plays one scene.
        record: receives each finished episode row: ``reset_index``, ``split``, the keys ``run_one`` returned,
            ``policy_seed``, ``checkpoint`` and ``seconds``.
        client: the client class (default ``PolicyClient``; tests pass a fake).
    """
    with (client or PolicyClient)(server_url(server)) as policy:
        info = policy.info()
        check_server(info, checkpoint)
        check_policy_matches(info, obs_cfg)
        for scene in todo:
            seed = episode_seed(int(eval_seed), int(training_seed), scene["index"])
            t0 = time.time()
            result = run_one(policy, scene, seed)
            record({"reset_index": scene["index"], "split": scene["split"], **result, "policy_seed": seed,
                    "checkpoint": checkpoint, "seconds": round(time.time() - t0, 2)})
