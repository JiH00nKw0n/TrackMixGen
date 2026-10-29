"""Replay demonstrations of a robomimic/MimicGen file in the evaluation environment and compare with the record.

    python scripts/check_replay_consistency.py --src <file.hdf5> --env ThreePieceAssembly_D2 [--demos 20] [--steps 0]

For each demo: build the evaluation environment exactly as ``tmg eval`` does (``rollout.make_env``), restore the
demo's first state with ``reset_to`` + ``sync_controllers``, compare the first observation with the recorded one
(object state, end-effector position), then replay the recorded actions open-loop and report the end-effector
drift against the recorded trajectory and whether the task succeeds. A high replay success rate and small drift
mean the evaluation environment matches the environment the data came from; otherwise the evaluator, not the
policy, explains low success rates.
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

from tmg.core.layout import indexed_keys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--demos", type=int, default=20)
    parser.add_argument("--steps", type=int, default=0, help="replay this many actions (0 = all)")
    parser.add_argument("--no-sync", action="store_true", help="skip sync_controllers after reset_to")
    args = parser.parse_args(argv)

    import mimicgen  # noqa: F401  registers the environments
    from tmg.evaluation.rollout import make_env, sync_controllers

    observation = {"cameras": [], "robot_state": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"], "object_state": ["object"]}
    env = make_env(args.env, observation, "Panda")
    successes, drifts, obs_errs = 0, [], []
    with h5py.File(args.src, "r") as f:
        keys = indexed_keys(f["data"].keys())[: args.demos]
        for key in keys:
            demo = f["data"][key]
            states, actions = np.asarray(demo["states"]), np.asarray(demo["actions"])
            model = demo.attrs.get("model_file")
            model = model.decode() if isinstance(model, bytes) else model
            obs = env.reset_to({"states": states[0], "model": model})
            if not args.no_sync:
                sync_controllers(env)
            obs = env.get_observation()
            rec_obj, rec_eef = np.asarray(demo["obs"]["object"][0]), np.asarray(demo["obs"]["robot0_eef_pos"][0])
            obs_errs.append((float(np.abs(np.asarray(obs["object"]) - rec_obj).max()), float(np.abs(np.asarray(obs["robot0_eef_pos"]) - rec_eef).max())))
            n = len(actions) if args.steps <= 0 else min(args.steps, len(actions))
            eef_rec = np.asarray(demo["obs"]["robot0_eef_pos"])
            drift = []
            success = False
            for t in range(n):
                obs, _, _, _ = env.step(actions[t])
                if t + 1 < len(eef_rec):
                    drift.append(float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - eef_rec[t + 1])))
                if env.is_success()["task"]:
                    success = True
                    break
            successes += int(success)
            drifts.append((max(drift) if drift else 0.0, drift[min(len(drift) - 1, 49)] if drift else 0.0))
            print(f"{key}: obs err object {obs_errs[-1][0]:.4f} eef {obs_errs[-1][1]:.4f} | replay {'success' if success else 'fail'} "
                  f"after {t + 1}/{len(actions)} steps | eef drift at step 50 {drifts[-1][1]:.4f} m, max {drifts[-1][0]:.4f} m", flush=True)
    print(json.dumps({"env": args.env, "demos": len(keys), "replay_successes": successes,
                      "median_obs_err_object": float(np.median([e[0] for e in obs_errs])),
                      "median_eef_drift_step50_m": float(np.median([d[1] for d in drifts])),
                      "median_eef_drift_max_m": float(np.median([d[0] for d in drifts]))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
