"""Feed recorded observations to a trained policy through the server's own act path and compare its actions
with the recorded ones (teacher forcing).

    python scripts/check_policy_teacher_forcing.py <checkpoint dir> <common hdf5 dataset> [n_demos]

Low action error on the training demonstrations means the query path (observation assembly, normalization,
hidden-state handling) reproduces what the policy learned; high error points at that path rather than at the
policy or the data. Reports per-dimension MSE, the MSE of predicting zero for scale, and the sign agreement of the
gripper channel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from tmg.policy.server import PolicyRunner


def main() -> int:
    ckpt, data = Path(sys.argv[1]), sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    runner = PolicyRunner(ckpt, "cuda")
    print("info:", runner.info())
    print("normalizer modes:", getattr(getattr(runner.io, "normalizer", None), "modes", None))
    all_err, all_act = [], []
    with h5py.File(data, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:n]
        for key in keys:
            g = f["data"][key]
            rs = np.asarray(g["obs"]["robot_state"], dtype=np.float32)
            es = np.asarray(g["obs"]["object_state"], dtype=np.float32) if "object_state" in g["obs"] else None
            acts = np.asarray(g["actions"], dtype=np.float32)
            runner.reset(seed=0)
            preds = []
            for t in range(len(acts)):
                obs = {"observation.state": rs[t]}
                if es is not None:
                    obs["observation.environment_state"] = es[t]
                preds.append(runner.act(obs))
            preds = np.asarray(preds)
            err = preds - acts
            all_err.append(err); all_act.append(acts)
            print(f"{key}: steps {len(acts)} | mse per dim {np.round((err ** 2).mean(0), 4).tolist()} | mean |err| {np.abs(err).mean():.4f} "
                  f"| mse of predicting 0: {(acts ** 2).mean():.4f} | gripper sign agreement {(np.sign(preds[:, -1]) == np.sign(acts[:, -1])).mean():.3f}", flush=True)
    err = np.concatenate(all_err); acts = np.concatenate(all_act)
    print(f"ALL: mse {float((err ** 2).mean()):.4f} vs predicting 0: {float((acts ** 2).mean()):.4f}; "
          f"arm (first 6 dims) mse {float((err[:, :6] ** 2).mean()):.4f} vs 0: {float((acts[:, :6] ** 2).mean()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
