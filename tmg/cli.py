"""``tmg <command> [key=value ...]`` and ``tmg show <command> [key=value ...]``."""
from __future__ import annotations

import argparse
import sys

from omegaconf import OmegaConf

from tmg.config import COMMANDS, compose, resolved
from tmg.core import egl

HELP = """commands:
  generate      one pool of generation attempts (generator, task, scene, method, generation)
  verify_baseline  run one chunk officially and through the adapter (method=none) and compare the demos
  rank_sources     order a task's source demonstrations by their generation success in a baseline pool
  train            train one policy on one dataset (train image)
  select_checkpoint  choose a run's checkpoint from validation.jsonl (or the last one)
  serve            policy server for one checkpoint (train image)
  resets           draw the fixed evaluation scenes of (task, scene) (simulator image)
  eval             play the fixed scenes against a served checkpoint (simulator image)
  dataset          select successful attempts of a finished pool and write one training hdf5
  check_scene      reset a (task, scene) environment many times and report edge overhang, contacts, falls, reach
  sources       download and prepare the human source demonstrations of a task
  show          print the composed configuration of a command without running it
examples:
  tmg generate task=square generation=test/pilot20
  tmg generate task=square scene=ours method=trackmix generation=pool_robosuite_image generation.attempts=5000
  tmg show generate task=square
"""


def main(argv=None) -> int:
    """Parse ``tmg <command> [key=value ...]`` and run the command.

    Args:
        argv: command-line arguments without the program name (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 when ``verify_baseline`` found a difference.

    Raises:
        SystemExit: for unknown commands or malformed overrides (argparse / Hydra errors).
    """
    egl.apply()  # before anything imports robosuite (its import-time EGL device check)
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="tmg", description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(COMMANDS) + ["show"])
    parser.add_argument("rest", nargs=argparse.REMAINDER, help="for show: <command> then key=value; else key=value")
    args = parser.parse_args(argv)
    if args.command == "show":
        if not args.rest or args.rest[0] not in COMMANDS:
            parser.error("show needs a command name, e.g. tmg show generate task=square")
        cfg = compose(args.rest[0], args.rest[1:])
        print(OmegaConf.to_yaml(OmegaConf.create(resolved(cfg))))
        return 0
    cfg = compose(args.command, args.rest)
    if args.command == "generate":
        from tmg.generation.runner import run_pool

        run_pool(cfg)
        return 0
    if args.command == "verify_baseline":
        from tmg.generation.verify import verify_baseline

        return 0 if verify_baseline(cfg)["identical"] else 1
    if args.command == "sources":
        from tmg.generation.generators import make_generator

        make_generator(resolved(cfg)).fetch_sources()
        return 0
    if args.command == "rank_sources":
        from tmg.generation.rank_sources import rank_sources

        rank_sources(cfg)
        return 0
    if args.command == "dataset":
        from tmg.data.dataset import build_dataset

        build_dataset(cfg)
        return 0
    if args.command == "train":
        from tmg.policy.trainer import train

        train(cfg)
        return 0
    if args.command == "select_checkpoint":
        from tmg.policy.trainer import select_checkpoint

        plain = resolved(cfg)
        select_checkpoint(plain["output"], str(plain["training"].get("selection", "best_validation")))
        return 0
    if args.command == "serve":
        from tmg.policy.server import serve

        serve(cfg)
        return 0
    if args.command == "resets":
        from tmg.evaluation.resets import make_resets

        make_resets(cfg)
        return 0
    if args.command == "eval":
        from tmg.evaluation.rollout import evaluate

        evaluate(cfg)
        return 0
    if args.command == "check_scene":
        from tmg.generation.scene_check import check_scene

        return 0 if check_scene(cfg)["ok"] else 1
    parser.error(f"command {args.command!r} is not implemented yet")
    return 2
