#!/usr/bin/env python3
"""Write derived SkillMix rule files (tmg/generation/derive_coefficients.py): a task's fitted rule file changed by the
transforms of a named condition, under configs/experiment/generator/<generator>/coefficients/derived/<condition>/.

  scripts/derive_coefficients.py --all                       every condition of configs/experiment/derived_coefficients.yaml
                                                              for every task with a fitted rule file (a condition that
                                                              chooses subtasks by kind skips the tasks without kinds)
  scripts/derive_coefficients.py --generator mimicgen --task threading coffee --condition select_only uniform
  scripts/derive_coefficients.py --generator cpgen --task threading --condition force_source0_insert \\
      --transform '{type: set_d, subtasks: all, value: -1}' --transform '{type: boost_a, subtasks: [1], source: 0, by: 100}'
                                                              a condition defined on the command line (its transforms are
                                                              recorded in the file; a method reads it with
                                                              mixing.derived: force_source0_insert)
  scripts/derive_coefficients.py --check                     exit 1 when a derived file no longer matches its fitted file

Needs numpy and PyYAML (run it where the repository's package is installed, or inside an image: the files are committed).
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmg import CONFIGS_ROOT  # noqa: E402
from tmg.generation.derive_coefficients import (  # noqa: E402
    derived_files,
    is_current,
    load_conditions,
    write_derived,
)
from tmg.generation.method.skillmix import COEFFICIENTS_DIR  # noqa: E402
from tmg.generation.simulators.scenes import GENERATOR_DIR  # noqa: E402


def fitted_tasks(rule: str, root: Path) -> list[tuple[str, str]]:
    """``(generator, task)`` of every fitted ``rule`` file under the configs root."""
    files = sorted((root / GENERATOR_DIR).glob(f"*/{COEFFICIENTS_DIR}/{rule}/*.yaml"))
    return [(f.parents[2].name, f.stem) for f in files]


def check(root: Path) -> int:
    stale = [f for f in derived_files(root) if not is_current(f, root)]
    for f in stale:
        print(f"STALE {f.relative_to(root.parent)}: derive it again")
    print(f"{len(derived_files(root)) - len(stale)} derived rule files current, {len(stale)} stale")
    return 1 if stale else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator")
    ap.add_argument("--task", nargs="+", default=[])
    ap.add_argument("--condition", nargs="+", default=[])
    ap.add_argument("--transform", action="append", default=[], help="a transform as YAML; defines the one --condition given")
    ap.add_argument("--rule", default="softmax", help="the fitted rule the file is derived from")
    ap.add_argument("--all", action="store_true", help="every configured condition for every task with a fitted rule file")
    ap.add_argument("--check", action="store_true", help="verify the derived files against their fitted files; write nothing")
    ap.add_argument("--configs", type=Path, default=CONFIGS_ROOT, help="configs root")
    args = ap.parse_args(argv)
    root = args.configs
    if args.check:
        return check(root)
    configured = load_conditions(root)
    if args.transform:
        if len(args.condition) != 1:
            ap.error("--transform defines exactly one --condition")
        conditions = {args.condition[0]: [yaml.safe_load(t) for t in args.transform]}
    else:
        names = args.condition or (list(configured) if args.all else [])
        unknown = [c for c in names if c not in configured]
        if unknown:
            ap.error(f"unknown conditions {unknown}; configured: {sorted(configured)} (or define one with --transform)")
        conditions = {c: configured[c] for c in names}
    tasks = fitted_tasks(args.rule, root) if args.all else [(args.generator, t) for t in args.task]
    if not conditions or not tasks or any(g is None for g, _ in tasks):
        ap.error("give --all, or --generator, --task and --condition")
    for generator, task in tasks:
        for name, transforms in conditions.items():
            try:
                path = write_derived(generator, task, name, transforms, rule=args.rule, root=root)
            except ValueError as err:
                if not args.all:
                    raise
                print(f"skipped {generator}/{task} {name}: {err}")
                continue
            print(f"wrote {path.relative_to(root.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
