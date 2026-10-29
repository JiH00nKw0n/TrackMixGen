"""``tmg <command> [key=value ...]`` and ``tmg show <command> [key=value ...]``.

The commands, their help lines and the functions they run are the ``COMMANDS`` table of ``tmg.config``; this module
parses the command line, composes the config and calls the command's function with it. ``show`` prints the composed
configuration of any command without running it.
"""
from __future__ import annotations

import argparse
import sys

from omegaconf import OmegaConf

from tmg.config import COMMANDS, compose, resolved
from tmg.core import egl

EXAMPLES = """examples:
  tmg generate task=square generation=test/pilot20
  tmg generate task=square scene=ours method=trackmix generation=pool_robosuite_image generation.attempts=5000
  tmg show generate task=square
"""


def help_text() -> str:
    """The command list of ``tmg --help``, one line per ``COMMANDS`` entry plus ``show``."""
    width = max(len(name) for name in COMMANDS) + 2
    lines = [f"  {name:<{width}}{spec.help}" for name, spec in COMMANDS.items()]
    lines.append(f"  {'show':<{width}}print the composed configuration of a command without running it")
    return "commands:\n" + "\n".join(lines) + "\n" + EXAMPLES


HELP = help_text()


def fetch_sources(cfg) -> None:
    """``tmg sources``: download and prepare the human source demonstrations through the chosen generator."""
    from tmg.generation.generators import make_generator

    make_generator(resolved(cfg)).fetch_sources()


def run_command(name: str, cfg) -> int:
    """Run ``COMMANDS[name]`` on a composed config and turn its result into the exit code (``CommandSpec.exit_code``)."""
    spec = COMMANDS[name]
    result = spec.resolve()(cfg)
    if spec.exit_code is None:
        return 0
    return 0 if result[spec.exit_code] else 1


def main(argv=None) -> int:
    """Parse ``tmg <command> [key=value ...]`` and run the command.

    Args:
        argv: command-line arguments without the program name (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 when ``verify_baseline`` found a difference or ``check_scene`` found the
        scene unusable.

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
    return run_command(args.command, compose(args.command, args.rest))
