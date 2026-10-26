"""TrackMixGen: synthetic demonstration generators with the TrackMix method, datasets, policies and evaluation.

Where to read: ``README.md`` at the repository root explains the code structure and how to run an experiment;
``tmg/README.md`` describes this package folder by folder and module by module; ``configs/README.md`` describes
the configuration layers and groups; ``docs/GENERATORS.md`` describes how each official generator is grafted.
"""
from pathlib import Path

__version__ = "0.1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
CONFIGS_ROOT = REPO_ROOT / "configs"
