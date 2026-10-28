"""Optional checkpoint storage reduction without changing inference weights."""
from pathlib import Path


def prune_training_states(run_dir: Path, newest: Path) -> None:
    """Keep the newest committed resume state and every checkpoint's weights."""
    newest = newest.resolve()
    root = (run_dir / "checkpoints").resolve()
    if newest.parent != root or not (newest / "training_state.pt").is_file():
        raise ValueError("newest checkpoint has no committed resume state")
    for folder in root.glob("[0-9]*"):
        if not (folder / "checkpoint.json").is_file():
            continue
        if folder.is_symlink() or folder.resolve().parent != root:
            raise ValueError("checkpoint path escapes run")
        if int(folder.name) < int(newest.name):
            (folder / "training_state.pt").unlink(missing_ok=True)
