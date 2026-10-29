"""Do not continue an experimental campaign after failed or incomplete generation."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from continue_after_pool import completed_pool


def test_pool_continuation_requires_successful_complete_exports(tmp_path):
    result = dict(successes=1000, chunks=[dict(ok=True, exit_code=0, successes=250,
                                            exported_episodes=250) for _ in range(4)])
    path = tmp_path / 'DONE.json'
    path.write_text(json.dumps(result))
    assert completed_pool(tmp_path, 1000) == result
    for field, value in [('exit_code', 1), ('ok', False), ('exported_episodes', 249), ('successes', 249)]:
        altered = json.loads(json.dumps(result))
        altered['chunks'][0][field] = value
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            completed_pool(tmp_path, 1000)


def test_missing_completion_record_never_starts_next_stage(tmp_path):
    with pytest.raises(FileNotFoundError):
        completed_pool(tmp_path, 1000)
