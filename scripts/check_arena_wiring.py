#!/usr/bin/env python3
"""Check pinned Arena source and graft anchors without starting Isaac Sim."""
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

from bootstrap_native import ARENA_COMMIT, ARENA_LAB_COMMIT

root = Path(os.environ.get('TMG_NATIVE_ROOT', '~/.local/share/tmg-native')).expanduser()
src = root/'src/isaaclab-arena'
result = {'simulation_started': False, 'sources': {}, 'anchors': {}}
for name, expected in [('IsaacLab', ARENA_LAB_COMMIT), ('IsaacLab-Arena', ARENA_COMMIT)]:
    actual = subprocess.check_output(['git','-C',str(src/name),'rev-parse','HEAD'],text=True).strip()
    assert actual == expected, (name, actual, expected)
    result['sources'][name] = actual
for file, cls, method, anchors in [
    ('data_generator.py','DataGenerator','generate',['new_initial_state','results']),
    ('data_generator.py','DataGenerator','generate_eef_subtask_trajectory',['src_subtask_object_pose','transformed_seq']),
    ('waypoint.py','MultiWaypoint','execute',['play_action','success']),
]:
    path = src/'IsaacLab/source/isaaclab_mimic/isaaclab_mimic/datagen'/file
    tree = ast.parse(path.read_text())
    node = next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls
                for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==method)
    names = {a.id for n in ast.walk(node) if isinstance(n,ast.Assign) for t in n.targets
             for a in ast.walk(t) if isinstance(a,ast.Name)}
    assert set(anchors) <= names, (method, set(anchors)-names)
    if method == 'generate': assert isinstance(node.body[-1],ast.Return)
    result['anchors'][method] = {'present': anchors, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
result['passed'] = True
print(json.dumps(result,indent=2))
