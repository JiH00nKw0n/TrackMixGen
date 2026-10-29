"""Policy evaluation in the simulator image.

- ``resets.py``: ``tmg resets`` draws the fixed evaluation scenes of a (task, scene) once (validation and test
  splits) and stores each initial state with its model xml (robosuite) or its reset seed (Isaac Lab), so every
  policy, condition and seed is tested on the same scenes.
- ``client.py``: the websocket client of the policy server (``tmg.policy.server``).
- ``session.py``: the loop both evaluators share: server checks, the per-episode policy seed, the episode rows,
  the summary with Wilson intervals; no simulator imports.
- ``envs.py``: which simulator a condition evaluates in, the robosuite settings shared by the evaluators, and the
  generation environment rebuilt from a dataset's ``env_args`` (CP-Gen).
- ``rollout.py``: ``tmg eval`` plays the fixed scenes with a served checkpoint, writes ``episodes.jsonl``
  (resumable per episode) and ``eval_summary.json`` under ``<run>/eval/<split>_h<horizon>/<checkpoint>/``, and can
  append the validation row of a training run; the robosuite episode lives here.
- ``isaaclab_rollout.py``: the Isaac Lab episode (Isaac Sim in the generator's image, scenes fixed by seeds).
"""
