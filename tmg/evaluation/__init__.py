"""Policy evaluation in the simulator image.

- ``resets.py``: ``tmg resets`` draws the fixed evaluation scenes of a (task, scene) once (validation and test
  splits) and stores each initial state with its model xml, so every policy, condition and seed is tested on
  the same scenes.
- ``client.py``: the websocket client of the policy server (``tmg.policy.server``).
- ``rollout.py``: ``tmg eval`` plays the fixed scenes with a served checkpoint, writes ``episodes.jsonl``
  (resumable per episode) and ``eval_summary.json`` (success rates with Wilson intervals), and can append the
  validation row of a training run.
"""
