"""Policies, the training loop and the policy server.

- ``nets.py``: plain torch modules of the BC-RNN port (ResNet18 trunk, spatial softmax, crop randomizer, LSTM
  with a Gaussian-mixture head), importable without LeRobot.
- ``bc_rnn.py``: BC-RNN in LeRobot's ``PreTrainedPolicy`` format (config class, ``forward``, ``select_action``
  with the rolling hidden state), values from MimicGen's official image settings.
- ``lerobot_policies.py``: the policy table (``POLICY_SPECS``) and the construction of LeRobot 0.6.1 policies (Diffusion
  Policy with the official image U-Net values, π0 with the hub weights) from our policy configs and dataset features.
- ``normalize.py``: per-key normalization (identity, mean/std, min/max) applied before ``forward`` and
  ``select_action``, with the statistics saved next to every checkpoint.
- ``ema.py``: exponential moving average of the weights (Diffusion Policy evaluates the EMA copy).
- ``trainer.py``: the one training loop (update budget, epochs of 500 updates, periodic checkpoints with
  rollout validation, gradient-norm clipping and logging, resume, checkpoint selection).
- ``server.py``: websocket + msgpack policy server that the evaluation loop in the simulator image calls.
"""
