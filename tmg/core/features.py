"""The batch keys shared by the loader, the policies, the server and the evaluators (LeRobot's naming).

A batch or an observation sent to the policy server carries ``observation.images.<camera>`` per camera,
``observation.state`` (the robot state keys concatenated), ``observation.environment_state`` (the object state
keys concatenated; low-dim datasets only) and ``action``. Every module spells these through the names below so
that a camera key is built and taken apart in exactly one way.
"""
from __future__ import annotations

IMAGE_PREFIX = "observation.images."
STATE = "observation.state"
ENV_STATE = "observation.environment_state"
ACTION = "action"
ACTION_IS_PAD = "action_is_pad"
TASK = "task"
OBSERVATION_PREFIX = "observation."


def image_feature(camera: str) -> str:
    """``observation.images.<camera>``."""
    return IMAGE_PREFIX + str(camera)


def is_image_key(key: str) -> bool:
    """True for ``observation.images.<camera>`` keys."""
    return str(key).startswith(IMAGE_PREFIX)


def camera_of(key: str) -> str:
    """The camera name of an ``observation.images.<camera>`` key."""
    return str(key).split(IMAGE_PREFIX, 1)[1]
