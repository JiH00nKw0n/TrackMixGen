"""Simulator-level helpers shared by the adapters that run on the same simulator.

``robosuite.py`` turns a scene config (initial-placement ranges, reach constraint) into a registered
robosuite environment class on top of MimicGen's public classes. An Isaac Lab counterpart will apply the same
scene configs to Isaac Lab environment configs.
"""
