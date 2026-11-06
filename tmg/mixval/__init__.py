"""Generation-stage check of source mixing on fixed scenes (branch mixval/threading-blend-ratio).

Question: on scenes where every one of the good, compatible sources fails, does a blend of two of them at an
intermediate share succeed, and does that success repeat under small scene perturbations? No policy is trained.

``plan`` writes the attempt list (scene, perturbation, primary, donor, primary share), ``entrypoint`` runs one shard of
it inside the CP-Gen image with every attempt fixed in advance (``runtime``), and ``summarize`` turns the per-attempt
results into the success tables.
"""
