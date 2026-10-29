import json
from pathlib import Path

from tmg.generation.generators.mimicgen.config_builder import build_generation_config, experiment_name

TEMPLATE = json.loads((Path(__file__).parent / "fixtures" / "mimicgen_square_template.json").read_text())
PROTOCOL = dict(keep_failed=True, max_num_failures=None, guarantee=False, select_src_per_subtask=True,
                selection_strategy="random", selection_strategy_kwargs=None, action_noise=0.05,
                interpolation_steps=5, num_fixed_steps=0, collect_obs=True, camera_names=[], camera_size=[84, 84],
                render_video=False, num_source_demos=None)


def test_build_generation_config():
    cfg = build_generation_config(TEMPLATE, PROTOCOL, env_name="Square_D1", source_dataset="/data/s.hdf5",
                                  output_folder="/out", attempts=20, seed=9001)
    assert cfg["name"] == "square"
    assert cfg["experiment"]["name"] == experiment_name("Square_D1", 20, 9001) == "Square_D1_n20_seed9001"
    assert cfg["experiment"]["generation"]["num_trials"] == 20
    assert cfg["experiment"]["generation"]["select_src_per_subtask"] is True
    assert cfg["experiment"]["num_demo_to_render"] == 0
    for key, sub in cfg["task"]["task_spec"].items():
        assert sub["selection_strategy"] == "random" and sub["selection_strategy_kwargs"] is None
        assert sub["action_noise"] == 0.05 and sub["num_interpolation_steps"] == 5
    # the template itself is untouched
    assert TEMPLATE["task"]["task_spec"]["subtask_1"]["selection_strategy"] == "nearest_neighbor_object"


def test_null_selection_keeps_the_official_per_subtask_values():
    """generation.protocol.select_src_per_subtask / selection_strategy = null: the template's source selection stays."""
    protocol = dict(PROTOCOL, select_src_per_subtask=None, selection_strategy=None, selection_strategy_kwargs=None)
    cfg = build_generation_config(TEMPLATE, protocol, env_name="Square_OURS", source_dataset="/data/s.hdf5",
                                  output_folder="/out", attempts=250, seed=30001)
    assert cfg["experiment"]["generation"]["select_src_per_subtask"] is False       # the template's value (one source per attempt)
    spec = cfg["task"]["task_spec"]
    assert spec["subtask_1"]["selection_strategy"] == "nearest_neighbor_object" and spec["subtask_1"]["selection_strategy_kwargs"] == {"nn_k": 3}
    assert spec["subtask_2"]["selection_strategy"] == "random" and spec["subtask_2"]["selection_strategy_kwargs"] is None
    assert spec["subtask_1"]["action_noise"] == 0.05 and spec["subtask_1"]["num_interpolation_steps"] == 5   # the protocol's other values still apply
