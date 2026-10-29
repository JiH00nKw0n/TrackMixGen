import pytest

from tmg.generation.method.ranking import (
    RankedSources,
    load_ranking,
    ranking_from_attempts,
    ranking_path,
    write_ranking,
)

TASKS = ("square", "coffee", "threading", "stack_three")


def test_committed_mimicgen_rankings_load_and_are_consistent():
    # the 3,000-attempt rankings (2026-10-29): every subtask orders all 10 sources by rate = successes / attempts,
    # the attempts of a subtask add up to the pool's 3,000, and the top half is the good group
    for task in TASKS:
        ranking = load_ranking(f"mimicgen/{task}")
        assert ranking.generator == "mimicgen" and ranking.task == task and ranking.default is None
        assert "3000 attempts" in ranking.basis and ranking.subtasks
        for subtask, ranked in ranking.subtasks.items():
            assert sorted(ranked.order) == list(range(10)), (task, subtask)
            assert sum(ranked.attempts.values()) == 3000, (task, subtask)
            for s in ranked.order:
                assert ranked.rate[s] == pytest.approx(ranked.successes[s] / ranked.attempts[s], abs=1e-4)
            rates = [ranked.rate[s] for s in ranked.order]
            assert rates == sorted(rates, reverse=True)
            assert ranked.split(good_fraction=0.5)[0] == set(ranked.order[:5])


def test_soft_and_dex_rankings_use_a_default_order():
    rope = load_ranking("softmimicgen/rope")
    assert rope.for_subtask(3).order == (0, 2, 1)
    assert rope.for_subtask(0).split(good_fraction=0.5) == ({0, 2}, {1})
    dex = load_ranking("dexmimicgen/pick_place")
    assert dex.for_subtask(0).split(good_count=2) == ({3, 1}, {0, 4, 2})
    with pytest.raises(ValueError):
        dex.for_subtask(0).split(good_count=5)
    with pytest.raises(ValueError):
        dex.for_subtask(0).split(good_fraction=1.0)


def test_ranking_from_attempts_orders_by_rate_then_index_and_roundtrips(tmp_path):
    records = [{"success": s, "src_demo_inds": inds} for s, inds in
               [(True, [0, 1]), (False, [0, 1]), (True, [1, 0]), (True, [1, 2]), (False, [2, 2]), (False, [2, 0])]]
    ranked = ranking_from_attempts(records, num_sources=4)
    assert ranked[0].order == (1, 0, 2, 3)            # rates 1.0, 0.5, 0.0, unseen
    assert ranked[0].attempts == {0: 2, 1: 2, 2: 2, 3: 0} and ranked[0].successes[1] == 2
    assert ranked[1].order == (0, 1, 2, 3)            # subtask 1: sources 0, 1, 2 all 1/2 -> ties broken by index
    path = write_ranking(ranking_path("gen/task", root=tmp_path), generator="gen", task="task", basis="test", subtasks=ranked)
    loaded = load_ranking("gen/task", root=tmp_path)
    assert loaded.for_subtask(1).order == (0, 1, 2, 3) and loaded.basis == "test"
    assert ranking_path("gen/task", root=tmp_path) == path


def test_ranked_sources_rejects_duplicates():
    with pytest.raises(ValueError):
        RankedSources.from_mapping({"order": [1, 1, 2]}, "x")
