import pytest

from tmg.generation.graft import Graft, GraftError, source_hash


class Thing:
    def run(self, items):
        total = 0
        for seq in items:
            for j in range(len(seq)):
                value = seq[j]
                total = total + value
        results = {"total": total}
        return results


def test_graft_inserts_and_replaces():
    calls = []

    class Hook:
        def begin(self, seq):
            calls.append(("begin", list(seq)))

        def indices(self, seq):
            return range(len(seq))

        def pick(self, seq, j):
            return seq[j] * 10

        def end(self, results):
            results["seen"] = True

    g = Graft(Thing.run, expected_hash=source_hash(Thing.run), label="Thing.run")
    g.insert_at_for_start("seq", "_h.begin(seq)")
    g.replace_for_iter("j", "_h.indices(seq)")
    g.replace_assign_value("value", "_h.pick(seq, j)")
    g.insert_before_final_return("_h.end(results)")
    Thing.run = g.compile({"_h": Hook()})
    out = Thing().run([[1, 2], [3]])
    assert out == {"total": 60, "seen": True}
    assert calls == [("begin", [1, 2]), ("begin", [3])]
    assert len(Thing.run.__tmg_graft__["edits"]) == 4


def test_hash_mismatch_refuses():
    with pytest.raises(GraftError):
        Graft(Thing.run, expected_hash="0" * 64, label="x")


def test_missing_anchor_refuses():
    g = Graft(Thing.run, expected_hash=None, label="x")
    with pytest.raises(GraftError):
        g.insert_after_assign("nonexistent", "pass")
