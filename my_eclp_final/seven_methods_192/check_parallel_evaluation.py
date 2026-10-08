"""Regression checks for round-five success and per-method writer isolation."""
import copy
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_io import writer_lock
from evaluate import summarize


def main():
    failure = {"success": False, "progress": "none"}
    accepted = {"success": True, "progress": "accepted"}
    win = {"credit": .5, "ab_credit": .5, "ba_credit": .5, "ab": "Tie", "ba": "Tie"}
    def record(goals):
        return {"goal_results": copy.deepcopy(goals), "win1": dict(win), "win_full": dict(win)}
    rows = [record([failure] * 4 + [accepted]), record([failure] * 5),
            record([None, failure, accepted, failure, failure]), record([None] + [failure] * 4)]
    result = summarize(rows, 4)
    assert result["success_count"] == 2
    assert result["success_evaluable_n"] == 3
    assert math.isclose(result["success_percent"], 200 / 3)
    assert math.isclose(result["mean_rounds_fail5"], 13 / 3)
    assert result["format_unresolved"] == 2
    with tempfile.TemporaryDirectory() as directory:
        first, second = Path(directory) / "first.lock", Path(directory) / "second.lock"
        with writer_lock(first), writer_lock(second):
            try:
                with writer_lock(first):
                    raise AssertionError("Same-method double writer was allowed")
            except RuntimeError:
                pass
    print({"passed": True, "round_five_success_counted": True,
           "unknown_denominator_preserved": True, "independent_method_locks": True})


if __name__ == "__main__":
    main()
