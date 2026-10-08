"""CPU-only fake-endpoint integration test of the real preference pipeline."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp_code_ljj"))
from repro.preference_optimization import build_preference_paths_batch, SIMULATION_PROTOCOL
from repro.public_answer import THINK, public_turn


class FakePersuader:
    def __init__(self):
        self.requests = []

    def complete(self, **kwargs):
        self.requests.append(kwargs)
        assert "PRIVATE_USER_BRAIN" not in kwargs["user"]
        return ("<think>PRIVATE_PERSUADER_BRAIN</think>Intent to Strategy Reasoning: context\n"
                "Selected Strategy: respectful request\nUtterance: Public request " + str(len(self.requests)))


class FakeUser:
    def __init__(self):
        self.requests = []

    def complete(self, **kwargs):
        self.requests.append(kwargs)
        assert "PRIVATE_PERSUADER_BRAIN" not in kwargs["user"]
        return "<think>PRIVATE_USER_BRAIN</think>Public reply"


seed = {"scenario_id":"fake-only-test", "source_id":"fake", "session_id":"fake", "target_turn":1,
        "scenario":{"background":"A harmless voluntary request.", "goal":"Consider the request.",
                    "persuader":"Alex", "persuadee":"Pat"}, "history":[], "strategies":["respectful request"]}
persuader, user = FakePersuader(), FakeUser()
rows = build_preference_paths_batch([seed], persuader, user, 4)
assert len(rows) == 1 and rows[0]["metadata"]["protocol"] == SIMULATION_PROTOCOL
assert not THINK.search(rows[0]["candidate_x"]) and not THINK.search(rows[0]["candidate_y"])
for path in (rows[0]["path_x"], rows[0]["path_y"]):
    assert len(path) == 8
    for position, turn in enumerate(path):
        assert turn["role"] == ("persuader" if position % 2 == 0 else "persuadee")
        assert turn["thinking_policy"] == "final_answer_only"
        assert turn["content"] and not THINK.search(turn["content"])
        assert "PRIVATE_" not in turn["content"]
assert rows[0]["metadata"]["thinking_filtered_turns"] >= 8
closing = public_turn("PRIVATE_USER_BRAIN</think>Public closing-only reply", seed=seed, role="persuadee", endpoint=user)
assert closing["content"] == "Public closing-only reply"


class RewriteUser:
    def complete(self, **kwargs):
        return "Only a public rewritten reply"


rewritten = public_turn("<think>PRIVATE_USER_BRAIN only", seed=seed, role="persuadee", endpoint=RewriteUser())
assert rewritten["content"] == "Only a public rewritten reply" and rewritten["thinking_filtered"]
print(json.dumps({"protocol":SIMULATION_PROTOCOL, "device":"CPU fake endpoints only",
                  "both_public_paths_clean":True, "no_private_text_forwarded_to_other_speaker":True,
                  "closing_only_tag_handled":True, "truncated_reasoning_rewritten":True}), flush=True)
