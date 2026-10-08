"""CPU-only test of actual MA2P agent calls, with deliberately private JSON."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'exp_code_ljj'))
from baseline4_ma2p import agent_json, ma2p_turn


class FakeModel:
    def __init__(self):
        self.calls = []

    def complete(self, **kwargs):
        assert 'PRIVATE' not in kwargs['user']
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return '<think>{"preventive":"PRIVATE","generative":"PRIVATE"}</think>{"preventive":"cost","generative":"time"}'
        if len(self.calls) == 2:
            return 'PRIVATE world reasoning</think>{"likely_reaction":"concern","strategy":"voluntary public step"}'
        return '<think>PRIVATE spoken reasoning</think>A voluntary public request.'


point = {'scenario_id': 'fake-ma2p-test', 'scenario': {'background': 'A voluntary request.',
         'goal': 'Consider a request.', 'persuader': 'Alex', 'persuadee': 'Pat'}}
model = FakeModel()
spoken, trace = ma2p_turn(point, [], [], model, 1)
assert spoken == 'A voluntary public request.'
assert trace['perception'] == {'preventive': 'cost', 'generative': 'time'}
assert trace['world_model']['strategy'] == 'voluntary public step'
assert all(trace['thinking_filtered'].values())
assert 'PRIVATE' not in json.dumps(trace)
assert agent_json('<think>{"x":"PRIVATE"}</think>{"x":"PUBLIC"}', ('x',)) == {'x': 'PUBLIC'}
print(json.dumps({'device': 'CPU fake endpoints', 'ma2p_public_boundaries': 'passed',
                  'private_json_never_forwarded': True, 'closing_only_tag_handled': True}), flush=True)
