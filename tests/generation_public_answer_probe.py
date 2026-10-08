"""CPU fake-endpoint proof for single and batched five-round generation."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'exp_code_ljj'))
from repro.persuadee_simulation import PersuaderMethod, clean_utterance, simulate_case, simulate_cases_batch
from repro.public_answer import public_turn


class FakePersuader:
    def complete(self, **kwargs):
        assert 'PRIVATE_USER' not in kwargs['user']
        return '<think>Utterance: PRIVATE_PERSUADER</think>Public request.'

    def complete_batch(self, requests):
        return [self.complete(**request) for request in requests]


class FakeUser:
    def complete(self, **kwargs):
        assert 'PRIVATE_PERSUADER' not in kwargs['user']
        return 'PRIVATE_USER thought</think>Public reply.'

    def complete_batch(self, requests):
        return [self.complete(**request) for request in requests]


point = {'scenario_id': 'fake-generation', 'history': [], 'strategies': [],
         'scenario': {'background': 'A request.', 'goal': 'Consider it.', 'persuader': 'Alex', 'persuadee': 'Pat'}}
method = PersuaderMethod('baseline3_isr_simdpo', 'ISR', 'fake-llama-isr', True)
args = {'method': method, 'persuader': FakePersuader(), 'persuadee': FakeUser(),
        'rounds': 5, 'temperature': .95, 'max_new_tokens': 512, 'top_p': .7, 'top_k': 50}
single = simulate_case(point=point, **args)
batched = simulate_cases_batch(points=[point, copy.deepcopy(point)], **args)
assert clean_utterance('<think>Utterance: PRIVATE_PERSUADER</think>Public final.', is_isr=True) == 'Public final.'
for generated in [single, *batched]:
    assert len(generated['trajectory']) == 10
    expected = [{'role': turn['role'], 'content': turn['content']} for turn in generated['trajectory']]
    assert generated['final_history'] == expected
    assert 'PRIVATE_' not in json.dumps(generated)
    assert '<think>' not in json.dumps(generated) and '</think>' not in json.dumps(generated)
    for turn in generated['trajectory']:
        assert turn['content'] == ('Public request.' if turn['role'] == 'persuader' else 'Public reply.')
        assert turn['thinking_filtered']


class PrivateOnlyFirstRewrite:
    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        return '<think>PRIVATE failed rewrite only' if self.calls == 1 else 'Public rewritten reply.'


rewriter = PrivateOnlyFirstRewrite()
reply = public_turn('A public-looking draft.', seed=point, role='persuadee',
                    endpoint=rewriter, force_rewrite=True)
assert reply['content'] == 'Public rewritten reply.'
assert reply['public_answer_retries'] == 2
assert reply['thinking_filtered'], 'A failed private-only rewrite must count as filtered'
print(json.dumps({'device': 'CPU fake endpoints only', 'single_and_batch_five_rounds': 'passed',
                  'private_utterance_not_extracted': True, 'public_trajectory_matches_final_history': True,
                  'private_only_failed_rewrite_is_counted': True}), flush=True)
