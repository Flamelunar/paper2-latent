import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / 'exp_code_ljj'
sys.path.insert(0, str(EXP))
spec = importlib.util.spec_from_file_location('b2_history_repair', EXP / 'repair_baseline2_history.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class FakeEndpoint:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.instances.append(self)

    def close(self):
        self.closed = True


class HistoryRepairTests(unittest.TestCase):
    def fixture(self, exp):
        adapter = exp / 'adapters'
        model = str(adapter / repair.FAMILY / repair.METHOD)
        points, rows = [], []
        for index in range(1000):
            point = {'scenario_id': f's-{index}', 'source_id': f'source-{index}',
                     'session_id': 1, 'target_turn': 1, 'scenario': {'goal': 'Voluntary request'},
                     'reference': 'Unchanged golden.', 'history': [{'role': 'persuadee', 'content': 'Earlier.'}]}
            turns = [{'round': position // 2 + 1, 'role': 'persuader' if position % 2 == 0 else 'persuadee',
                      'content': f'Original public {position}'} for position in range(10)]
            row = {**point, 'index': index, 'method': repair.METHOD, 'persuader_model': model,
                   'persuadee_model': repair.PERSUADEE, 'rounds': 5, 'temperature': .95,
                   'top_p': .7, 'top_k': 50, 'max_new_tokens': 512,
                   'initial_history': copy.deepcopy(point['history']), 'trajectory': turns,
                   'final_history': copy.deepcopy(point['history']) + [{'role': turn['role'], 'content': turn['content']} for turn in turns]}
            if index in repair.ALLOWED_REPAIRS:
                row['final_history'][4]['content'] = 'Old inconsistent history.'
            points.append(point)
            rows.append(row)
        data = exp / 'data'
        data.mkdir()
        payload = ''.join(json.dumps(point) + '\n' for point in points)
        (data / 'evaluation_points.jsonl').write_text(payload)
        (data / 'manifest.json').write_text(json.dumps({'files': {'evaluation_points': {
            'sha256': hashlib.sha256(payload.encode()).hexdigest()}}}))
        file = exp / 'artifacts/generation_5rounds' / repair.FAMILY / (repair.METHOD + '.jsonl')
        file.parent.mkdir(parents=True)
        file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return adapter, points, rows, file

    def simulate(self, **kwargs):
        self.assertEqual(kwargs['rounds'], 5)
        self.assertEqual(kwargs['temperature'], .95)
        self.assertEqual(kwargs['top_p'], .7)
        self.assertEqual(kwargs['top_k'], 50)
        self.assertEqual(kwargs['max_new_tokens'], 512)
        point = kwargs['point']
        self.calls.append(point['scenario_id'])
        turns = [{'round': position // 2 + 1, 'role': 'persuader' if position % 2 == 0 else 'persuadee',
                  'content': f'Regenerated actual turn {position}'} for position in range(10)]
        return {'initial_history': copy.deepcopy(point['history']), 'trajectory': turns,
                'final_history': copy.deepcopy(point['history']) + [{'role': turn['role'], 'content': turn['content']} for turn in turns]}

    def setUp(self):
        self.calls = []
        FakeEndpoint.instances = []

    def test_only_three_full_scenes_are_regenerated_and_other_997_stay_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory)
            adapter, points, original, file = self.fixture(exp)
            changed = repair.repair_llama_baseline2(exp_root=exp, adapter_root=adapter,
                       endpoint_factory=FakeEndpoint, simulator=self.simulate)
            self.assertEqual(set(changed), repair.ALLOWED_REPAIRS)
            self.assertEqual(self.calls, ['s-455', 's-478', 's-618'])
            with file.open() as handle:
                actual = [json.loads(line) for line in handle]
            for index in range(1000):
                if index not in repair.ALLOWED_REPAIRS:
                    self.assertEqual(actual[index], original[index])
                else:
                    self.assertEqual(len(actual[index]['trajectory']), 10)
                    self.assertTrue(all(turn['content'].startswith('Regenerated actual') for turn in actual[index]['trajectory']))
                    self.assertEqual(actual[index]['reference'], original[index]['reference'])
            self.assertTrue(all(endpoint.closed for endpoint in FakeEndpoint.instances))
            self.assertEqual(repair.repair_llama_baseline2(exp_root=exp, adapter_root=adapter,
                             endpoint_factory=FakeEndpoint, simulator=self.simulate), [])
            self.assertEqual(len(FakeEndpoint.instances), 2)

    def test_failed_repair_preserves_source_and_reuses_only_complete_scene_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory)
            adapter, points, rows, file = self.fixture(exp)
            source = file.read_bytes()
            def failing(**kwargs):
                if kwargs['point']['scenario_id'] == 's-478':
                    raise RuntimeError('simulated interruption')
                return self.simulate(**kwargs)
            with self.assertRaises(RuntimeError):
                repair.repair_llama_baseline2(exp_root=exp, adapter_root=adapter,
                           endpoint_factory=FakeEndpoint, simulator=failing)
            self.assertEqual(file.read_bytes(), source)
            self.assertTrue(all(endpoint.closed for endpoint in FakeEndpoint.instances))
            repair.repair_llama_baseline2(exp_root=exp, adapter_root=adapter,
                       endpoint_factory=FakeEndpoint, simulator=self.simulate)
            self.assertEqual(self.calls, ['s-455', 's-478', 's-618'])

    def test_unreviewed_mismatch_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory)
            adapter, points, rows, file = self.fixture(exp)
            rows[999]['final_history'][-1]['content'] = 'Another unexpected mismatch.'
            file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            source = file.read_bytes()
            with self.assertRaises(ValueError):
                repair.repair_llama_baseline2(exp_root=exp, adapter_root=adapter,
                           endpoint_factory=FakeEndpoint, simulator=self.simulate)
            self.assertEqual(file.read_bytes(), source)
            self.assertEqual(FakeEndpoint.instances, [])
