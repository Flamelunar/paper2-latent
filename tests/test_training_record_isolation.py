"""Exercise the real lightweight record writer without loading ML packages."""

import ast
import dataclasses
import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / 'exp_code_ljj/repro/train_backend.py'


@dataclasses.dataclass
class Spec:
    model_name_or_path: str
    output_dir: Path


class TrainingRecordIsolationTests(unittest.TestCase):
    def writer(self, directory):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in ['_write_run_config', '_atomic_json_record']]
        module = ast.Module(body=selected, type_ignores=[])
        namespace = {'Path': Path, 'TrainSpec': Spec, 'Any': object,
                     'json': json, 'os': os, 'hashlib': hashlib, 'tempfile': tempfile,
                     'datetime': datetime, 'timezone': timezone,
                     'asdict': dataclasses.asdict,
                     'sha256': lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                     'RUN_ROOT': directory / 'experiment',
                     'RUN_CONFIG_DIR': directory / 'experiment/configs'}
        exec(compile(module, str(SOURCE), 'exec'), namespace)
        return namespace

    def test_same_stage_two_families_never_share_central_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            namespace = self.writer(directory)
            configs = namespace['RUN_CONFIG_DIR']
            configs.mkdir(parents=True)
            old = configs / 'baseline3_isr_sft.json'
            old.write_text('OLD-HISTORICAL-RECORD', encoding='utf-8')
            for family in ['meta_llama31', 'deepseek_r1_qwen14b']:
                output = directory / 'adapters' / family / 'baseline3_isr_sft'
                namespace['_write_run_config'](output, Spec(family, output.parent),
                    {'output_dir': str(output), 'model_name_or_path': family, 'bf16': True}, 'Test')
            actual = [path for path in configs.glob('*.json') if path != old]
            self.assertEqual(len(actual), 2)
            self.assertEqual(old.read_text(encoding='utf-8'), 'OLD-HISTORICAL-RECORD')
            self.assertEqual({json.loads(path.read_text())['train_spec']['model_name_or_path']
                              for path in actual}, {'meta_llama31', 'deepseek_r1_qwen14b'})

    def test_adapter_record_has_exact_data_hash_and_runtime_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            namespace = self.writer(directory)
            data = directory / 'training.jsonl'
            data.write_text('{"case": 1}\n', encoding='utf-8')
            output = directory / 'adapters/meta_llama31/baseline3_simdpo'
            with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
                namespace['_write_run_config'](output, Spec('base-model', output.parent),
                    {'train_file': str(data), 'output_dir': str(output), 'bf16': True,
                     'quantization': 'none'}, 'Test')
            config = json.loads((output / 'run_config.json').read_text())
            self.assertEqual(config['train_sha256'], hashlib.sha256(data.read_bytes()).hexdigest())
            self.assertEqual(config['cuda_visible_devices'], '0')
            self.assertEqual(config['physical_gpu'], 0)
            self.assertTrue(config['bf16'])
            payload = json.loads((output / 'training_provenance.json').read_text())
            self.assertEqual(payload['resolved_args'], config)
            self.assertEqual(payload['recording_phase'], 'training_process_pre_model_load')

    def test_flat_dpo_config_preserves_existing_keys_and_validates_data_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            namespace = self.writer(directory)
            data = directory / 'preferences.jsonl'
            data.write_text('{}\n', encoding='utf-8')
            output = directory / 'adapters/deepseek/baseline3_simdpo'
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            resolved = {'stage': 'dpo', 'beta': .1, 'train_file': str(data),
                        'train_sha256': digest, 'simulation_config': {'family': 'deepseek'}}
            namespace['_write_run_config'](output, Spec('model', output.parent), resolved, 'Test')
            saved = json.loads((output / 'run_config.json').read_text())
            for key, value in resolved.items():
                self.assertEqual(saved[key], value)
            with self.assertRaises(ValueError):
                namespace['_write_run_config'](output, Spec('model', output.parent),
                    {**resolved, 'train_sha256': 'NOT-THE-DATA-HASH'}, 'Test')
            self.assertEqual(json.loads((output / 'run_config.json').read_text()), saved)

    def test_symlink_alias_of_same_output_uses_same_central_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            namespace = self.writer(directory)
            actual = directory / 'adapters/family/stage'
            actual.mkdir(parents=True)
            alias = directory / 'adapter_alias'
            alias.symlink_to(actual, target_is_directory=True)
            for output in [actual, alias]:
                namespace['_write_run_config'](output, Spec('model', output.parent),
                                               {'stage': 'sft'}, 'Test')
            self.assertEqual(len(list(namespace['RUN_CONFIG_DIR'].glob('*.json'))), 1)

    def test_failed_atomic_replace_preserves_existing_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            namespace = self.writer(directory)
            path = directory / 'run_config.json'
            path.write_text('{"old": true}\n', encoding='utf-8')
            with mock.patch.object(os, 'replace', side_effect=OSError('simulated I/O failure')):
                with self.assertRaises(OSError):
                    namespace['_atomic_json_record'](path, {'new': True})
            self.assertEqual(path.read_text(), '{"old": true}\n')
            self.assertEqual(list(directory.glob('.run_config.json.*')), [])


if __name__ == '__main__':
    unittest.main()
