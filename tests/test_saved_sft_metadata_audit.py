import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / 'exp_code_ljj/audit_saved_sft_metadata.py'
spec = importlib.util.spec_from_file_location('saved_sft_metadata_audit', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class SavedMetadataAuditTests(unittest.TestCase):
    def fixture(self, directory, *, model=None, bf16=True, fp16=False):
        output = directory / 'meta_llama31/baseline2_dialogue_sft'
        output.mkdir(parents=True)
        (output / 'training_args.bin').write_bytes(b'TEMPORARY TEST FIXTURE, NOT AN EXPERIMENT')
        base = model or audit.FAMILIES['meta_llama31'][0]
        (output / 'adapter_config.json').write_text(json.dumps({'base_model_name_or_path': base}))
        (output / 'train_metrics.json').write_text(json.dumps({'epoch': 1}))
        arguments = types.SimpleNamespace(output_dir=str(output), bf16=bf16, fp16=fp16,
                                          num_train_epochs=1, gradient_accumulation_steps=16)
        torch = types.SimpleNamespace(load=lambda *args, **kwargs: arguments)
        split = {'sha256': {'dialogue_sft': 'FIXTURE-HASH'},
                 'sft_sessions': 59868, 'sft_scenarios': 10000}
        return output, torch, split

    def test_retrospective_record_does_not_invent_historical_gpu_or_runtime_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, torch, split = self.fixture(directory)
            with mock.patch.object(audit, 'ADAPTER_ROOT', directory):
                report = audit.audit_one('meta_llama31', 'baseline2_dialogue_sft', split, torch)
            self.assertEqual(report['provenance_status'], 'retrospective_audit_not_original_runtime_record')
            self.assertEqual(report['requested_physical_gpu'], 0)
            self.assertNotIn('actual_physical_gpu', report)
            self.assertNotIn('original_runtime_data_sha256', report)
            self.assertTrue(report['saved_trainer_arguments']['bf16'])
            self.assertEqual(len(report['limitations']), 3)

    def test_wrong_family_and_wrong_precision_are_rejected(self):
        for changes in [{'model': '/another/base'}, {'bf16': False}, {'fp16': True}]:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                _, torch, split = self.fixture(directory, **changes)
                with mock.patch.object(audit, 'ADAPTER_ROOT', directory):
                    with self.assertRaises(ValueError):
                        audit.audit_one('meta_llama31', 'baseline2_dialogue_sft', split, torch)

    def test_audit_write_never_replaces_runtime_config_or_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output, torch, split = self.fixture(directory)
            runtime = output / 'run_config.json'
            runtime.write_text('ORIGINAL RUNTIME RECORD')
            weights = output / 'adapter_model.safetensors'
            weights.write_bytes(b'UNCHANGED TEST WEIGHT BYTES')
            with mock.patch.object(audit, 'ADAPTER_ROOT', directory):
                report = audit.audit_one('meta_llama31', 'baseline2_dialogue_sft', split, torch)
                target = audit.write_audit(report)
            self.assertEqual(json.loads(target.read_text()), report)
            self.assertEqual(runtime.read_text(), 'ORIGINAL RUNTIME RECORD')
            self.assertEqual(weights.read_bytes(), b'UNCHANGED TEST WEIGHT BYTES')

    def test_missing_stage_is_not_claimed_as_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(audit, 'ADAPTER_ROOT', Path(temporary)):
                self.assertIsNone(audit.audit_one('meta_llama31', 'baseline3_isr_sft', {}, None))


if __name__ == '__main__':
    unittest.main()
