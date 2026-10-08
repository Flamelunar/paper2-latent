import importlib.util
import json
import tempfile
import unittest
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "baseline5_latentRATT/stage_checkpoint.py"
sys.path.insert(0, str(MODULE_PATH.parent))
spec = importlib.util.spec_from_file_location("stage_checkpoint", MODULE_PATH)
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)
MODEL = "/app/hzj/models/Meta-Llama-3.1-8B-Instruct"
DIGEST = "test-data-sha256"


def write_checkpoint(path, *, stage="sft", step=3742, complete=True, max_steps=None):
    (path / "adapter").mkdir(parents=True)
    for name in ("attention.pt", "training_state.pt", "adapter/adapter_model.safetensors"):
        (path / name).write_bytes(b"test fixture")
    (path / "adapter/adapter_config.json").write_text(json.dumps({"base_model_name_or_path": MODEL}))
    config = {"stage": stage, "model": MODEL, "train_sha256": DIGEST, "train_rows": 59868,
              "train_scenarios": 10000, "epochs": 1, "max_steps": max_steps,
              "batch_size": 2, "gradient_accumulation": 16,
              "compute_dtype": "bfloat16", "quantization": "none", "cuda_visible_devices": "0",
              "completed_steps": step, "training_complete": complete,
              "numerical_validation": "finite_loss_reward_gradient_v1",
              "last_loss": 0.5, "last_gradient_norm": 0.1, "last_reward": -0.2}
    (path / "run_config.json").write_text(json.dumps(config))


class StageCheckpointTests(unittest.TestCase):
    def inspect(self, output, stage="sft", max_steps=None):
        return checkpoint.inspect_stage(output, stage=stage, model=MODEL,
                                        train_sha256=DIGEST, max_steps=max_steps)

    def test_empty_stage_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.inspect(Path(directory)), ("start", None))

    def test_only_exact_full_stage_can_be_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft"
            write_checkpoint(path)
            self.assertEqual(self.inspect(path), ("complete", path))

    def test_sft_weights_cannot_be_misidentified_as_finished_rl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft"
            write_checkpoint(path)
            with self.assertRaises(ValueError):
                self.inspect(path, stage="rl")

    def test_smoke_checkpoint_cannot_replace_full_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke"
            write_checkpoint(path, step=1, max_steps=1, complete=False)
            self.assertEqual(self.inspect(path, max_steps=1), ("complete", path))
            with self.assertRaises(ValueError):
                self.inspect(path)

    def test_latest_complete_recovery_files_are_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_checkpoint(path / "checkpoint-100", step=100, complete=False)
            write_checkpoint(path / "oom-step-150", step=150, complete=False)
            self.assertEqual(self.inspect(path), ("resume", path / "oom-step-150"))

    def test_incomplete_newer_save_falls_back_to_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_checkpoint(path / "checkpoint-100", step=100, complete=False)
            (path / "checkpoint-200").mkdir()
            self.assertEqual(self.inspect(path), ("resume", path / "checkpoint-100"))

    def test_changed_data_never_silently_restarts_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft"
            write_checkpoint(path)
            with self.assertRaises(ValueError):
                checkpoint.inspect_stage(path, stage="sft", model=MODEL,
                                         train_sha256="different-data")

    def test_full_rl_and_separate_sft_checkpoint_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline5"
            write_checkpoint(path / "sft")
            write_checkpoint(path, stage="rl", step=29934)
            self.assertEqual(self.inspect(path, stage="rl"), ("complete", path))
            self.assertEqual(self.inspect(path / "sft"), ("complete", path / "sft"))

    def test_physical_gpu_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft"
            write_checkpoint(path)
            with self.assertRaises(ValueError):
                checkpoint.inspect_stage(path, stage="sft", model=MODEL,
                                         train_sha256=DIGEST, physical_gpu="1")

    def test_nan_smoke_is_never_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke"
            write_checkpoint(path, step=1, max_steps=1, complete=False)
            file = path / "run_config.json"
            config = json.loads(file.read_text())
            config["last_loss"] = float('nan')
            file.write_text(json.dumps(config))
            with self.assertRaises(FloatingPointError):
                self.inspect(path, max_steps=1)

    def test_missing_validation_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft"
            write_checkpoint(path)
            file = path / "run_config.json"
            config = json.loads(file.read_text())
            config.pop('numerical_validation')
            file.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                self.inspect(path)

    def test_zero_gradient_smoke_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'smoke'
            write_checkpoint(path, step=1, max_steps=1, complete=False)
            file = path / 'run_config.json'
            config = json.loads(file.read_text())
            config['last_gradient_norm'] = 0
            file.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                self.inspect(path, max_steps=1)


if __name__ == "__main__":
    unittest.main()
