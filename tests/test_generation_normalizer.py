import copy
import importlib.util
import json
import tempfile
import unittest
import stat
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "exp_code_ljj/normalize_generation_thinking.py"
spec = importlib.util.spec_from_file_location("generation_normalizer", MODULE_PATH)
normalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalizer)


class NormalizerTests(unittest.TestCase):
    def test_fixed_initial_context_is_not_trimmed(self):
        initial = [{'role': 'persuadee', 'content': '  Source context.\n'}]
        row = {'index': 0, 'initial_history': copy.deepcopy(initial),
               'trajectory': [{'role': 'persuader', 'content': 'Public reply.'}],
               'final_history': copy.deepcopy(initial) + [{'role': 'persuader', 'content': 'Public reply.'}]}
        normalizer.normalize_row(row)
        self.assertEqual(row['initial_history'], initial)
        self.assertEqual(row['final_history'][0]['content'], initial[0]['content'])

    def test_different_source_history_cannot_be_silently_reconstructed(self):
        row = {'index': 0, 'initial_history': [{'role': 'persuadee', 'content': 'Source.'}],
               'trajectory': [], 'final_history': [{'role': 'persuadee', 'content': 'Changed.'}]}
        with self.assertRaises(ValueError):
            normalizer.normalize_row(row)

    def test_all_generated_views_are_cleaned_without_changing_reference(self):
        row = {
            "index": 0,
            "reference": "Keep the original reference <think> literally.",
            "scenario_id": "scenario-0",
            "trajectory": [{
                "round": 1, "role": "persuader",
                "content": "<think>private</think>Public reply",
                "raw_output": "<think>private</think>Public reply",
                "agent_trace": {"world_model_raw": "<think>secret</think>{\"strategy\":\"ask\"}",
                                "nested": ["<think>secret</think>safe", 1, True]},
            }],
            "final_history": [{"role": "persuader", "content": "<think>private</think>Public reply"}],
        }
        reference = row["reference"]
        normalizer.normalize_row(row)
        self.assertEqual(row["reference"], reference)
        self.assertEqual(row["trajectory"][0]["content"], "Public reply")
        self.assertEqual(row["final_history"][0]["content"], "Public reply")
        self.assertEqual(row["trajectory"][0]["agent_trace"]["nested"], ["safe", 1, True])
        self.assertTrue(row["trajectory"][0]["thinking_filtered"])
        once = copy.deepcopy(row)
        normalizer.normalize_row(row)
        self.assertEqual(row, once)

    def test_template_supplied_opening_tag_does_not_leak_analysis(self):
        self.assertEqual(normalizer.clean_text("private analysis</think>\nPublic answer"),
                         ("Public answer", True))

    def test_unfinished_thinking_is_not_accepted_as_empty_public_reply(self):
        row = {"index": 0, "trajectory": [{"content": "<think>analysis only"}]}
        with self.assertRaises(ValueError):
            normalizer.normalize_row(row)

    def test_failed_normalization_does_not_overwrite_original_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation.jsonl"
            original = json.dumps({"index": 0, "trajectory": [{"content": "<think>private"}]}) + "\n"
            path.write_text(original, encoding="utf-8")
            with self.assertRaises(ValueError):
                normalizer.normalize(path)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_normalization_preserves_file_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation.jsonl"
            path.write_text(json.dumps({"index": 0, "trajectory": [{"content": "public"}]}) + "\n", encoding="utf-8")
            path.chmod(0o640)
            normalizer.normalize(path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)


if __name__ == "__main__":
    unittest.main()
