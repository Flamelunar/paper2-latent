import copy
import importlib.util
import json
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "exp_code_ljj/report_new_baselines_v18.py"
spec = importlib.util.spec_from_file_location("v18_new_report", MODULE_PATH)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
METHOD = "baseline1_zero_shot"
MODEL = "/app/hzj/models/Meta-Llama-3.1-8B-Instruct"
JUDGE = "/app/hzj/models/Qwen3-14B"


def row(index, rounds=None, wins=("A", "B")):
    base = {"index": index, "method": METHOD, "trial": 1, "scenario_id": f"s-{index}",
            "model": MODEL, "judge_model": JUDGE, "prompt_version": "v18",
            "prompt_revision": "joint_strict_v18_json_schema", "judge_compute_dtype": "bfloat16",
            "judge_quantization": "none", "max_parse_retries": 3}
    if rounds is None:
        return {**base, "parse_error": "No parseable JSON object", "verdict": None,
                "raw_outputs": ["invalid"] * 4, "raw_output": "invalid", "retry_outputs": ["invalid"] * 3,
                "model_calls": 4}
    raw = {"round_success": rounds, "win_ab": wins[0], "win_ba": wins[1]}
    output = json.dumps(raw)
    first = next((i + 1 for i, value in enumerate(rounds) if value), None)
    verdict = {**raw, "first_success_round": first, "success_within_5_rounds": first is not None,
               "win_credit": ((wins[0] == "A") + (wins[1] == "B")) / 2}
    return {**base, "parse_error": None, "verdict": verdict, "raw_outputs": [output],
            "raw_output": output, "retry_outputs": [], "model_calls": 1}


class V18ReportTests(unittest.TestCase):
    def generation_fixture(self):
        points, generations = [], []
        for index in range(1000):
            point = {'scenario_id': f's-{index}', 'source_id': f'source-{index}',
                     'session_id': 1, 'target_turn': 1, 'scenario': {'goal': 'A voluntary request'},
                     'reference': 'Golden request.', 'history': [{'role': 'persuadee', 'content': 'Earlier reply.'}]}
            turns = [{'round': position // 2 + 1,
                      'role': 'persuader' if position % 2 == 0 else 'persuadee',
                      'content': f'Public turn {position}'} for position in range(10)]
            generation = {**point, 'index': index, 'method': METHOD, 'persuader_model': MODEL,
                          'persuadee_model': report.PERSUADEE, 'rounds': 5, 'temperature': .95,
                          'top_p': .7, 'top_k': 50, 'max_new_tokens': 512,
                          'initial_history': copy.deepcopy(point['history']), 'trajectory': turns,
                          'final_history': copy.deepcopy(point['history']) + [{'role': turn['role'], 'content': turn['content']} for turn in turns]}
            points.append(point)
            generations.append(generation)
        return points, generations

    def test_full_generation_preflight_accepts_exact_public_histories(self):
        points, generations = self.generation_fixture()
        for generation in generations:
            for turn in generation['final_history']:
                turn['thinking_policy'] = 'final_answer_only'
        report.validate_generations(generations, points, method=METHOD, expected_model=MODEL)

    def test_full_generation_preflight_rejects_altered_or_private_histories(self):
        for view in ['initial_history', 'final_history', 'trajectory']:
            with self.subTest(view=view):
                points, generations = self.generation_fixture()
                generations[461][view][0]['content'] = '<think>private content</think>'
                with self.assertRaises(ValueError):
                    report.validate_generations(generations, points, method=METHOD, expected_model=MODEL)

    def fixture(self):
        generations = [{"scenario_id": f"s-{i}", "persuader_model": MODEL} for i in range(3)]
        rows = [row(0, [False, True, True, True, True]), row(1, [False] * 5, ("A", "A")), row(2)]
        return rows, generations

    def summarize(self, rows, generations):
        return report.summarize_rows(rows, generations, method=METHOD,
                                     judge_model=JUDGE, expected_rows=3)

    def test_unresolved_is_excluded_and_split_vote_is_half(self):
        rows, generations = self.fixture()
        stats = self.summarize(rows, generations)
        self.assertEqual(stats["parsed"], 2)
        self.assertEqual(stats["unresolved"], 1)
        self.assertEqual(stats["strict_acceptance_percent"], 50)
        self.assertEqual(stats["win_rate_percent"], 75)
        self.assertEqual(stats["mean_rounds"], 3.5)
        self.assertEqual(stats["extra_retry_calls"], 3)

    def test_mixed_prompt_or_judge_results_are_rejected(self):
        rows, generations = self.fixture()
        rows[0]["prompt_version"] = "v11"
        with self.assertRaises(ValueError):
            self.summarize(rows, generations)

    def test_duplicate_indices_are_rejected(self):
        rows, generations = self.fixture()
        rows[1]["index"] = 0
        with self.assertRaises(ValueError):
            self.summarize(rows, generations)

    def test_native_boolean_schema_and_derived_metrics_are_verified(self):
        rows, generations = self.fixture()
        rows[0]["verdict"]["win_credit"] = 0
        with self.assertRaises(ValueError):
            self.summarize(rows, generations)
        rows, generations = self.fixture()
        raw = json.loads(rows[0]["raw_output"])
        raw["round_success"][0] = "false"
        invalid = json.dumps(raw)
        rows[0]["raw_output"] = invalid
        rows[0]["raw_outputs"] = [invalid]
        with self.assertRaises(ValueError):
            self.summarize(rows, generations)

    def test_reporting_preserves_other_judges_and_our_placeholder(self):
        rows, generations = self.fixture()
        stats = self.summarize(rows, generations)
        text = ("| Qwen3-14B (BF16) | Baseline 1: Zero-shot | base | - | - | - | - | - |\n"
                "|   | Our | base | - | - | - | - | - |\n"
                "| GPT-6-Luna (API) | Baseline 1: Zero-shot | base | - | - | - | - | - |\n")
        updated = report.update_table(text, {METHOD: stats}, judge_prefix="Qwen3-14B")
        self.assertIn("| 2 | 50.00 | 75.00 | 3.500 | 1 |", updated)
        self.assertEqual(updated.splitlines()[1:], text.splitlines()[1:])


if __name__ == "__main__":
    unittest.main()
