from __future__ import annotations

import sys
import unittest
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from evaluate_qwen25_table2 import parse_success, score_row_batched, strict_success_prompt
from repro.simulation import user_prompt
from repro.train_backend import _qwen_assistant_mask_template


class Qwen25PipelineTests(unittest.TestCase):
    def test_native_qwen_template_gets_assistant_only_markers(self) -> None:
        template = "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
        adapted = _qwen_assistant_mask_template(template)
        self.assertIn("{% generation %}", adapted)
        self.assertIn("{%- if message.role == 'assistant' %}", adapted)
        self.assertIn("{%- else %}", adapted)

    def test_existing_generation_blocks_are_preserved(self) -> None:
        template = "{% generation %}{{ message.content }}{% endgeneration %}"
        self.assertEqual(_qwen_assistant_mask_template(template), template)

    def test_qwen_user_prompt_uses_only_existing_scenario_roles(self) -> None:
        scenario = {
            "background": "Alex is unsure about cycling.",
            "goal": "persuade Alex to cycle safely",
            "persuader": "Sam",
            "persuadee": "Alex",
        }
        system, prompt = user_prompt({"scenario": scenario}, [])
        self.assertIn("Alex", system)
        self.assertIn("Sam", system)
        self.assertIn(scenario["background"], prompt)
        self.assertIn(scenario["goal"], prompt)

    def test_strict_success_parser_handles_json_and_thinking_wrapper(self) -> None:
        self.assertEqual(
            parse_success('{"success":true,"evidence":"I will attend."}'),
            (True, "I will attend."),
        )
        self.assertEqual(
            parse_success('<think>Maybe not.</think>{"success":false,"evidence":"only considering"}'),
            (False, "only considering"),
        )

    def test_strict_success_prompt_excludes_future_rounds(self) -> None:
        row = {
            "scenario": {
                "background": "Alex is unsure about cycling.",
                "goal": "persuade Alex to cycle safely",
                "persuader": "Sam",
                "persuadee": "Alex",
            },
            "initial_history": [],
        }
        prefix = [
            {"role": "persuader", "content": "We can use a protected lane."},
            {"role": "persuadee", "content": "I will try cycling with you."},
            {"role": "persuader", "content": "Great."},
            {"role": "persuadee", "content": "Future response."},
        ]
        prompt = strict_success_prompt(row, prefix[:2], 1)
        self.assertIn("Prefix ends after round 1", prompt)
        self.assertIn("I will try cycling with you", prompt)
        self.assertNotIn("Future response", prompt)

    def test_batched_judging_keeps_order_swap_and_all_five_prefixes(self) -> None:
        class BatchJudge:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []
                self.requests: list[dict] = []
                self.outputs = iter([
                    '{"winner":"A"}',
                    '{"winner":"B"}',
                    '{"success":false,"evidence":"not yet"}',
                    '{"success":false,"evidence":"still considering"}',
                    '{"success":true,"evidence":"I will do it"}',
                    '{"success":true,"evidence":"I will do it"}',
                    '{"success":true,"evidence":"I will do it"}',
                ])

            def complete_batch(self, requests: list[dict]) -> list[str]:
                self.batch_sizes.append(len(requests))
                self.requests.extend(requests)
                return [next(self.outputs) for _ in requests]

        row = {
            "scenario": {
                "background": "Alex is unsure about cycling.",
                "goal": "persuade Alex to cycle safely",
                "persuader": "Sam",
                "persuadee": "Alex",
            },
            "initial_history": [],
            "reference": "Try the protected lane.",
            "trajectory": [
                {"role": role, "content": f"turn {index}"}
                for index, role in enumerate(["persuader", "persuadee"] * 5)
            ],
        }
        judge = BatchJudge()
        result = score_row_batched(row, judge, judge_batch_size=4)
        self.assertEqual(judge.batch_sizes, [4, 3])
        self.assertIn("Round 1: turn 0", judge.requests[0]["user"])
        self.assertIn("Round 5: turn 8", judge.requests[0]["user"])
        self.assertIn("Try the protected lane.", judge.requests[1]["user"])
        self.assertEqual(result["comparison_unit"], "all_five_persuader_replies_vs_single_golden_reply")
        self.assertEqual(result["win_credit"], 1.0)
        self.assertTrue(result["position_consistent"])
        self.assertEqual(len(result["strict_success_by_round"]), 5)
        self.assertEqual(result["first_success_round"], 3)
        self.assertTrue(result["strict_success_within_5_rounds"])

    def test_reusing_success_judgments_only_sends_pairwise_prompts(self) -> None:
        class BatchJudge:
            def __init__(self) -> None:
                self.requests: list[dict] = []

            def complete_batch(self, requests: list[dict]) -> list[str]:
                self.requests.extend(requests)
                return ['{"winner":"A"}', '{"winner":"B"}']

        row = {
            "scenario": {"background": "A", "goal": "G", "persuader": "P", "persuadee": "U"},
            "initial_history": [],
            "reference": "golden reply",
            "trajectory": [
                {"role": role, "content": f"turn {index}"}
                for index, role in enumerate(["persuader", "persuadee"] * 5)
            ],
        }
        saved = {
            "protocol": "previous_success_protocol",
            "strict_success_by_round": [
                {"round": number, "success": number == 3, "evidence": "saved"}
                for number in range(1, 6)
            ],
        }
        judge = BatchJudge()
        result = score_row_batched(row, judge, judge_batch_size=4, success_record=saved)
        self.assertEqual(len(judge.requests), 2)
        self.assertEqual(result["first_success_round"], 3)
        self.assertEqual(result["success_source_protocol"], "previous_success_protocol")

    def test_local_batched_decode_left_pads_and_slices_each_output(self) -> None:
        import threading

        import torch

        from repro.simulation import LocalTransformersEndpoint

        class FakeTokenizer:
            pad_token_id = 0

            def apply_chat_template(self, messages: list[dict], **kwargs: object) -> torch.Tensor:
                content = messages[-1]["content"]
                return torch.tensor([[1, 2, 3] if "long" in content else [4, 5]])

            def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
                return {9: "long answer", 8: "short answer"}[int(token_ids[-1])]

        class FakeModel:
            def generate(self, **kwargs: object) -> torch.Tensor:
                inputs = kwargs["input_ids"]
                masks = kwargs["attention_mask"]
                self_mask = masks.tolist()
                if self_mask != [[1, 1, 1], [0, 1, 1]]:
                    raise AssertionError(f"Unexpected left padding mask: {self_mask}")
                if kwargs["do_sample"]:
                    raise AssertionError("temperature-zero batches must use greedy decoding")
                return torch.cat([inputs, torch.tensor([[9], [8]])], dim=1)

        endpoint = LocalTransformersEndpoint.__new__(LocalTransformersEndpoint)
        endpoint.torch = torch
        endpoint.tokenizer = FakeTokenizer()
        endpoint.model = FakeModel()
        endpoint.input_device = "cpu"
        endpoint.context_limit = 16
        endpoint._generation_lock = threading.Lock()
        output = endpoint.complete_batch([
            {"system": "judge", "user": "long prompt", "temperature": 0.0, "max_tokens": 2},
            {"system": "judge", "user": "short prompt", "temperature": 0.0, "max_tokens": 2},
        ])
        self.assertEqual(output, ["long answer", "short answer"])


if __name__ == "__main__":
    unittest.main()
