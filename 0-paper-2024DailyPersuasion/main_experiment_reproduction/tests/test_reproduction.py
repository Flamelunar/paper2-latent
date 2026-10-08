from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from repro.data import (
    build_persuader_rows,
    build_user_model_rows,
    public_scenario_key,
    scenario_id,
    select_preference_seed,
    trajectory_quality,
)
from repro.constants import PAPER_TARGETS
from repro.evaluation import parse_winner, rouge_l, rouge_l_released, run_evaluation
from repro.simulation import (
    LocalTransformersEndpoint,
    build_one_preference,
    endpoint_api_key,
    extract_utterance,
    generation_endpoint,
    parse_comparison,
)
from repro.train_backend import dpo_reference_mode


def fixture(raw_id: str = "non-unique-id", background: str = "Alex dislikes cycling.") -> dict:
    return {
        "id": raw_id,
        "scenario": {
            "tag": "Cycling",
            "background": background,
            "persuadee": "Alex",
            "persuader": "Sam",
            "goal": "persuade Alex to cycle",
            "domain": [{"tag": "Cycling", "domain": ["Health"]}],
        },
        "strategy": ["Address concerns", "Offer evidence"],
        "dialog": {
            "session_1": [
                {
                    "role": "persuader",
                    "analysis": "Alex is worried.",
                    "strategy": "Address concerns",
                    "response": "What worries you about cycling?",
                },
                {"role": "persuadee", "response": "It seems unsafe."},
                {
                    "role": "persuader",
                    "analysis": "Safety is the main concern.",
                    "strategy": "Offer evidence",
                    "response": "We can use a protected cycle lane.",
                },
                {"role": "persuadee", "response": "I could try that."},
            ]
        },
    }


class DataTests(unittest.TestCase):
    def test_identity_uses_scenario_content_not_released_id(self) -> None:
        first = fixture()
        same = fixture(raw_id="another-id")
        different = fixture(background="Alex enjoys cycling.")
        self.assertEqual(public_scenario_key(first), public_scenario_key(same))
        self.assertEqual(scenario_id(first), scenario_id(same))
        self.assertNotEqual(scenario_id(first), scenario_id(different))

    def test_sft_has_one_multiturn_row_per_session(self) -> None:
        dialogue = list(build_persuader_rows([fixture()], with_isr=False))
        isr = list(build_persuader_rows([fixture()], with_isr=True))
        self.assertEqual(len(dialogue), 1)
        self.assertEqual(len(isr), 1)
        self.assertEqual(len(dialogue[0]["history"]), 1)
        self.assertNotIn("Intent to Strategy Reasoning:", dialogue[0]["output"])
        self.assertIn("Intent to Strategy Reasoning:", isr[0]["output"])
        self.assertIn("Selected Strategy:", isr[0]["output"])
        self.assertEqual(dialogue[0]["metadata"]["scenario_id"], scenario_id(fixture()))

    def test_user_model_row_preserves_prior_turn(self) -> None:
        rows = list(build_user_model_rows([fixture()]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["history"]), 1)
        self.assertEqual(rows[0]["instruction"], "We can use a protected cycle lane.")
        self.assertEqual(rows[0]["output"], "I could try that.")

    def test_preference_sampling_is_deterministic(self) -> None:
        import random

        first = select_preference_seed(fixture(), random.Random(42))
        second = select_preference_seed(fixture(), random.Random(42))
        self.assertEqual(first, second)
        self.assertEqual(first["scenario_id"], scenario_id(fixture()))

    def test_quality_selector_requires_a_complete_adaptive_trajectory(self) -> None:
        item = fixture()
        session = item["dialog"]["session_1"]
        session.extend(
            [
                {
                    "role": "persuader",
                    "analysis": "Alex remains uncertain about safety.",
                    "strategy": "Offer evidence",
                    "response": "The protected lane keeps cyclists separated from traffic.",
                },
                {"role": "persuadee", "response": "I am still not sure about it."},
                {
                    "role": "persuader",
                    "analysis": "Alex needs a low-risk way to begin.",
                    "strategy": "Address concerns",
                    "response": "We can start with a short ride together on the protected lane.",
                },
                {"role": "persuadee", "response": "That makes sense; I am willing to try."},
                {
                    "role": "persuader",
                    "analysis": "Alex is ready for a specific next step.",
                    "strategy": "Offer evidence",
                    "response": "I will bring a helmet and meet you Saturday morning.",
                },
                {"role": "persuadee", "response": "I will give it a try."},
            ]
        )
        quality = trajectory_quality(item, "session_1", session)
        self.assertIsNotNone(quality)
        assert quality is not None
        self.assertEqual(quality["quality_level"], "A")
        self.assertEqual(quality["persuader_turns"], 5)


class ProtocolTests(unittest.TestCase):
    def test_extract_utterance(self) -> None:
        raw = "Intent to Strategy Reasoning: x\nSelected Strategy: y\nUtterance: Hello."
        self.assertEqual(extract_utterance(raw), "Hello.")

    def test_pairwise_parsers(self) -> None:
        self.assertEqual(parse_winner('{"winner":"b","reason":"clearer"}'), "B")
        parsed = parse_comparison(
            '{"summary_history":"h","summary_X":"x","summary_Y":"y",'
            '"explain":"x is better","result":"Uni-X"}'
        )
        self.assertEqual(parsed["result"], "Uni-X")

    def test_rouge_l_identical_text_is_100(self) -> None:
        self.assertGreater(rouge_l("Use the protected lane.", "Use the protected lane."), 99.99)

    def test_headline_rouge_is_character_level(self) -> None:
        candidate = "ab c"
        reference = "a bc"
        self.assertEqual(rouge_l(candidate, reference), 100.0)
        self.assertLess(rouge_l_released(candidate, reference), 100.0)

    def test_endpoint_key_matches_provider(self) -> None:
        environment = {"OPENAI_API_KEY": "openai", "CHATANYWHERE_API": "proxy"}
        with patch.dict("os.environ", environment, clear=True):
            self.assertEqual(endpoint_api_key("https://api.openai.com/v1"), "openai")
            self.assertEqual(endpoint_api_key("https://api.chatanywhere.tech/v1"), "proxy")
        with patch.dict("os.environ", {"CHATANYWHERE_API": "proxy"}, clear=True):
            self.assertIsNone(endpoint_api_key("https://api.openai.com/v1"))

    def test_local_generation_endpoint_dispatch(self) -> None:
        with patch("repro.simulation.LocalTransformersEndpoint") as local_endpoint:
            sentinel = object()
            local_endpoint.return_value = sentinel
            result = generation_endpoint(
                base_url="local://cuda:1",
                model="/tmp/adapter",
                api_key="unused",
                max_retries=3,
            )
        self.assertIs(result, sentinel)
        local_endpoint.assert_called_once_with(model="/tmp/adapter", device="cuda:1")

    def test_local_endpoint_accepts_json_judgment_mode(self) -> None:
        import threading
        import torch

        class FakeTokenizer:
            def __init__(self) -> None:
                self.messages = None
                self.pad_token_id = 0

            def apply_chat_template(self, messages: list[dict], **_: object):
                self.messages = messages
                return torch.tensor([[1, 2]])

            def decode(self, *_: object, **__: object) -> str:
                return '{"result":"Uni-X"}'

        class FakeModel:
            def generate(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
                return torch.cat((input_ids, torch.tensor([[3]])), dim=1)

        endpoint = object.__new__(LocalTransformersEndpoint)
        endpoint.tokenizer = FakeTokenizer()
        endpoint.model = FakeModel()
        endpoint.input_device = "cpu"
        endpoint.torch = torch
        endpoint._generation_lock = threading.Lock()
        result = endpoint.complete(
            system="Judge the paths.", user="Choose one.", temperature=0.0, json_mode=True
        )
        self.assertEqual(result, '{"result":"Uni-X"}')
        self.assertIn("Return only a valid JSON object", endpoint.tokenizer.messages[1]["content"])

    def test_evaluation_rejects_invalid_worker_and_retry_counts(self) -> None:
        kwargs = {
            "endpoint_specs": {},
            "judge_url": "https://api.openai.com/v1",
            "judge_model": "gpt-4o-mini",
            "workers": 0,
            "max_retries": 1,
            "force": False,
            "generate_only": False,
            "only_method": None,
            "judge_only": False,
        }
        with self.assertRaisesRegex(ValueError, "workers"):
            run_evaluation(**kwargs)
        kwargs.update(workers=1, max_retries=0)
        with self.assertRaisesRegex(ValueError, "max_retries"):
            run_evaluation(**kwargs)

    def test_dpo_reference_modes(self) -> None:
        self.assertEqual(dpo_reference_mode("lora"), "trl_reference_adapter")
        self.assertEqual(dpo_reference_mode("full"), "separate_full_model")
        with self.assertRaises(ValueError):
            dpo_reference_mode("bad")

    def test_paper_main_table_targets(self) -> None:
        values = [(row["win_rate"], row["rouge_l"]) for row in PAPER_TARGETS.values()]
        self.assertEqual(
            values, [(20.1, 17.4), (46.7, 24.4), (47.8, 25.6), (60.4, 25.2)]
        )

    def test_dpo_row_uses_longest_prefix_to_break_reward_tie(self) -> None:
        class QueueEndpoint:
            def __init__(self, outputs: list[str]) -> None:
                self.outputs = iter(outputs)

            def complete(self, **_: object) -> str:
                return next(self.outputs)

        item = fixture()
        seed = {
            "scenario_id": scenario_id(item),
            "source_id": item["id"],
            "session_id": "session_1",
            "target_turn": 1,
            "scenario": item["scenario"],
            "strategies": item["strategy"],
            "history": [],
        }
        persuader = QueueEndpoint([
            "Intent to Strategy Reasoning: x\nSelected Strategy: A\nUtterance: Candidate X",
            "Intent to Strategy Reasoning: y\nSelected Strategy: B\nUtterance: Candidate Y",
            "Intent to Strategy Reasoning: x2\nSelected Strategy: A\nUtterance: Follow-up X",
            "Intent to Strategy Reasoning: y2\nSelected Strategy: B\nUtterance: Follow-up Y",
        ])
        user = QueueEndpoint(["Reply X1", "Reply X2", "Reply Y1", "Reply Y2"])
        judge = QueueEndpoint([
            '{"summary_history":"","summary_X":"x","summary_Y":"y",'
            '"explain":"x","result":"Uni-X"}',
            '{"summary_history":"","summary_X":"x","summary_Y":"y",'
            '"explain":"y","result":"Uni-Y"}',
        ])
        row = build_one_preference(seed, persuader, user, judge, [1.0, 1.0])
        self.assertEqual(row["output"][0].split("Utterance: ")[-1], "Candidate Y")
        self.assertTrue(row["metadata"]["reward_tied"])
        self.assertEqual(row["metadata"]["tie_break"], "longest_prefix_judgment")
        self.assertEqual(row["metadata"]["prefix_decisions"][0]["prefix_utterances"], 2)


if __name__ == "__main__":
    unittest.main()
