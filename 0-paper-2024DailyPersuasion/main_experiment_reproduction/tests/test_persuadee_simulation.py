from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from repro.persuadee_simulation import (
    METHODS,
    TEST_PROTOCOL,
    generate_method,
    simulate_case,
    simulate_cases_batch,
    validate_devices,
)


def point() -> dict:
    return {
        "history": [],
        "scenario_id": "scenario-1",
        "source_id": "source-1",
        "session_id": "session-1",
        "target_turn": 1,
        "scenario": {
            "background": "Alex is unsure about cycling.",
            "goal": "persuade Alex to cycle safely",
            "persuader": "Sam",
            "persuadee": "Alex",
            "tag": "cycling",
        },
        "strategies": ["Address safety concerns"],
    }


class QueueEndpoint:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict] = []

    def complete(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return next(self.responses)


class BatchQueueEndpoint:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.batches: list[list[dict]] = []

    def complete_batch(self, requests: list[dict]) -> list[str]:
        self.batches.append(requests)
        return [next(self.responses) for _ in requests]


class PersuadeeSimulationTests(unittest.TestCase):
    def test_registry_matches_the_three_requested_checkpoints(self) -> None:
        self.assertEqual([method.slug for method in METHODS], [
            "01_zero_shot", "02_dialogue_sft", "03_isr_sft",
        ])
        self.assertEqual(METHODS[0].model_path, "/data1/liujianjian/model/Meta-Llama-3.1-8B-Instruct")
        self.assertEqual(
            METHODS[1].model_path,
            "/data1/liujianjian/Persuader/llama3.1-8B-Instruct-dialogue-merged",
        )
        self.assertEqual(
            METHODS[2].model_path,
            "/data1/liujianjian/Persuader/llama3.1-8B-Instruct-isr-merged",
        )

    def test_two_rounds_alternate_four_turns_and_keep_isr_raw_output(self) -> None:
        persuader = QueueEndpoint([
            "Intent to Strategy Reasoning: Alex worries about safety.\n"
            "Selected Strategy: Address safety concerns\nUtterance: We can use a protected lane.",
            "Intent to Strategy Reasoning: Alex is considering it.\n"
            "Selected Strategy: Address safety concerns\nUtterance: I can ride with you the first time.",
        ])
        persuadee = QueueEndpoint(["That sounds safer.", "I could try that."])
        result = simulate_case(
            point=point(), method=METHODS[2], persuader=persuader, persuadee=persuadee,
            rounds=2, temperature=0.0, max_new_tokens=1024,
        )
        self.assertEqual([turn["role"] for turn in result["trajectory"]], [
            "persuader", "persuadee", "persuader", "persuadee",
        ])
        self.assertEqual(result["trajectory"][0]["content"], "We can use a protected lane.")
        self.assertIn("Intent to Strategy Reasoning", result["trajectory"][0]["raw_output"])
        self.assertEqual(result["trajectory"][0]["prompt_type"], "isr")
        self.assertIn("Dialogue so far:\nPersuader: We can use a protected lane.", persuadee.calls[0]["user"])

    def test_five_round_batched_simulation_preserves_each_context_history(self) -> None:
        second = point()
        second["scenario_id"] = "scenario-2"
        second["source_id"] = "source-2"
        second["session_id"] = "session-2"
        second["scenario"] = {**second["scenario"], "persuader": "Taylor", "persuadee": "Morgan"}
        points = [point(), second]
        persuader_responses = [
            f"Persuader case {case} round {round_number}."
            for round_number in range(1, 6)
            for case in range(1, 3)
        ]
        persuadee_responses = [
            f"Persuadee case {case} round {round_number}."
            for round_number in range(1, 6)
            for case in range(1, 3)
        ]
        persuader = BatchQueueEndpoint(persuader_responses)
        persuadee = BatchQueueEndpoint(persuadee_responses)
        results = simulate_cases_batch(
            points=points,
            method=METHODS[1],
            persuader=persuader,
            persuadee=persuadee,
            rounds=5,
            temperature=0.95,
            max_new_tokens=512,
            top_p=0.7,
            top_k=50,
        )
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(
                [turn["role"] for turn in result["trajectory"]],
                ["persuader", "persuadee"] * 5,
            )
        second_round_prompt = persuader.batches[1][0]["user"]
        self.assertIn("Persuader case 1 round 1", second_round_prompt)
        self.assertIn("Persuadee case 1 round 1", second_round_prompt)
        self.assertNotIn("Persuader case 2 round 1", second_round_prompt)

    def test_batched_generation_records_batch_size_and_resumes(self) -> None:
        second = point()
        second["scenario_id"] = "scenario-2"
        second["source_id"] = "source-2"
        second["session_id"] = "session-2"
        with tempfile.TemporaryDirectory() as directory:
            persuader = BatchQueueEndpoint(["Offer the protected lane.", "Come with me."])
            persuadee = BatchQueueEndpoint(["I might.", "Maybe next week."])
            output = generate_method(
                points=[point(), second],
                method=METHODS[0],
                persuader=persuader,
                persuadee=persuadee,
                output_dir=Path(directory),
                persuadee_model="/models/qwen",
                rounds=1,
                temperature=0.95,
                max_new_tokens=128,
                force=False,
                context_batch_size=2,
                protocol="test_context_batch_v1",
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["context_batch_size"] for row in rows], [2, 2])
            self.assertEqual([len(row["trajectory"]) for row in rows], [2, 2])
            self.assertEqual([len(batch) for batch in persuader.batches], [2])
            generate_method(
                points=[point(), second],
                method=METHODS[0],
                persuader=BatchQueueEndpoint([]),
                persuadee=BatchQueueEndpoint([]),
                output_dir=Path(directory),
                persuadee_model="/models/qwen",
                rounds=1,
                temperature=0.95,
                max_new_tokens=128,
                force=False,
                context_batch_size=2,
                protocol="test_context_batch_v1",
            )

    def test_dialogue_methods_do_not_use_isr_prompt(self) -> None:
        persuader = QueueEndpoint(["A protected lane would make cycling much safer."])
        persuadee = QueueEndpoint(["I will consider it."])
        result = simulate_case(
            point=point(), method=METHODS[1], persuader=persuader, persuadee=persuadee,
            rounds=1, temperature=0.0, max_new_tokens=1024,
        )
        self.assertEqual(result["trajectory"][0]["prompt_type"], "dialogue")
        self.assertNotIn("Selected Strategy:", persuader.calls[0]["user"])

    def test_qwen_reasoning_delimiters_are_not_replayed(self) -> None:
        from repro.persuadee_simulation import clean_utterance

        self.assertEqual(
            clean_utterance("private plan</think> I am still unsure.", is_isr=False),
            "I am still unsure.",
        )

    def test_output_is_resumable_and_rejects_stale_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            persuader = QueueEndpoint(["Try the protected lane."])
            persuadee = QueueEndpoint(["Maybe."])
            output = generate_method(
                points=[point()], method=METHODS[0], persuader=persuader, persuadee=persuadee,
                output_dir=output_dir, persuadee_model="/models/qwen", rounds=1, temperature=0.0, max_new_tokens=1024,
                force=False,
            )
            self.assertEqual(len(persuader.calls), 1)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8").splitlines()[0])["protocol"], TEST_PROTOCOL)
            generate_method(
                points=[point()], method=METHODS[0], persuader=QueueEndpoint([]), persuadee=QueueEndpoint([]),
                output_dir=output_dir, persuadee_model="/models/qwen", rounds=1, temperature=0.0, max_new_tokens=1024,
                force=False,
            )
            output.write_text('{"protocol":"old"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Stale"):
                generate_method(
                    points=[point()], method=METHODS[0], persuader=QueueEndpoint([]), persuadee=QueueEndpoint([]),
                    output_dir=output_dir, persuadee_model="/models/qwen", rounds=1, temperature=0.0, max_new_tokens=1024,
                    force=False,
                )

    def test_same_local_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same device"):
            validate_devices("cuda:0", "cuda:0")
