from __future__ import annotations

import sys
import unittest
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from repro.trajectory_evaluation import parse_winner


class TrajectoryEvaluationTests(unittest.TestCase):
    def test_parse_strict_json_winner(self) -> None:
        self.assertEqual(parse_winner('{"winner":"b","reason":"more relevant"}'), "B")

    def test_parse_local_model_json_wrapper(self) -> None:
        raw = "<think>Response A addresses the concern.</think>\n{" \
            '"winner":"A","reason":"addresses the concern"}'
        self.assertEqual(parse_winner(raw), "A")

    def test_parse_labelled_fallback(self) -> None:
        self.assertEqual(parse_winner("Winner: B"), "B")

