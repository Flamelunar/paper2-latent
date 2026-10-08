"""Tests for evaluation boundaries that could silently change the metric."""
import json
import sys
import unittest
from pathlib import Path

test_root = Path(__file__).resolve().parents[1]
code_dir = test_root / "remote_patch"
if not code_dir.is_dir():
    code_dir = test_root / "my_eclp_final"
sys.path.insert(0, str(code_dir))
from matched_eval_protocol_v2 import TokenTrie, order_credit, parse_verdict, success_prompt, win_prompt
from prepare_matched_golden_v2 import recover, scenario_id
from evaluate_matched_v2 import summarize
from update_matched_table_v2 import START, END, render_section, replace_section


def item(goal="join the club", length=15):
    return {"id": "released-id-can-repeat", "scenario": {
        "goal": goal, "background": "public background", "tag": "tag",
        "persuader": "speaker", "persuadee": "listener", "domain": "social"},
        "dialog": {"session_1": [{"role": "persuader" if i % 2 == 0 else "persuadee",
                                   "response": f"original_turn_{i}"} for i in range(length)]}}


def point(source, start):
    turns = source["dialog"]["session_1"]
    return {"scenario_id": scenario_id(source), "scenario": source["scenario"], "source_id": source["id"],
            "session_id": "session_1", "target_turn": start // 2 + 1,
            "history": [{"role": t["role"], "content": t["response"]} for t in turns[:start]],
            "reference": turns[start]["response"]}


def generated(p):
    return {"scenario_id": p["scenario_id"], "scenario": p["scenario"], "initial_history": p["history"],
            "reference": p["reference"], "trajectory": [
                {"role": "persuader" if i % 2 == 0 else "persuadee", "content": f"NEW_TURN_{i}"}
                for i in range(10)]}


class StrictOutputTests(unittest.TestCase):
    def test_success_is_never_coerced(self):
        for raw in ('{"success":"true"}', '{"success":1}', '{"success":null}',
                    '{"success":true,"reason":"ok"}', '{"success":true,"success":false}',
                    '```json\n{"success":true}\n```', '{"success":true} trailing', 'true'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_verdict(raw, "success")
        self.assertIs(parse_verdict('{"success":false}', "success")["success"], False)
        self.assertIs(parse_verdict('{"success":true}', "success")["success"], True)

    def test_win_accepts_only_anonymous_labels(self):
        for winner in ("A", "B", "Tie"):
            self.assertEqual(parse_verdict(json.dumps({"winner": winner}), "win")["winner"], winner)
        for value in (True, 1, [], {}, "tie", "Golden", "AB"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_verdict(json.dumps({"winner": value}), "win")

    def test_grammar_permits_only_legal_paths_and_eos(self):
        trie = TokenTrie([[1, 2, 3], [1, 2, 4]], 9)
        self.assertEqual(trie.allowed([]), [1])
        self.assertCountEqual(trie.allowed([1, 2]), [3, 4])
        self.assertEqual(trie.allowed([1, 2, 3]), [9])
        with self.assertRaises(ValueError):
            trie.allowed([1, 2, 8])

    def test_order_reversal_and_ties_have_correct_credit(self):
        self.assertEqual(order_credit("A", "ab"), 1)
        self.assertEqual(order_credit("B", "ba"), 1)
        self.assertEqual(order_credit("A", "ba"), 0)
        self.assertEqual(order_credit("Tie", "ab"), 0.5)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.source = item()
        self.p = point(self.source, 2)
        self.gold = recover([self.p], [self.source])[0][0]
        self.row = generated(self.p)

    def test_success_prompt_has_no_future_or_reference(self):
        prompt = success_prompt(self.row, 2)
        self.assertIn("NEW_TURN_3", prompt)
        self.assertNotIn("NEW_TURN_4", prompt)
        self.assertNotIn(self.p["reference"], prompt)
        self.assertIn("original_turn_1", prompt)

    def test_win1_excludes_later_dialogue_and_swaps_candidates(self):
        ab = win_prompt(self.row, self.gold, "win1", "ab")
        ba = win_prompt(self.row, self.gold, "win1", "ba")
        self.assertIn("Candidate A:\nNEW_TURN_0", ab)
        self.assertIn("Candidate B:\nNEW_TURN_0", ba)
        self.assertNotIn("NEW_TURN_1", ab)
        self.assertNotIn("original_turn_3", ab)
        self.assertNotIn("win_ab", ab)
        self.assertNotIn("golden", ab.lower())

    def test_win_full_keeps_both_roles_and_all_golden_turns(self):
        prompt = win_prompt(self.row, self.gold, "win_full", "ab")
        self.assertIn("Persuadee: NEW_TURN_9", prompt)
        self.assertIn("Persuadee: original_turn_13", prompt)
        self.assertIn("Persuader: original_turn_14", prompt)
        self.assertEqual(len(self.gold["golden_trajectory"]), 13)


class RecoveryTests(unittest.TestCase):
    def test_short_golden_is_retained_without_padding(self):
        source = item(length=9)
        p = point(source, 8)
        rows, manifest = recover([p], [source])
        self.assertEqual(len(rows[0]["golden_trajectory"]), 1)
        self.assertEqual(rows[0]["golden_remaining_complete_rounds"], 0)
        self.assertFalse(rows[0]["golden_ends_with_persuadee"])
        self.assertEqual(manifest["win_full_eligible"], 1)
        self.assertFalse(manifest["padding_or_synthetic_golden"])

    def test_nonunique_source_id_does_not_match_wrong_scenario(self):
        a, b = item("join club A"), item("join club B")
        self.assertEqual(a["id"], b["id"])
        rows, _ = recover([point(b, 4)], [a, b])
        self.assertEqual(rows[0]["scenario"]["goal"], "join club B")

    def test_changed_reference_or_history_is_rejected(self):
        source = item()
        for field in ("reference", "history"):
            p = point(source, 2)
            if field == "reference":
                p[field] = "different reference"
            else:
                p[field][0]["content"] = "different history"
            with self.subTest(field=field), self.assertRaises(ValueError):
                recover([p], [source])

    def test_summary_distinguishes_retraction_and_order_disagreement(self):
        source = item(length=9)
        gold = recover([point(source, 0)], [source])[0][0]
        row = {"index": 0, "round_success": [False, True, False, False, False],
               "win1": {"ab": "A", "ba": "A", "credit": 0.5},
               "win_full": {"ab": "Tie", "ba": "Tie", "credit": 0.5}}
        summary = summarize([row], {0: gold}, {"judge": "test", "judge_model": "test"})
        self.assertEqual(summary["success_within5_percent"], 100)
        self.assertEqual(summary["final_round5_success_percent"], 0)
        self.assertEqual(summary["win1_order_disagreement_count"], 1)
        self.assertEqual(summary["win1_explicit_tie_judgments"], 0)
        self.assertEqual(summary["win_full_explicit_tie_judgments"], 2)


class TableTests(unittest.TestCase):
    def test_new_table_preserves_legacy_and_never_copies_old_scores(self):
        rows = []
        methods = ("Baseline 1: Zero-shot", "Baseline 2: Dialogue-SFT", "Baseline 3: ISR-SFT + SimDPO",
                   "Baseline 4: MA2P", "Baseline 5: LatentRATT", "Our")
        for judge in ("Selene-1-Mini-Llama-3.1-8B (BF16, strict V17 v7)", "Gemma 4 E4B-it (BF16)", "Qwen3-14B (BF16)"):
            for i, method in enumerate(methods):
                rows.append(f"| {judge if i == 0 else ''} | {method} | Qwen2.5-7B-Instruct | 1000/1000 | 66.80 | 67.90 | 3.469 | 0 |")
        original = "<!-- baseline-five-rounds:start -->\n" + "\n".join(rows) + "\n<!-- baseline-five-rounds:end -->\n\n"
        section = render_section(original, Path("/unused"), template_only=True)
        updated = replace_section(original, section)
        self.assertTrue(updated.startswith(original))
        self.assertEqual(updated.count(START), 1)
        self.assertEqual(updated.count(END), 1)
        self.assertEqual(section.count("待评估 / 1000"), 12)
        self.assertNotIn("Gemma", section)
        self.assertNotIn("67.90", section)
        self.assertNotIn("66.80", section)
        self.assertEqual(section.count("| Our |"), 2)
        self.assertIn("| Win@1 (%) | Win-Full (%) |", section)
        self.assertIn('`{"success":true}`', section)
        self.assertEqual(replace_section(updated, section), updated)


if __name__ == "__main__":
    unittest.main()
