import importlib.util
import unittest
from pathlib import Path
MODULE_PATH = Path(__file__).resolve().parents[1] / "exp_code_ljj/repro/public_answer.py"
spec = importlib.util.spec_from_file_location("public_answer_utils", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PublicUtilsTests(unittest.TestCase):
    def test_ma2p_private_json_is_removed_before_agent_forwarding(self):
        class Endpoint:
            def complete(self, **kwargs):
                return '<think>{"strategy":"PRIVATE"}</think>{"strategy":"PUBLIC"}'
        visible, filtered, retries = module.public_agent_answer(
            Endpoint(), system='JSON only', user='Public context', decode={})
        self.assertEqual(visible, '{"strategy":"PUBLIC"}')
        self.assertTrue(filtered)
        self.assertEqual(retries, 0)

    def test_ma2p_private_only_answer_retries_but_never_becomes_plan(self):
        class Endpoint:
            def __init__(self):
                self.calls = 0
            def complete(self, **kwargs):
                self.calls += 1
                return '<think>PRIVATE only'
        endpoint = Endpoint()
        with self.assertRaises(ValueError):
            module.public_agent_answer(endpoint, system='JSON only', user='Context', decode={})
        self.assertEqual(endpoint.calls, 4)

    def test_private_blocks_are_not_public_replies(self):
        self.assertEqual(module.strip_thinking("<think>private</think>Public"), ("Public", True))
        self.assertEqual(module.strip_thinking("private</think>Public"), ("Public", True))
        self.assertEqual(module.strip_thinking("<think>unfinished"), ("", True))

    def test_new_family_preferences_have_isolated_roots(self):
        data = Path("/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/data")
        for family in ("meta_llama31", "deepseek_r1_qwen14b"):
            model = "/app/hzj/liujianjian/persuader_atapter_save/" + family + "/baseline3_isr_sft"
            self.assertEqual(module.model_preference_directory(model, data),
                             data.parent / "artifacts/preference_simulations" / family)
        self.assertEqual(module.model_preference_directory("/app/hzj/models/Qwen2.5-7B-Instruct", data), data)


if __name__ == "__main__":
    unittest.main()
