import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "baseline5_latentRATT/direct_answer_prompt.py"
spec = importlib.util.spec_from_file_location("direct_answer_prompt", MODULE_PATH)
prompt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt)


class IgnoringR1Tokenizer:
    name_or_path = "/app/hzj/models/DeepSeek-R1-Distill-Qwen-14B"

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "USER" + ("<｜Assistant｜><think>" if kwargs["add_generation_prompt"] else "")

    def __call__(self, rendered, **kwargs):
        self.rendered = rendered
        return {"input_ids": [1, 2, 3]}


class DirectAnswerPromptTests(unittest.TestCase):
    def test_r1_template_ignoring_flag_cannot_open_thinking(self):
        tokenizer = IgnoringR1Tokenizer()
        messages = [{"role": "system", "content": "System"}, {"role": "user", "content": "Goal"}]
        self.assertEqual(prompt.direct_answer_ids(tokenizer, messages), [1, 2, 3])
        self.assertFalse(tokenizer.template_kwargs["add_generation_prompt"])
        self.assertTrue(tokenizer.rendered.endswith("<think>\n\n</think>\n\n"))
        self.assertEqual(messages[0]["content"], "System")

    def test_llama_explicitly_disables_thinking_flag(self):
        class LlamaTokenizer:
            name_or_path = "Meta-Llama-3.1-8B-Instruct"

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return [4, 5]

        tokenizer = LlamaTokenizer()
        self.assertEqual(prompt.direct_answer_ids(tokenizer, []), [4, 5])
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)


if __name__ == "__main__":
    unittest.main()
