"""Build a final-answer chat prefix without relying on ignored template flags."""


def direct_answer_ids(tokenizer, messages, *, model_hint=None):
    hint = str(model_hint or getattr(tokenizer, "name_or_path", "")).lower()
    is_r1 = any(marker in hint for marker in ("deepseek-r1", "deepseek_r1", "deepseekr1"))
    if is_r1:
        messages = [dict(message) for message in messages]
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += (
                "\nAnswer directly with only the final answer. Do not write analysis,"
                " reasoning, or <think> tags."
            )
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        # R1's released template ignores enable_thinking=False and opens a
        # thinking block in its generation-prompt branch. Close an empty block
        # explicitly, as in the Baseline 1--4 generation endpoint.
        rendered += "<｜Assistant｜><think>\n\n</think>\n\n"
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    else:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        )
    return ids.input_ids if hasattr(ids, "input_ids") else ids
