"""Check complete JSON prefixes at schema length limits without loading weights."""
import json
from transformers import AutoTokenizer
from generate_ma2p_recovery import SafeSchemaFactory


def main():
    tokenizer = AutoTokenizer.from_pretrained("/app/hzj/models/Qwen2.5-7B-Instruct", local_files_only=True)
    factory = SafeSchemaFactory(tokenizer)
    schema = {"type": "object", "properties": {
        key: {"type": "string", "minLength": 1, "maxLength": 300}
        for key in ("preventive", "generative")
    }, "required": ["preventive", "generative"], "additionalProperties": False}
    cases = [
        {"preventive": "unknown", "generative": "Discuss observable benefits."},
        {"preventive": "a" * 300, "generative": "b" * 300},
        {"preventive": "Respect the person's concerns.", "generative": "Offer a small voluntary next step."},
    ]
    checked = 0
    token_text = {item[0]: item[1] for item in factory.data.regular_tokens}
    for value in cases:
        target = json.dumps(value)
        constraint = factory.constraint(schema, 0)
        prefix = []
        import torch
        while tokenizer.decode(prefix) != target:
            allowed = constraint(0, torch.tensor(prefix, dtype=torch.long))
            remaining = target[len(tokenizer.decode(prefix)):]
            candidates = [token for token in allowed if token in token_text
                          and remaining.startswith(token_text[token])]
            assert candidates, (target, tokenizer.decode(prefix))
            token = max(candidates, key=lambda item: len(token_text[item]))
            try:
                json.loads(tokenizer.decode(prefix))
                complete = True
            except json.JSONDecodeError:
                complete = False
            assert complete or tokenizer.eos_token_id not in allowed, tokenizer.decode(prefix)
            prefix.append(token)
            assert target.startswith(tokenizer.decode(prefix))
            checked += 1
        assert tokenizer.eos_token_id in constraint(0, torch.tensor(prefix))
        assert json.loads(tokenizer.decode(prefix)) == value
    assert all("\\" not in item[1] and all(char in "\t\r\n" or 32 <= ord(char) <= 126
               for char in item[1]) for item in factory.data.regular_tokens)
    print(json.dumps({"passed": True, "complete_json_cases": len(cases), "checked_prefixes": checked,
                      "unicode_byte_fragments_excluded": True, "premature_eos_excluded": True}))


if __name__ == "__main__":
    main()
