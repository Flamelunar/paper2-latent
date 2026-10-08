"""Resume MA2P with ASCII-safe JSON constraints and an explicit recovery record."""
import sys
from pathlib import Path

import generate_ma2p_core as generate


class SafeSchemaFactory(generate.SchemaFactory):
    """Exclude byte fragments and Unicode escapes from private JSON analysis."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        from lmformatenforcer import TokenEnforcerTokenizerData

        regular = [
            token for token in self.data.regular_tokens
            if token[1] and "\\" not in token[1]
            and all(character in "\t\r\n" or 32 <= ord(character) <= 126
                    for character in token[1])
        ]
        self.data = TokenEnforcerTokenizerData(
            regular, self.data.decoder, self.data.eos_token_id,
            self.data.use_bitmask, self.data.vocab_size,
        )


def main():
    def argument(name):
        return sys.argv[sys.argv.index(name) + 1]

    if argument("--method") != "ma2p_repaired":
        raise ValueError("This recovery is restricted to MA2P")
    root = Path(argument("--repo-root")).resolve()
    output = Path(argument("--output")).resolve()
    sys.path.insert(0, str(root / "my_eclp_final"))
    from experiment_io import digest, immutable_json, object_hash, read_json

    frozen = output / "generation_run.json"
    if not frozen.exists():
        raise ValueError("Recovery requires the original immutable generation contract")
    record = {
        "revision": "ma2p_ascii_json_recovery_v1",
        "original_contract_sha256": digest(frozen),
        "original_run_hash": object_hash(read_json(frozen)),
        "recovery_source_sha256": digest(Path(__file__)),
        "recovery_core_sha256": digest(Path(generate.__file__)),
        "experiment_io_sha256": digest(root / "my_eclp_final/experiment_io.py"),
        "fix": "Private analysis uses ASCII tokens without backslashes, avoiding invalid Unicode parser transitions and escaped surrogates",
        "scope": "MA2P Perception and World Model JSON only; strict JSON checks and schema limits remain active",
        "existing_records": "Retained verbatim with their original run_hash",
    }
    immutable_json(output / "generation_recovery_revision.json", record)
    original = generate.base_turn

    def marked_turn(*args, **kwargs):
        result = yield from original(*args, **kwargs)
        result["generation_recovery_revision"] = record["revision"]
        result["generation_recovery_source_sha256"] = record["recovery_source_sha256"]
        return result

    generate.SchemaFactory = SafeSchemaFactory
    generate.base_turn = marked_turn
    print("MA2P recovery:", record["revision"], record["recovery_source_sha256"], flush=True)
    generate.main()


if __name__ == "__main__":
    main()
