# Archived Qwen3-14B v1 judgments

These 5,000 records used prompt revision `qwen3_gemma_v11_strict_json_v1`.
Its JSON example fixed `win_ab` to `A` and `win_ba` to `B`, which biased the
model's answers. On the same 20 cases, the original results favored the
generated reply 18 times, removing the example did so 9 times, and reversing
the example did so 0 times. Some success judgments also changed.

These records are retained for provenance and excluded from the main table.
The replacement evaluation uses an example-free output contract.
