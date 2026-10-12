# Qwen2.5 Dialogue-SFT: Table 2-style automatic evaluation

Held-out contexts: 1000. Persuadee: local Qwen3.5-9B fine-tuned model. Judge: `/data1/liujianjian/PersuGPT` on `cuda:2` using BF16 (2 replicas).

| Persuader | Win-Rate (%) | Character ROUGE-L | Strict success by round 5 (%) | Mean first-success round (successes) | Mean rounds to success or cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct zero-shot | 51.05 | 42.52 | 0.00 | n/a | 5.00 |
| Qwen2.5-7B-Instruct Dialogue-SFT | 47.10 | 47.40 | 0.20 | 4.00 | 5.00 |

## Protocol notes

- The BF16 Judge runs independent prompt instances in batches of up to 4; each prefix and each order-swapped comparison remains a separate prompt.
- 2 Judge model replica(s) score the zero-shot and Dialogue-SFT sets.
- Win-Rate compares each model's first Persuader reply with the test set's original target reply; the BF16 local Judge decides twice with response order swapped, and a split vote contributes 0.5.
- An unparseable pairwise verdict after two clarifications contributes neutral 0.5 credit for that comparison and is flagged as `pairwise_unresolved` in the JSONL.
- Character ROUGE-L is computed on that same first reply and target reply.
- Strict success is an added automatic measure: every one of the five dialogue prefixes is judged independently; success requires explicit, unconditional acceptance, commitment, or action toward the stated goal.
- When the Judge gives no parseable verdict after two clarifications, that prefix is conservatively scored unsuccessful and flagged as `parse_unresolved` in the JSONL; the JSON report counts these cases.
- Generation runs all five rounds even after a prefix is judged successful. Mean generated rounds therefore remains 5; the capped mean counts first success round for successful cases and 5 for unsuccessful cases.
- Dialogue-SFT uses one BF16 epoch with assistant-only loss. The paper does not specify LoRA, optimizer, or batch hyperparameters; this run uses the documented reproduction defaults: LoRA rank 16, alpha 32, dropout 0.05, max length 2048, batch size 1 with gradient accumulation 16, and learning rate 1e-4.
- The paper's Table 2 uses GPT-4+ISR reference replies and ChatGPT as Judge. Here, test-set target replies and a local Qwen3.5-9B Persuadee replace the reference setup, while `PersuGPT` replaces ChatGPT as Judge; scores are not directly comparable to the published values.
- Human evaluation is omitted.

## Strict success by round

| Persuader | Round 1 (%) | Round 2 (%) | Round 3 (%) | Round 4 (%) | Round 5 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct zero-shot | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Qwen2.5-7B-Instruct Dialogue-SFT | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 |
