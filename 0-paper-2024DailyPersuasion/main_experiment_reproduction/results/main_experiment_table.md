<!-- qwen25-five-rounds:start -->
## Qwen2.5 five-round evaluation

Each row covers 1,000 held-out five-round trajectories. The Judge models use BF16.

| Judge | Method | N | Strict acceptance within 5 rounds (%) | Win-Rate (vs golden) (%) | Mean rounds to success (failures = 5) | Success parse unresolved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PersuGPT (BF16) | Qwen2.5-7B-Instruct zero-shot | 1000 | 0.00 (0/1000) | pending | 5.000 | 17 |
|  | Qwen2.5-7B-Instruct Dialogue-SFT | 1000 | 0.20 (2/1000) | pending | 4.998 | 21 |
| Qwen3.5-9B (BF16) | Qwen2.5-7B-Instruct zero-shot | 1000 | 33.60 (336/1000) | pending | 4.415 | 0 |
|  | Qwen2.5-7B-Instruct Dialogue-SFT | 1000 | 35.70 (357/1000) | pending | 4.333 | 0 |

Strict success requires the persuadee to explicitly and unconditionally accept the exact goal in any of five rounds. Unparseable success judgments count as failure.
Mean rounds uses the first successful round; a trajectory without success counts as round 5.
Win-Rate (vs golden) compares all five generated Persuader replies with the single held-out golden reply using two order-swapped Judge decisions; a split vote contributes 0.5.
Pending Win-Rate cells require new Judge evaluations; previous first-turn Win-Rate scores are not reused.
<!-- qwen25-five-rounds:end -->
