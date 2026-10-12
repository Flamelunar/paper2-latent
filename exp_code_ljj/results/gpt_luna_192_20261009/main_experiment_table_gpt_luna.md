# GPT-6-luna API evaluation — fixed192

Latest shared v3 goal and Win@1 rubrics; public dialogue only. Five rounds, first accepted round; failures=5. Swapped preference credits averaged. Partial scores are provisional. Generation inputs retain their existing family-specific conditions.

| Model | Method | N | Success (%) | Win@1 (%) | Mean rounds (failure=5) | Unresolved success rounds |
|---|---|---:|---:|---:|---:|---:|
| deepseek14b | dialogue_sft | 192/192 | 47.40 | 86.20 | 4.33 | 0 |
| deepseek14b | ma2p_repaired | 192/192 | 38.02 | 81.38 | 4.41 | 0 |
| deepseek14b | persugpt | 192/192 | 47.40 | 87.63 | 4.22 | 0 |
| deepseek14b | zero_shot_repaired | 192/192 | 37.50 | 80.60 | 4.41 | 0 |
| llama31 | dialogue_sft | 192/192 | 31.05 | 48.18 | 4.64 | 2 |
| llama31 | latentratt | 192/192 | 34.38 | 47.66 | 4.53 | 0 |
| llama31 | ma2p_repaired | 192/192 | 25.65 | 52.73 | 4.66 | 1 |
| llama31 | our_direct | 192/192 | 38.54 | 64.32 | 4.32 | 1 |
| llama31 | persugpt | 192/192 | 39.79 | 91.41 | 4.29 | 1 |
| llama31 | zero_shot_repaired | 192/192 | 30.21 | 44.01 | 4.48 | 0 |
| qwen25 | dialogue_sft | 192/192 | 38.74 | 43.10 | 4.72 | 2 |
| qwen25 | latentratt | 192/192 | 35.94 | 42.45 | 4.55 | 0 |
| qwen25 | ma2p_repaired | 192/192 | 44.74 | 80.34 | 4.19 | 2 |
| qwen25 | our_direct | 192/192 | 39.06 | 56.77 | 4.54 | 0 |
| qwen25 | persugpt | 192/192 | 36.46 | 81.38 | 4.54 | 0 |
| qwen25 | zero_shot_repaired | 192/192 | 31.25 | 49.48 | 4.60 | 0 |
