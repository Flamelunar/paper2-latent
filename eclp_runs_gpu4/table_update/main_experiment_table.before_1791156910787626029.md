<!-- baseline-five-rounds:start -->
## Main experiment: five-round evaluation

This table uses Qwen2.5-7B-Instruct as the Persuader (`/app/hzj/models/Qwen2.5-7B-Instruct`) and the fixed merged Qwen3.5-9B Persuadee (`/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged`). All generation runs use `/app/hzj/anaconda3/envs/ljj/bin/python` and disable thinking/reasoning output. Separate tables are maintained for Meta-Llama-3.1-8B-Instruct and DeepSeek-R1-Distill-Qwen-14B.

Completed rows use the same 1,000 held-out contexts and five-round protocol. The Judge models use BF16. `-` means no complete result is available.

| Judge | Method | Persuader model | N | Strict acceptance within 5 rounds (%) | Win-Rate (vs golden) (%) | Mean rounds to success (failures = 5) | Success parse unresolved |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Selene-1-Mini-Llama-3.1-8B (BF16, strict V17 v7) | Baseline 1: Zero-shot | Qwen2.5-7B-Instruct | 1000/1000 | 66.80 (668/1000) | 67.90 | 3.469 | 0 |
|  | Baseline 2: Dialogue-SFT | qwen2.5-7b-instruct-dialogue-sft-paper10k | 1000/1000 | 62.50 (625/1000) | 65.55 | 3.554 | 0 |
|  | Baseline 3: ISR-SFT + SimDPO | - | - | - | - | - | - |
|  | Baseline 4: MA2P | Qwen2.5-7B-Instruct | 1000/1000 | 75.60 (756/1000) | 58.85 | 3.147 | 0 |
|  | Baseline 5: LatentRATT | - | - | - | - | - | - |
|  | Our | Qwen2.5-7B-Instruct | 1000/1000 | 70.80 (708/1000) | 61.00 | 2.883 | 0 |
| Gemma 4 E4B-it (BF16) | Baseline 1: Zero-shot | Qwen2.5-7B-Instruct | 969/1000 | 53.25 (516/969) | 47.42 | 3.990 | 31 |
|  | Baseline 2: Dialogue-SFT | qwen2.5-7b-instruct-dialogue-sft-paper10k | 939/1000 | 56.98 (535/939) | 48.08 | 3.962 | 61 |
|  | Baseline 3: ISR-SFT + SimDPO | Qwen2.5-7B-Instruct | 940/1000 | 49.68 (467/940) | 51.33 | 3.935 | 60 |
|  | Baseline 4: MA2P | Qwen2.5-7B-Instruct | 892/1000 | 54.48 (486/892) | 62.44 | 3.988 | 108 |
|  | Baseline 5: LatentRATT | rl | 941/1000 | 58.77 (553/941) | 48.99 | 3.829 | 59 |
|  | Our |  |  |  |  |  |  |
| Qwen3-14B (BF16) | Baseline 1: Zero-shot | Qwen2.5-7B-Instruct | 1000/1000 | 66.00 (660/1000) | 46.70 | 3.775 | 0 |
|  | Baseline 2: Dialogue-SFT | qwen2.5-7b-instruct-dialogue-sft-paper10k | 1000/1000 | 66.70 (667/1000) | 45.30 | 3.880 | 0 |
|  | Baseline 3: ISR-SFT + SimDPO | Qwen2.5-7B-Instruct | 1000/1000 | 60.90 (609/1000) | 42.10 | 3.739 | 0 |
|  | Baseline 4: MA2P | Qwen2.5-7B-Instruct | 1000/1000 | 73.60 (736/1000) | 49.00 | 3.584 | 0 |
|  | Baseline 5: LatentRATT | rl | 1000/1000 | 66.30 (663/1000) | 38.05 | 3.740 | 0 |
|  | Our |  |  |  |  |  |  |

Mean rounds uses the first successful round; a trajectory without success counts as round 5.
Win-Rate (vs golden) compares all five generated Persuader replies with the single held-out golden reply using two order-swapped Judge decisions; a split vote contributes 0.5.
A missing Win-Rate requires new five-round Judge evaluations; previous first-turn Win-Rate scores are not reused.
The original Selene-1 row uses the compact v17 prompt; the additional strict V17 v7 row uses explicit goal-acceptance criteria. Gemma 4 retains prompt v11, and Qwen3-14B uses the same v11 decision criteria with an example-free strict JSON output contract. All rows use the same Win-Rate comparison. Success rate, Win-Rate, and mean rounds use only parsed cases; N and unresolved show parsed and unresolved counts out of 1,000. Each case allows up to three additional generations after a parse or evidence validation error.
<!-- baseline-five-rounds:end -->
