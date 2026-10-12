# Gemma 4 joint success and Win-Rate prompt diagnostic

Judge: `/data1/liujianjian/model/gemma-4-E4B-it`, BF16 on GPU 1. The final diagnostic uses `evaluate_joint_prompt_probe.py --prompt-version v11 --max-parse-retries 3 --max-tokens 384`. Each attempt requests five success decisions and both order-swapped Win-Rate decisions together. A parsing or evidence failure allows up to three **additional** generations. The complete outputs, errors, and call counts are in `artifacts/success_judgments/gemma4/prompt_diagnostics/revisions/gemma4_joint_v2_r11.jsonl`.

The manifest in `artifacts/success_judgments/gemma4/prompt_diagnostics/shared_case_control_manifest.json` deliberately selects cases from old-Judge agreement and disagreement groups. Trial 2 repeats trial 1; trial 3 uses different cases. These are diagnostic samples, **not a random sample of the 1,000 held-out cases**. The rates below are not full-test estimates.

| Trial | Generation method | Parsed / attempted | Success among parsed | Win-Rate (vs golden) among parsed | Mean rounds (failures = 5) among parsed | Unresolved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Zero-shot | 9 / 10 | 44.44% (4/9) | 50.00% | 4.111 | 1 |
| 1 | Dialogue-SFT | 9 / 10 | 55.56% (5/9) | 55.56% | 3.444 | 1 |
| 2, repeat of trial 1 | Zero-shot | 9 / 10 | 44.44% (4/9) | 50.00% | 4.111 | 1 |
| 2, repeat of trial 1 | Dialogue-SFT | 9 / 10 | 55.56% (5/9) | 55.56% | 3.444 | 1 |
| 3, new cases | Zero-shot | 8 / 10 | 75.00% (6/8) | 50.00% | 4.250 | 2 |
| 3, new cases | Dialogue-SFT | 10 / 10 | 60.00% (6/10) | 60.00% | 4.100 | 0 |

Success uses the first successful round; failures count as round 5 for the mean. Win-Rate gives one point when generated text wins both swapped comparisons, zero when the golden reply wins both, and half a point when the two comparisons disagree. Unresolved cases are excluded from both denominators. Across all 60 calls, 54 parsed; 8 cases needed retries, 2 recovered, and 6 remained unresolved. All 18 cases parsed in both repeated trials had identical success arrays and Win-Rate credits.

## Why the existing Judges disagree

The main table's original v1 success prompt ends with a prefilled `success: false` JSON example and demands unconditional acceptance. It is unchanged for the archived 1,000-case scores: Meta-Llama-3.1-8B-Instruct gives 2/1,000 for Zero-shot and 1/1,000 for Dialogue-SFT; Qwen3.5-9B gives 336/1,000 and 357/1,000. On Dialogue-SFT index 29, the Persuadee says “I'll make a donation”; original Llama v1 says false, while Qwen3.5 and Gemma 4 say success in round 3. Zero-shot index 19 similarly contains “Let's do it” about trying Capoeira and is accepted by Qwen3.5 and Gemma 4 but rejected by Llama v1. The separate Llama v2 pilot changed 8 of 10 selected cases to success, including these clear examples; this demonstrates prompt sensitivity, not a validated full-test replacement score.

Gemma 4 also makes substantive mistakes. The v9/v11 prompt correctly rejects merely touring a Montessori school when the goal is choosing it (Zero-shot index 471) and accepts “I'll donate” to the local nonprofit (Dialogue-SFT index 524). It still labels Zero-shot index 71 successful after a commitment to schedule a health check-up and **research** bicycles, although the goal also requires using a bicycle. For Zero-shot index 117, it first marks success in round 3 even though round 2 already says “I'm willing to give it a try. Let's start planning our European backpacking adventure!” These examples mean parse validity and repeat stability alone do not establish Judge accuracy.

The previous prompt revisions and outputs are retained under `artifacts/success_judgments/gemma4/prompt_diagnostics/revisions/`. Revision r7 parsed 55/60 calls without retries but missed several explicit commitments; r8 introduced a false positive for attending a park information meeting; r9 improved those selected cases but parsed 52/60 without retries. Revision r11 keeps r9's task prompt and records up to three parsing retries. The full 1,000-case Gemma 4 results are now included in the main experiment rows and archived under `artifacts/success_judgments/gemma4/`; the selected-sample figures in this diagnostic remain prompt-development results, not full-test estimates.
