# Gemma 4 evaluation archive

Judge: `/data1/liujianjian/model/gemma-4-E4B-it`, BF16 without quantization. The canonical 1,000-row results use GPU 1, prompt v11 (`joint_v2_r11`), and the shared held-out generations in `../../generation_5rounds/`.

## Contents

- `qwen25_zero_shot.jsonl`, `qwen25_dialogue_sft.jsonl`: canonical full results, one GPU 1 record for each index 0–999.
- `summary.json`: independently recomputed metrics and denominators.
- `run_metadata.json`, `canonicalization_report.json`: settings, code hashes, and source counts used to select the GPU 1 rows.
- `raw/`: original pre-canonicalization outputs, including records from other devices.
- `partial_runs/gpu2_dialogue_sft/`: interrupted GPU 2 run, explicitly excluded from the main table.
- `prompt_diagnostics/`: small-sample prompt revisions and quickcheck manifests; these are diagnostic data, not full-test estimates.
- `logs/full_evaluation/`: full-run, retry, scheduler, and interruption logs.
- `code/current/`: current evaluation and score-processing code; `code/historical_snapshots/` preserves run-time script snapshots.

## Metric rules

Success rate, Win-Rate, and mean rounds use parsed records only. Win-Rate compares all five Persuader replies with the held-out golden reply in both orders; a split vote gets half credit. Mean rounds uses the first successful round, with failure capped at round 5. Unresolved parse failures are reported separately out of 1,000. See `summary.json` and `../../../results/main_experiment_table.md`.
