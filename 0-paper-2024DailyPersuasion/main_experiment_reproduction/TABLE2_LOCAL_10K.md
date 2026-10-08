# Isolated local-model Table 2 run

This run lives under `runs/paper10k/` and leaves the existing `small5000`, root-level paper data, and earlier `/data1` model outputs intact. The paper reports 10,000 SFT scenarios, 2,000 preference scenarios, 1,000 test scenarios, and one epoch for SFT and preference optimization. The exact split IDs, random seed, optimizer, batch size, LoRA settings, and DPO simulation depth are not disclosed; this project uses its seeded split and records its implementation choices.

The full-size Alpaca JSONL is built from the prepared paper split. Each source session remains one row with all previous dialogue turns in `history`; ISR rows target the intent reasoning, selected strategy, and utterance. Assistant-only loss supervises the persuader turns while user turns remain input-only, matching the paper's Table 8 description. A full rendering scan confirmed that all 59,868 Dialogue-SFT and 59,868 ISR-SFT rows fit within 2,048 tokens (ISR maximum: 2,045), retain target tokens, and contain all three ISR fields.

## Run

`run_table2_paper10k.sh` runs the pipeline in order: data validation, one-epoch Dialogue-SFT, one-epoch ISR-SFT, 2,000 local simulation preferences, one-epoch DPO, four-method generation on the 1,000 held-out contexts, and local Qwen evaluation. It defaults to physical GPUs 2 and 1, mapped to logical `cuda:0` and `cuda:1`. Before doing any work it requires at least 64 GiB free on the first visible GPU and 20 GiB free on the second; it exits without touching active processes if that capacity is unavailable. Completed SFT stages are skipped, and interrupted stages resume from their newest complete Trainer checkpoint.

`run_paper10k_sft_3090.sh` can run the two BF16 SFT stages on one RTX 3090 with at least 20 GiB free while the A100 is occupied. It writes to the same isolated `paper10k` outputs; after the A100 is available, the full pipeline launcher skips those completed stages and continues with preference simulation, DPO, and evaluation.

```bash
bash run_table2_paper10k.sh
```

For the SFT-only 3090 route, run `bash run_paper10k_sft_3090.sh`.

Override device placement with `RUN_GPUS`, `PERSUADER_DEVICE`, `PERSUADEE_DEVICE`, and `JUDGE_DEVICE` only after mapping visible GPUs and checking free memory. The run reuses the existing Qwen3.5 persuadee checkpoint, which was trained on the earlier 5,000-scenario subset for two epochs; this is recorded as a model-side deviation from the paper's LLaMA-2 13B user model trained on the 10,000-scenario split.

Training adapters, DPO preferences, logs, model generations, local-judge decisions, and the final Markdown/JSON table are stored under `runs/paper10k/`. The report compares the four Table 2 rows with the published LLaMA-2 values but labels all measurements as the Llama-3.1/Qwen local adaptation. Human ratings are omitted because they require annotators.

`outputs/smoke/` and `outputs/smoke_isr/` contain one-row, one-step BF16 LoRA checks of the Dialogue-SFT and ISR-SFT training paths. They are execution checks only and are not used by the Table 2 pipeline.
