# Small-profile training and continuation

The active small profile has 5,000 SFT scenarios, 1,000 preference scenarios,
and 500 held-out test scenarios. Every selected SFT scenario retains exactly
one complete high-quality trajectory; training sees all assistant turns through
the multi-turn chat template. The active SFT routes use two epochs; DPO remains
one epoch when that later stage is enabled.

The original paper-sized files are never changed. Source subsets are stored in
`/home1/liujianjian/2-paper-CoLING/data-dailypersuasion/prepared/small5000/`,
and all generated data, adapters, merged models, logs, and results live below
`runs/small5000/`.

## Initial run

```bash
cd /home1/liujianjian/2-paper-CoLING/0-paper-2024DailyPersuasion/main_experiment_reproduction

python prepare_data.py --profile small5000
python 02_Llama31_Dialogue_SFT.py --profile small5000 --epochs 2
python 03_Llama31_ISR_SFT.py --profile small5000 --epochs 2
```

The default effective batch size is 16 (`batch_size=1`, gradient accumulation
16). The 5,000-session run therefore has about 313 optimizer updates and saves
resumable checkpoints every 100 updates. The final LoRA adapters are:

- `runs/small5000/outputs/01_llama31_dialogue_sft`
- `runs/small5000/outputs/02_llama31_isr_sft`

## Continue an adapter on more data or another epoch

Use `--adapter-checkpoint` to load a completed adapter as trainable. It starts a
fresh optimizer and scheduler, and requires a different `--output-dir` so the
source result stays intact. This is the correct option for an additional epoch
or a newly selected scenario subset.

```bash
# An additional continuation epoch after the two-epoch baseline (if needed).
python 02_Llama31_Dialogue_SFT.py --profile small5000 \
  --adapter-checkpoint runs/small5000/outputs/01_llama31_dialogue_sft \
  --output-dir runs/small5000/outputs/dialogue_sft_epoch2 \
  --epochs 1 --learning-rate 5e-5

# Expand ISR-SFT with a different prepared JSON/JSONL file of new scenarios.
python 03_Llama31_ISR_SFT.py --profile small5000 \
  --train-file /absolute/path/to/new_isr_sessions.jsonl \
  --adapter-checkpoint runs/small5000/outputs/02_llama31_isr_sft \
  --output-dir runs/small5000/outputs/isr_sft_expanded \
  --epochs 1 --learning-rate 5e-5
```

`--train-file` must use the same Alpaca fields as `runs/small5000/data/sft_add_isr.jsonl`
or `sft_dialogue.jsonl`; it bypasses automatic preparation. `--max-samples` is
only for smoke tests because it limits rows rather than unique scenarios.

## Resume an interrupted run exactly

Use `--resume-from-checkpoint` only with unchanged data, seed, batch settings,
and the original output root. It restores the optimizer, scheduler, RNG, and
step state; it is not a way to switch datasets.

```bash
python 02_Llama31_Dialogue_SFT.py --profile small5000 --epochs 1 \
  --resume-from-checkpoint \
  runs/small5000/outputs/01_llama31_dialogue_sft/checkpoint-200
```

Do not resume the old paper-scale `outputs/01_llama31_dialogue_sft/checkpoint-500`
for this experiment: it was trained with a different 10,000-scenario data
ordering and is not a checkpoint of the controlled small profile.

When an expanded ISR adapter is selected for the final method, use that exact
adapter for the subsequent simulation and DPO run. Never train, simulate, or
preference-optimize on the 500 final test scenarios.
