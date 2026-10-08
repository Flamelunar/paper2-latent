# Qwen2.5 Dialogue-SFT Table 2-style run

This route trains only Dialogue-SFT and compares the local Qwen2.5-7B-Instruct base with its Dialogue-SFT LoRA adapter. Training and every local model inference load in BF16; the runner uses no quantization.

## Model roles

- Zero-shot Persuader: `/data1/liujianjian/model/Qwen2.5-7B-Instruct`.
- Dialogue-SFT Persuader: `/data1/liujianjian/Persuader/qwen2.5-7b-instruct-dialogue-sft-paper10k`.
- Persuadee: `/data1/liujianjian/Persuadee/Qwen3.5-9B-persuadee/merged`.
- Automatic Judge (default): `/data1/liujianjian/PersuGPT`.

## Data and training

The run uses `runs/paper10k/data/dialogue_sft.jsonl` (59,868 assistant replies) and `runs/paper10k/data/evaluation_points.jsonl` (1,000 held-out contexts). The launch script checks their manifest hashes and rejects any scenario overlap. It saves the adapter and resumable checkpoints under `/data1/liujianjian/Persuader/`.

The fixed training setup is one epoch, assistant-only loss, BF16, LoRA rank 16/alpha 32/dropout 0.05, max length 2,048, batch size 1, gradient accumulation 16, learning rate 1e-4, and checkpoint interval 100 steps (3,742 total optimizer steps). The paper states one fine-tuning epoch but does not disclose these LoRA and optimizer settings; they are recorded as reproduction defaults.

```bash
./run_qwen25_dialogue_sft_paper10k.sh
```

The launcher runs no ISR-SFT, DPO, or persuadee training. If interrupted, it resumes from the latest complete Trainer checkpoint.

## Inference and scoring

The released PersuGPT generation defaults are used for both Persuader variants: sampling enabled, temperature 0.95, top-p 0.7, top-k 50, and at most 512 new tokens. Every held-out context is simulated for five Persuader-Persuadee rounds. The paper does not define a maximum round count; five is the selected reproduction setting, close to the dataset's 5.08-round mean.

To reduce full-run latency, the local runner decodes up to two independent test contexts together. Each context still follows its own Persuader-then-Persuadee turn order with the same prompt and sampling parameters. The Judge batches up to four independent prompts per decode; the five prefix decisions and both order-swapped pairwise judgments remain separate prompts. Batch sizes are recorded in the run configuration and result rows.

Run a one-context, five-round smoke trajectory and automatic Judge check first:

```bash
python run_qwen25_table2_generation.py --profile paper10k --limit 1 \
  --output-dir runs/qwen25_dialogue_sft/generation_smoke
python evaluate_qwen25_table2.py --profile paper10k --limit 1 \
  --generation-dir runs/qwen25_dialogue_sft/generation_smoke \
  --evaluation-dir runs/qwen25_dialogue_sft/evaluation_smoke
```

After the smoke check succeeds, generate and score all 1,000 test contexts:

```bash
python run_qwen25_table2_generation.py --profile paper10k
python evaluate_qwen25_table2.py --profile paper10k
```

To wait for an already-running SFT process and run the smoke check before the full evaluation automatically, use:

```bash
./run_qwen25_table2_after_sft.sh <training-pid>
```

The evaluator compares all five generated Persuader replies as one candidate against the test set's single golden reply. The Judge compares both response orders; opposing decisions contribute 0.5 to Win-Rate (vs golden). It also judges strict goal acceptance after every dialogue prefix. The main table lists strict acceptance within five rounds, Win-Rate (vs golden), mean rounds to first success (failures count as round 5), and unresolved success parses. Strict success requires an explicit, unconditional acceptance, commitment, or action toward the stated goal. Generation always completes all five rounds; judging success does not stop a trajectory early.

Table 2 used GPT-4+ISR reference replies and ChatGPT as Judge. This route uses the test set's original target replies and defaults to the local PersuGPT Judge; the five-reply comparison also differs from the paper's single-reply comparison. Human evaluation is omitted.

Raw trajectories, per-prefix Judge outputs, resolved run settings, and aggregate scores are stored below `runs/qwen25_dialogue_sft/`.
