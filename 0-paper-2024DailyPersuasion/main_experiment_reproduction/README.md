# DailyPersuasion main-experiment adaptation

This directory is an isolated implementation of the paper's three main
training routes. It keeps the paper's data protocol and automatic metrics, but
uses the requested local `Meta-Llama-3.1-8B-Instruct` checkpoint instead of the
gated LLaMA-2-13B checkpoint. Results are therefore an adaptation and must not
be reported as an exact reproduction of the paper's LLaMA-2 numbers.

For a fresh full-scale four-row Table 2 run, use the isolated `paper10k`
profile and follow [`TABLE2_LOCAL_10K.md`](TABLE2_LOCAL_10K.md). It writes all
new data, models, logs, and results under `runs/paper10k/` without replacing
the existing `small5000` run or root-level paper data.

The entry points are deliberately explicit:

1. `01_Llama31_ZeroShot.py` - test only the untuned local Llama 3.1 control
   against the fixed learned Qwen persuadee.
2. `02_Llama31_Dialogue_SFT.py` - dialogue-only SFT.
3. `03_Llama31_ISR_SFT.py` - intent-to-strategy-reasoning (ISR) SFT.
4. `04_Llama31_Ours_ISR_SimDPO_DPO.py` - ISR-SFT, simulated preferences, and
   DPO. It supports both OpenAI-compatible APIs and local Transformer models
   for the simulated persuadee and judge.
5. `tests/05_Test_Persuader_With_Qwen_Persuadee.py` - the shared multi-turn test
   runner for methods 01--03, plus method 04 in the `paper10k` profile.

Only the five canonical filenames above are kept; old duplicate compatibility
symlinks have been removed so there is one unambiguous entry point per method.

`evaluate_main_table.py` is retained only as the older, one-utterance
reference-response evaluator. It does not use the learned Qwen persuadee and
is not the entry point for the fixed-persuadee tests described below.

## Existing reduced protocol

- The pre-existing reduced protocol uses `--profile small5000`. It uses 5,000 quality-
  selected SFT scenarios, 1,000 preference scenarios, and 500 shared held-out
  test scenarios. Use the separate `paper10k` profile for new full-scale runs.
- Exactly one complete high-quality trajectory is retained per SFT scenario.
  The quality manifest records deterministic checks for completion, initial
  resistance, strategy adaptation, positive movement, and ISR-field coverage.
- The preference and test samples are deterministic scenario samples from their
  own original paper-sized partitions, so no selected scenario can cross splits.
- The small5000 SFT routes run for two epochs; the full-scale paper routes run
  for one epoch and DPO remains one epoch.
  one epoch unless explicitly changed.
- Training context cutoff: 2,048 tokens.
- Generated/evaluation output limit: 1,024 tokens everywhere.
- Default adaptation model:
  `/data1/liujianjian/model/Meta-Llama-3.1-8B-Instruct`.
- The paper's Human column is not produced automatically because it requires
  human annotation. Paper Table 2 values are shown only as labeled LLaMA-2
  reference values beside the new measurements.

The generated SFT files contain one complete multi-turn session per Alpaca row,
matching the released LLaMA-Factory-style loader. Earlier turns are in
`history`; they are not duplicated into separate prefix rows.

Each completed small-profile route keeps its lightweight LoRA adapter in
`runs/small5000/outputs/01_llama31_dialogue_sft`,
`runs/small5000/outputs/02_llama31_isr_sft`, or
`runs/small5000/outputs/03_llama31_isr_sim_dpo`. Run
`python save_merged_models.py --profile small5000` after training to also export
standalone checkpoints with the `_merged` suffix. Adapters are retained for
reproducibility; merged directories can be loaded directly with
`AutoModelForCausalLM.from_pretrained(...)`. To export manually after a training
stage has finished, use the profile-specific command above.

## Prepare and validate data

```bash
cd /home1/liujianjian/2-paper-CoLING/0-paper-2024DailyPersuasion/main_experiment_reproduction
python prepare_data.py --profile small5000
```

Expected files:

| File | Rows | Meaning |
| --- | ---: | --- |
| `runs/small5000/data/sft_dialogue.jsonl` | 5,000 | persuader Dialogue-SFT: one full dialogue-only session per selected scenario |
| `runs/small5000/data/sft_add_isr.jsonl` | 5,000 | persuader ISR-SFT: the same sessions with intent-to-strategy targets |
| `runs/small5000/data/sft_persuadee.jsonl` | 5,000 | optional persuadee SFT: predicts the next user/persona reply |
| `runs/small5000/data/preference_seed.jsonl` | 1,000 | one sampled history per preference scenario |
| `runs/small5000/data/test.jsonl` | 500 | one fixed test context per scenario; never used for training |

`sft_persuadee.jsonl` is not another persuader baseline. It trains an optional
local persuadee/user simulator: the input contains the persona, scenario, and
persuader history, while the target is the next real persuadee reply. The
default simulation route uses the `gpt-4o-mini` API instead, so this file is
only needed when running the optional `train-user` command.

The readable, pretty-printed source subsets are retained separately at
`/home1/liujianjian/2-paper-CoLING/data-dailypersuasion/prepared/small5000/original_format/`.
They keep the released `id` / `scenario` / `strategy` / `dialog` schema. The
selection audit, including the chosen session and quality evidence for every
SFT scenario, is `prepared/small5000/manifest.json`.

## Environment

Use an environment with the versions listed in
`requirements-paper-repro.txt`. The current machine already has compatible
versions of Transformers, Datasets, Accelerate, PEFT, TRL, and PyTorch.

```bash
conda create -n dailypersuasion-repro python=3.10 -y
conda activate dailypersuasion-repro
pip install -r requirements-paper-repro.txt
```

The scripts default to the local model path above; pass `--model` only when
using another local Llama 3.1 checkpoint. LoRA is the practical default. The
paper does not disclose whether its original run used full fine-tuning or PEFT,
so the selected training mode is recorded in `run_configs/`.

## Train dialogue and ISR SFT

Both commands show a Trainer progress bar and write checkpoints under
`outputs/`:

```bash
python 02_Llama31_Dialogue_SFT.py --profile small5000 --epochs 2
python 03_Llama31_ISR_SFT.py --profile small5000 --epochs 2
```

Use `--dry-run --max-samples 8` to inspect the resolved configuration without
loading the checkpoint. A small smoke run can be launched with
`--max-samples 1 --output-dir /tmp/dailypersuasion-l31-smoke`.

For an additional continuation epoch or a new selected scenario subset, use
`--adapter-checkpoint` with a new output root; for an exact interrupted-job
resume, use `--resume-from-checkpoint`. See `CONTINUATION.md` for safe commands
and the distinction between these two operations.

For full-parameter multi-GPU training, pass `--finetuning-type full` and a
DeepSpeed config if desired. The paper does not specify the optimizer,
micro-batch, or parallelism settings, so these are implementation choices and
are recorded as such.

## Build GPT-4o-mini simulation preferences and run DPO

Set the API key used both for the default persuadee and (unless overridden)
the judge:

```bash
export OPENAI_API_KEY='YOUR_KEY'
```

The currently configured ChatAnywhere-compatible route is also supported by
passing `--user-url https://api.chatanywhere.tech/v1 --user-api-key-env
CHATANYWHERE_API --judge-url https://api.chatanywhere.tech/v1`. API keys are
selected strictly by endpoint, so a proxy key is never sent to OpenAI.

After ISR-SFT completes, construct the 1,000 preference pairs directly from
the local LoRA checkpoint. The built-in `local://cuda:0` backend requires no
vLLM or web server. The persuadee prompt keeps the API model in character,
preserves beliefs and concerns, permits natural resistance or acceptance, and
forbids mentioning the simulation.

```bash
python 04_Llama31_Ours_ISR_SimDPO_DPO.py --profile small5000 simulate \
  --persuader-url local://cuda:0 \
  --persuader-model runs/small5000/outputs/02_llama31_isr_sft \
  --user-url https://api.openai.com/v1 \
  --user-model gpt-4o-mini \
  --judge-url https://api.openai.com/v1 \
  --judge-model gpt-4o-mini \
  --simulation-rounds 4
```

The simulation is resumable and displays a progress bar. One simulation round
ends after the persuadee has replied, so every reward comparison contains user
feedback; later rounds add another persuader/persuadee exchange. The paper does
not disclose its value of `k`, so the default of four is a documented
reconstruction choice. Use `--workers` to
parallelize API requests after checking provider rate limits. The resulting
`runs/small5000/data/dpo_preferences.jsonl` stores both simulated paths, all prefix
judgments, weights, raw judge responses, and the chosen/rejected order.

Finally run DPO from the ISR adapter:

```bash
python 04_Llama31_Ours_ISR_SimDPO_DPO.py --profile small5000 dpo \
  --isr-checkpoint runs/small5000/outputs/02_llama31_isr_sft
```

The optional paper-style local persuadee route is still available, but it is
not used by default:

```bash
python 04_Llama31_Ours_ISR_SimDPO_DPO.py --profile small5000 train-user
```

## Test three persuaders with the fixed Qwen persuadee

The active comparison has exactly these three persuaders, all on the same 500
fixed contexts from `data/small5000/test.jsonl`:

| Output file / method | Persuader checkpoint | Prompt format |
| --- | --- | --- |
| `01_zero_shot.jsonl` | `/data1/liujianjian/model/Meta-Llama-3.1-8B-Instruct` | dialogue |
| `02_dialogue_sft.jsonl` | `/data1/liujianjian/Persuader/llama3.1-8B-Instruct-dialogue-merged` | dialogue |
| `03_isr_sft.jsonl` | `/data1/liujianjian/Persuader/llama3.1-8B-Instruct-isr-merged` | ISR, with the spoken `Utterance` extracted |

The fixed counterpart for every method is
`/data1/liujianjian/Persuadee/Qwen3.5-9B-persuadee/merged`. The runner stores
the initial held-out history, every generated turn, the complete final history,
and ISR raw outputs when applicable. It never reads the test `reference` field
for generation.

The two BF16 local models must use different GPUs: an 8B Llama plus the 9B
Qwen model do not fit safely together on a 24 GB RTX 3090. A comma-separated
Persuader device such as `cuda:0,cuda:2` enables Accelerate layer sharding when
each of those cards has only partial free capacity; Qwen remains on `cuda:1`.
Each method is loaded sequentially, while Qwen stays loaded for the whole
invocation.

```bash
cd /home1/liujianjian/2-paper-CoLING/0-paper-2024DailyPersuasion/main_experiment_reproduction
python tests/05_Test_Persuader_With_Qwen_Persuadee.py \
  --persuader-device cuda:0,cuda:2 --persuadee-device cuda:1 --rounds 4
```

This creates the three resumable files under `results/persuadee_simulation/`
plus `run_config.json`. A context consists of its original test history followed
by four generated `persuader -> persuadee` exchanges. Four rounds is a visible,
configurable reconstruction choice, not a value disclosed by the paper. All
generation calls use deterministic decoding by default (`temperature=0.0`) and
cap new output at 1,024 tokens.

To run one route only, either use the generic method selector or the explicit
zero-shot wrapper:

```bash
python tests/05_Test_Persuader_With_Qwen_Persuadee.py --method 02_dialogue_sft \
  --persuader-device cuda:0,cuda:2 --persuadee-device cuda:1
python 01_Llama31_ZeroShot.py --persuader-device cuda:0,cuda:2 --persuadee-device cuda:1
```

Use `--limit 2 --force` for a two-context smoke run in a different
`--output-dir`; do not mix it with the final output directory. The runner
rejects an incompatible existing output (model path, number of rounds,
temperature, or token limit) unless `--force` replaces that method's file.

`tests/test_reproduction.py` remains the common unit-test suite for data,
metrics, and training/simulation helpers. `tests/test_persuadee_simulation.py`
adds common tests for this runner; it is intentionally shared by all three
methods rather than duplicated per checkpoint.

## Score the fixed-persuadee outputs

After the three JSONL trajectory files are complete, run
`tests/06_Evaluate_Main_Experiment.py`. It uses the first generated Persuader
turn for the paper-comparable automatic metrics: two order-swapped blind
judgments for Win-Rate and character-level ROUGE-L against the held-out
reference. The saved rows also retain the full trajectory and final Persuadee
response. By default the local `/data1/liujianjian/model/Qwen3.5-9B` judges and
updates `results/main_experiment_table.md`:

```bash
python tests/06_Evaluate_Main_Experiment.py \
  --judge qwen --qwen-device cuda:1
```

The per-method judgment files are stored under
`results/persuadee_evaluation/qwen35_9b/`. The optional API judge creates a
second table without replacing the Qwen table:

```bash
export OPENAI_API_KEY='YOUR_KEY'
python tests/06_Evaluate_Main_Experiment.py --judge gpt \
  --gpt-model gpt-5.6-luna
```

The API table is `results/main_experiment_table_gpt5_6_luna.md`. Both scoring
stages are resumable; use `--force` only when intentionally replacing a judge
run.

## Legacy one-utterance reference evaluation

The evaluator directly loads the untouched base model and each trained LoRA
checkpoint in turn through `local://cuda:0`, releasing GPU memory between
methods. No local model server is required. The judge defaults to OpenAI
`gpt-4o-mini` and therefore requires `OPENAI_API_KEY`:

```bash
python evaluate_main_table.py --profile small5000 \
  --judge-url https://api.openai.com/v1 \
  --judge-model gpt-4o-mini
```

To run one method at a time, use
`--only-method ... --generate-only`, then use `--judge-only`. Generation and
judgment files are append-only and resumable. All long stages show `tqdm`
progress bars.

The table reports:

- Win-Rate: the mean of two order-swapped pairwise judgments, with a split
  decision worth 0.5.
- ROUGE-L: character-level LCS F1, as stated in the paper. Whitespace is
  excluded and the score is multiplied by 100. The JSON summary additionally
  records the released trainer's `jieba` + `rouge_chinese` variant so the
  discrepancy between the paper prose and released generic metric is visible.
- Position-order consistency as a diagnostic, not a paper headline metric.

The paper's published LLaMA-2 Table 2 values remain in the table as reference
columns. The local Llama 3.1 measurements and their deltas are the actual
adaptation results.

## Legacy paper-protocol zero-shot control

Generate and judge the same 500 fixed random-turn contexts with the untouched
local Llama 3.1 checkpoint. It is loaded directly, without a model server:

```bash
export OPENAI_API_KEY='YOUR_KEY'
python evaluate_main_table.py --profile small5000 --only-method zero_shot
```

Use `--generate-only` if judgment should run later. Successful examples are
resumable from append-only JSONL files. The legacy evaluator's zero-shot prompt
never reads strategy annotations or future turns. The older 200-scenario interactive baseline remains at
`/home1/liujianjian/2-paper-CoLING/train/Baseline1_zero_shot.py`; it is a
different demo protocol and is intentionally excluded from the paper table.

## Verification

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
python prepare_data.py --profile small5000
python 02_Llama31_Dialogue_SFT.py --profile small5000 --dry-run --max-samples 8
python 03_Llama31_ISR_SFT.py --profile small5000 --dry-run --max-samples 8
python 04_Llama31_Ours_ISR_SimDPO_DPO.py --profile small5000 train-user --dry-run --max-samples 8
python 04_Llama31_Ours_ISR_SimDPO_DPO.py --profile small5000 dpo \
  --isr-checkpoint runs/small5000/outputs/02_llama31_isr_sft \
  --dry-run --max-samples 8
```

The paper does not publish split IDs, the exact optimizer/batch settings,
simulation round count, numerical prefix weights, complete utterance-level judging
prompt, or split-vote aggregation. This reconstruction exposes those choices
and records them instead of presenting them as official details.
