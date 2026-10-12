# Five-round generation artifacts

The root directory contains the current canonical test generations. Every canonical
baseline uses a model-independent filename and exactly 1,000 rows for the fixed
`data/evaluation_points.jsonl` split:

- `baseline1_zero_shot.jsonl`
- `baseline2_dialogue_sft.jsonl`
- `baseline3_isr_simdpo.jsonl`
- `baseline4_ma2p.jsonl`
- `baseline5_latentRATT.json`

`legacy/` contains the original Qwen2.5 files and names used by the earlier Judge
runs. They are retained for reproducibility and source tracing; new generation and
table code should use the canonical files in the root directory.

`run_config_baseline3.json` records the Baseline 3 generation settings. All three
canonical Baseline 1/2/3 files use the common five-round protocol, the same fixed
1,000 test contexts, and BF16 model inference.
