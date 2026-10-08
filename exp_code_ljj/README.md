# DailyPersuasion 四个 Baseline（固定 paper10k）

## 当前固定测试约定（2026-10-06 生效）

用户明确要求：今后主实验仅测试原测试集中 **192 个无原始前置对话的场景**。筛选条件为 `target_turn == 1` 且原始 `history == []`，使用 `exp_code_ljj/data/test_selection_192.json` 固定原始索引，数据文件为 `exp_code_ljj/data/evaluation_points_from_start192.jsonl`。各方法与 Selene/Qwen3 必须使用相同子集；禁止重新随机抽取 192 条或重新编号后误配 golden。

第一轮只有背景和目标，无原始对话、候选策略或额外心理状态标注；后续保留本次生成的公开对话。最多五轮，任一轮达到目标即场景成功。Win@1 比较双方首条完整劝说回复；Win-Full 比较模型全部五轮与同场景原始 golden 全部可用对话。换序独立评分及平均轮数（失败计5）沿用统一 v2。

训练数据、SFT/RL 数据和开发集不变。`data/evaluation_points.jsonl` 的原 1000 条及其 manifest 保留作来源档案，不作为今后全量主测试入口。当前 192 条分数通过筛选已有逐条判断获得，不启动训练、生成或 Judge。旧的 1000 条运行命令/合同属于历史记录；未来启动前必须确认生成和评估仅覆盖清单中的原始索引，结果目录另建，禁止将 192 条任务追加进原 1000 条合同。GPU4 上其他任务不得停止。

当前主表 `exp_code_ljj/results/main_experiment_table.md` 的 unified v2 部分使用 192 条；上方旧协议 1000 条表仅作历史档案。格式失败不能从分母中删除，未完成方法显示 `待评估 / 192`。

四种方法共用 `data/evaluation_points.jsonl` 的 1,000 个独立测试场景、同一个五轮 Persuader/Persuadee 生成流程，以及 `evaluate_baselines.py` 中原有的两种 Judge 提示词和评分规则。训练方法和模型路径由入口参数决定。所有新实验使用模型无关的 `baselineN_*` 文件名；先前完成的 Qwen2.5 轨迹和 Judge 记录保留原始文件名作溯源。

| 方法 | 入口 | 训练内容 | 当前状态 |
| --- | --- | --- | --- |
| Baseline 1: Zero-shot | `baseline1_zero_shot.py` | 无训练，直接用基座生成 | 已有 Qwen2.5 评测 |
| Baseline 2: Dialogue-SFT | `baseline2_dialogue_sft.py`、`run_baseline2_dialogue_sft.sh` | 原始劝说者对话 SFT | 已有 Qwen2.5 adapter 和评测 |
| Baseline 3: ISR-SFT + SimDPO | `baseline3_isr_sft_simdpo.py` | ISR SFT → 模拟偏好 → DPO | ISR-SFT 训练中，暂无完整评测结果 |
| Baseline 4: MA2P | `baseline4_ma2p.py` | 预留 | 尚未实现 |

## 固定数据与公平性

`data/dialogue_sft.jsonl` 和 `data/isr_sft.jsonl` 各有 59,868 段会话，按行对应同一批 10,000 个训练场景。ISR 数据中的 assistant 标签含 `Intent to Strategy Reasoning`、`Selected Strategy`、`Utterance`，初始提示含候选策略。`data/preference_seeds.jsonl` 含独立的 2,000 个场景；`data/evaluation_points.jsonl` 含独立的 1,000 个测试场景。`data/user_model_sft.jsonl` 为可选的本地用户模型训练数据。这五份数据的 SHA-256 均记录在 `data/manifest.json`；baseline3 在开始前核对文件、会话对齐和三个场景集合无交叉。

**测试提示词共享。** `repro/persuadee_simulation.py` 对 Baseline 1、2、3 都调用同一个 `dialogue_prompt`，固定 persuadee 用同一个 `user_prompt`。ISR 模型的原始输出会保存，只有 `Utterance:` 后的发言送入下一轮；Judge 一律只看公开发言。`evaluate_baselines.py` 保留原有的五轮严格成功率与全部五句对 golden 的 Win-Rate 提示词。统一测试提示词可能与 ISR 训练阶段的三字段提示词不完全匹配，这一取舍由共享测试协议决定。

Baseline 1/2 当前沿用 Qwen2.5-7B-Instruct；Baseline 3 ISR-SFT + SimDPO 当前以 Meta-Llama-3.1-8B-Instruct 为 Persuader 基座。ISR-SFT 和 SimDPO adapter 保存在 `/data1/liujianjian/Persuader/` 下，不合并。偏好模拟直接使用已经微调好的 `/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged` 作为用户模型。SimDPO 偏好 Judge 和主表中的一个 Judge 使用 `/app/hzj/models/Qwen3.5-9B`，另一个主表 Judge 使用 Meta-Llama-3.1-8B-Instruct。训练、模拟、生成和评分全部在 A100 上使用 BF16；显存暂时不足时等待，不改变精度。

## Baseline 3：准备、训练、模拟和 DPO

本次运行把物理 GPU 2（A100）暴露给训练进程，权重与训练计算使用 BF16。`train-isr` 使用与 Baseline 2 相同的 SFT 场景和训练超参数，只换为已标注的 ISR 样本；训练输出写在 `/data1/liujianjian/Persuader/`：

```bash
cd /app/hzj/liujianjian/ljj-2-paper/exp_code_ljj
CUDA_VISIBLE_DEVICES=2 python baseline3_isr_sft_simdpo.py --profile paper10k train-isr
```

完整流程可通过 `./run_baseline3_reproduction_a100.sh` 启动。它会等物理 GPU 2 空闲后自动训练、续跑 checkpoint、生成与评分；日志写入 `/data1/liujianjian/Persuader/logs-baseline3-paper10k/`。

现有 persuadee 的 `merge_metadata.json` 表明它已由 Qwen3.5-9B 和已训练 adapter 合并，权重精度为 BF16。本次完整复现脚本直接传入这个现有模型，不执行可选的 `train-user`。三个本地 BF16 模型均加载到 A100 `cuda:2`；运行前会按权重总量检查显存。

```bash
python baseline3_isr_sft_simdpo.py --profile paper10k simulate \
  --persuader-url local://cuda:2 \
  --persuader-model /data1/liujianjian/Persuader/baseline3_isr_sft_adapter \
  --user-url local://cuda:2 \
  --user-model /app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged \
  --judge-url local://cuda:2 \
  --judge-model /app/hzj/models/Qwen3.5-9B \
  --simulation-rounds 4
CUDA_VISIBLE_DEVICES=2 python baseline3_isr_sft_simdpo.py --profile paper10k dpo
```

模拟从每个 preference 场景抽取一个真实历史，ISR 模型采样两个完整候选；两条路径各进行四轮模拟反馈，Judge 比较每轮前缀，按递减权重聚合，写出 `[chosen, rejected]` 给 DPO。原始路径、每轮判断与设置都保存在 `data/simulation_preferences.jsonl` 及相邻的配置/摘要文件。`--limit` 与 `--max-samples` 可用于小规模试运行，正式 DPO 默认要求 2,000 个偏好对。

论文使用 LLaMA-2 Chat 13B、训练过的用户模型与 GPT-4；这里使用 Meta-Llama-3.1 ISR Persuader、本地 Qwen3.5 persuadee 和本地 Qwen3.5 偏好 Judge。四轮模拟及递减权重也是可配置的重现选择；论文未公布模拟轮数和数值权重。测试中的固定 Qwen3.5 persuadee 与两个本地 BF16 Judge 属于本项目统一评估协议。

模拟与测试默认使用同一份 Qwen3.5 persuadee 权重，但场景划分严格分离。Baseline 3 的偏好训练会对该 persuadee 的行为有所适配；同时，模拟偏好 Judge 与主表中的 Qwen3.5 Judge 使用同一基座，可能影响该 Judge 下的相对胜率。主表中的 Meta-Llama-3.1 Judge 提供另一组判断。解释与其他方法的差距时应保留这些设定。

## 统一生成、评估与主表

```bash
python baseline1_zero_shot.py --profile paper10k --base-model /path/to/base
python run_baseline_generation.py --profile paper10k --method baseline2_dialogue_sft \
  --base-model /path/to/base --baseline2-model /path/to/dialogue_adapter
python run_baseline_generation.py --profile paper10k \
  --base-model /app/hzj/models/Qwen2.5-7B-Instruct \
  --baseline3-model /data1/liujianjian/Persuader/baseline3_isr_simdpo_adapter \
  --persuader-device cuda:2 --persuadee-device cuda:2
python evaluate_baselines.py --profile paper10k \
  --generation-dir artifacts/generation_5rounds \
  --judge-model /app/hzj/models/Qwen3.5-9B --judge-device cuda:2 \
  --evaluation-dir artifacts/success_judgments/qwen35
python evaluate_baselines.py --profile paper10k \
  --generation-dir artifacts/generation_5rounds \
  --judge-model /app/hzj/models/Meta-Llama-3.1-8B-Instruct --judge-device cuda:2 \
  --evaluation-dir artifacts/success_judgments/llama31
python finalize_main_table.py
```

所有当前主实验生成文件统一保存在 `artifacts/generation_5rounds/`，按 `baseline1_*` 到 `baseline4_*` 命名；LatentRATT 也保存在该目录。旧版 Qwen2.5 原始轨迹移到 `artifacts/generation_5rounds/legacy/`，仅供旧评测和来源核验使用。`finalize_main_table.py` 只读取统一目录中的规范文件。依赖列在 `requirements.txt`。

**固定前 500 条评测子集。** `artifacts/generation_5rounds_first500/` 保存旧版 Zero-shot 和 Dialogue-SFT 记录的索引 0-499 子集。旧版 Qwen2.5 全量轨迹位于 `artifacts/generation_5rounds/legacy/`；当前主表生成结果位于 `artifacts/generation_5rounds/` 根目录。
