# Llama-3.1 8B 修复版生成与评估计划（2026-10-07）

## 范围

本轮只处理 Llama-3.1 8B 结果，训练、对话生成及 Judge 评估仅调度到 GPU4。固定使用原始 `target_turn == 1` 且 `history == []` 的 192 条；每个场景从空历史开始，最多生成五轮。训练数据及训练集不变。

## 修复依据

旧模拟器的 user prompt 含有精确 `Persuasion goal`，尽管 system prompt 声称模拟对象只依据背景和对话反应。这会向被说服者模拟器泄露劝说者的隐藏目标，可能抬高 Success。所有本轮方法改用相同的目标盲模拟对象：只给背景和截至当前轮的公开对话，不给精确目标、心理标注、策略标注或 golden reply。

zero-shot 劝说者只看背景和公开历史，不接收目标。Dialogue-SFT、ISR-SFT + SimDPO、MA2P 及 Our 保留各自方法所需的目标条件输入；这是方法设定差异，结果解释时应注明。Our 使用完整 ECLP 检查点、隐式状态及策略向量，保持 persuasion-only 输出，不复用 zero-shot 的生成提示词。

旧版目标可见模拟器生成的 112 条部分结果保留在 `model_family_fresh192_20261007`，并以 `superseded_run.json` 标记；不评估、不合并、不写入新表。

## 生成与评估

在同一固定子集、Qwen3.5-9B persuadee、BF16、温度 0.95、top-p 0.7、top-k 50、最多 512 个新 token、相同场景种子和公开输出边界下，依次处理：

1. Zero-shot：全新生成 192 条，随后使用 Selene 和 Qwen3 分别评估。
2. Dialogue-SFT：全新生成 192 条，随后使用两位 Judge 评估。
3. ISR-SFT + SimDPO：全新生成 192 条，随后使用两位 Judge 评估。
4. MA2P：全新生成 192 条，保留 Perception → World Model → Persuader 流程，随后使用两位 Judge 评估。
5. Our：等待当前 ECLP SFT 与 offline RL 完成，使用最终 checkpoint 全新生成 192 条，随后使用两位 Judge 评估。

评分沿用最新统一 v2 提示词、golden 和代码：Success 为五轮内任意一轮接受准确目标；Win@1 比较首轮完整劝说语句；Win-Full 比较生成的完整对话与 golden 的全部可用对话；均报告平均轮数和格式未解析数。AB/BA 换序按 win=1、tie=0.5、loss=0 取平均，不把换序请求算作独立样本。

已知 Judge 风险：Selene 曾把“认真考虑，稍后答复”判为投资目标接受。该例记录为 Judge 语义误判；保留统一提示词和规则，不针对单个方法改分。新结果应结合 Qwen3 分数及该误判记录解释。

LatentRATT 当前没有完整的 Llama SFT+RL checkpoint，本轮保留待完成状态，不以 smoke checkpoint 或其他骨干模型代替。

## 产物

- 新实验目录：`exp_code_ljj/artifacts/model_family_fresh192_nogoal_20261007/meta_llama31`
- 方法调度：`my_eclp_final/model_family_jobs/llama_fresh_campaign.py`
- 生成器：`my_eclp_final/model_family_jobs/generate_family.py`，启用 `--persuadee-goal-blind`
- 状态：`my_eclp_final/model_family_jobs/llama_fresh_status.json`
- 结果表：`exp_code_ljj/results/main_experiment_table_llama31.md`

每个方法的 generation/evaluation manifest 保存模型、固定索引、提示词来源、checkpoint、GPU、精度及源行哈希。只有完整 192 条生成并由两位 Judge 完成评估后，分数才作为正式结果写入表格。
