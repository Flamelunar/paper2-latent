# Qwen2.5-7B LatentRATT 劝说实验

这份代码把 ICLR 2026 LatentR3 的连续隐式推理结构用于 DailyPersuasion。论文 PDF 在本目录；官方仓库已下载到 `upstream_R3/`，当前 HEAD 为 `c558b4404fefff7321d8f96c25c0a7cbfd47dde7`。`latent_model.py` 直接导入官方 `src/latent/layers.py` 的 `SelfAttentionLayer`，其余代码适配 Qwen2.5-7B、ISR 劝说数据和现有五轮模拟接口。

## 方法

- 在 Qwen2.5-7B-Instruct 最后一层 hidden states 上运行 LatentRATT，生成 1 个连续推理向量，插入提示词与公开回答之间。
- SFT 只把 `Utterance` 作为公开输出目标。心理分析和策略标签以 0.2 权重参与辅助损失，不会作为生成时的文本前缀。训练历史中的 ISR 推理也会转成公开发言，以匹配测试提示词。
- LR-GRPO 固定 Qwen 和 LoRA 权重，只更新 LatentRATT；对隐式向量加 Gaussian 噪声，以正确发言的负困惑度给奖励，使用 batch 中未加噪声样本的平均奖励作为基线。
- LR-GRPO 为每条会话缓存冻结模型的提示 hidden states 和 KV 前缀，再从该缓存计算各个 latent proposal；这只减少重复提示前向，奖励、优势和策略损失保持不变。
- 训练与生成模型权重及矩阵计算使用 BF16，均限定 A100；没有量化。PEFT 保存的 LoRA adapter 权重文件为 FP32，这是优化器权重存储格式，前向计算进入 BF16 autocast。

## 完整 ISR 运行

源数据位于 `../exp_code_ljj/data/isr_sft.jsonl`，本轮直接使用全部 **59,868 条会话**，覆盖10,000个训练场景，不做每场景抽样。`prepare_full_isr.py` 校验来源哈希、会话数、场景数和测试集交叉后，生成 `data/isr_all_59868.manifest.json`。SFT 与 LR-GRPO 都使用完整源文件；独立的2,000条偏好数据不参与训练，也不运行DPO。

正式入口：

```bash
bash run_full_isr_59868_a100.sh
```

脚本直接从原始 Qwen2.5-7B-Instruct 开始，先运行一轮 SFT，再运行一轮 LR-GRPO；不做预热，也不以 baseline3 结束作为启动条件。训练指定 A100、BF16、无量化。检查点包含模型、优化器、随机状态和数据游标；遇到 CUDA OOM 时写入恢复点，释放训练进程并等待 A100 显存恢复后续跑，不终止其他训练任务。每阶段最多保留最近三个定期检查点。

训练结束后，`audit_full_run.py` 再核验完整数据清单、两阶段步数、A100/BF16 配置和最终模型权重，并将验证结果写入模型目录。入口随后用与 Baseline 1/2 相同的测试提示和采样设置（五轮、temperature 0.95、top-p 0.7、top-k 50、最多512个新 token）生成1,000组对话，保存到 `../exp_code_ljj/artifacts/generation_5rounds/baseline5_latentRATT.json`。生成程序逐条追加，并在完成时校验固定测试集索引及五轮双方发言；不调用 Judge。

Judge 评估等待提示词确认后手动启动。先设置 `LATENTRATT_EVALUATION_READY=1`，再运行 `evaluate_latentratt.sh`；脚本复用既有评测代码，运行 Selene v17 和 Gemma v11 联合评测（各自同时判断五轮成功与对 golden reply 的 Win-Rate），最后只更新主表中的 LatentRATT 行。未显式设置该变量时，评估入口会记录 deferred 并退出。

本轮权重目录为 `/app/hzj/liujianjian/qwen2.5-7b-latentRATT-all-59868/{sft,rl}`，日志在 `logs/full_isr_59868/`。SFT 每25步、LR-GRPO 每100步保存一次检查点；每阶段保留最新三个检查点和阶段最终权重。`run_config.json`记录数据量和训练设置。
