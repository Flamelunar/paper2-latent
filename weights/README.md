# 未合并微调权重

权重附件位于 [snapshot-20261012 Release](https://github.com/Flamelunar/paper2-latent/releases/tag/snapshot-20261012)，不合并到基础模型。`manifest.json` 列出实际已发布包、原 checkpoint、每个文件及压缩包的 SHA-256。

下载并校验所有包，在仓库根目录运行：

```bash
python3 weights/download.py --all
```

只下载最终 ECLP（例如 Qwen）：

```bash
python3 weights/download.py --name qwen25-latent_rl
```

包会恢复其清单记录的相对目录。ECLP 包包括 `adapter/`、`heads.pt`、`reference_policy.pt`、`checkpoint.json` 等加载元数据。当前 `Runner` 加载器需要 reference policy 文件，因此一并保留。它们不包含 `train_state.pt` 或其他优化器状态，仅适合推理／以新优化器继续训练，不能按原 optimizer 无缝恢复。

基线适配器恢复至 `fine_tuned_baselines/<package>/`，需要把原配置中的服务器绝对路径改为该本地路径。基础 LLM、judge 和模拟器模型须单独准备。本快照不包含 merged 基础模型。

清单包含 Qwen/Llama/DeepSeek ECLP 两阶段共 6 包、Qwen 普通 SFT/text GRPO 2 包、已定位的基线权重 8 包，以及完整 ISR 训练数据 1 包。未定位到的 Qwen 历史基线适配器及 DeepSeek LatentRATT 不在本清单；不能据此认为其权重已上传。

ISR 数据包解压后直接得到 `exp_code_ljj/data/isr_sft.jsonl`，文件内容未改。已有 gzip/LFS 数据入口继续保留。
