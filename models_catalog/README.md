# 权重目录索引

这些目录为原始checkpoint的符号链接，不复制或修改权重。joint_sft保留状态/策略模块；latent_rl为该骨干记录的最终RL checkpoint。历史保存点保留原位。

| 模型 | 阶段 | 分类入口 | 原始checkpoint | 保存点数 |
|---|---|---|---|---:|
| qwen25 | joint_sft | models_catalog/qwen25/eclp/joint_sft | eclp_runs_gpu4/sft_full/checkpoint-007484 | 16 |
| qwen25 | latent_rl | models_catalog/qwen25/eclp/latent_rl | eclp_runs_gpu4/offline_rl/checkpoint-001800 | 29 |
| llama31 | joint_sft | models_catalog/llama31/eclp/joint_sft | eclp_runs_meta_llama31_20261007/sft_full_retry_20261007/checkpoint-007484 | 150 |
| llama31 | latent_rl | models_catalog/llama31/eclp/latent_rl | eclp_runs_meta_llama31_20261007/offline_rl/checkpoint-001800 | 29 |
| deepseek14b | joint_sft | models_catalog/deepseek14b/eclp/joint_sft | eclp_runs_deepseek_r1_qwen14b_20261007/sft_full_retry_20261007/checkpoint-007484 | 150 |
| deepseek14b | latent_rl | models_catalog/deepseek14b/eclp/latent_rl | eclp_runs_deepseek_r1_qwen14b_20261007/offline_rl/checkpoint-001800 | 29 |
| qwen25 | ordinary_sft | models_catalog/qwen25/standard_grpo/ordinary_sft | exp_code_ljj/artifacts/qwen25_standard_grpo_20261011/training/sft/checkpoint-007484 | 15 |
| qwen25 | text_grpo | models_catalog/qwen25/standard_grpo/text_grpo | exp_code_ljj/artifacts/qwen25_standard_grpo_20261011/training/grpo/checkpoint-001800 | 29 |

只核对checkpoint元数据和必要文件存在；未重新加载大模型，也未重新计算所有大权重文件的哈希。完整checkpoint版本ID见weights_index.json。

Llama和DeepSeek各有150个SFT保存点，它们含optimizer状态，占用较大。整理未删除中间checkpoint或optimizer状态；保留它们用于恢复和历史复现。

基座、模拟器和Judge一般在外部 /app/hzj/models 或配置指定的位置，未移入本项目。exp_code_ljj/models原目录只有Model.md，不代表项目没有训练权重。

## GitHub 快照

上述保存点数描述原服务器。本公开仓库只发布清单中的最终未合并微调权重，下载 weights/README.md 所述 Release 包后，此处相对链接才会指向实际文件。历史保存点与 optimizer 状态没有上传。权重文件和包的实测 SHA-256 见 weights/manifest.json。
