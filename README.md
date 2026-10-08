# paper2-latent

ECLP 与多轮劝说基线的研究代码、数据及实验结果快照（2026-10-08）。

## 内容

- `my_eclp_final/`：ECLP 隐式立场建模、策略学习、SFT、离线强化学习与评估代码。
- `my_eclp_final/model_family_jobs/`：Qwen、Llama 与 DeepSeek 的训练、生成、评估及调度脚本。
- `baseline5_latentRATT/`：LatentRATT 劝说适配实现及上游代码。
- `0-paper-2024DailyPersuasion/`、`exp_code_ljj/`：zero-shot、Dialogue-SFT、PersuGPT、MA2P 及数据准备实现。
- `exp_code_ljj/data/`：训练数据、原始测试数据与固定 192 条测试合同。
- `exp_code_ljj/results/main_experiment_table*.md`：三个模型的结果表；未完成项保持待完成状态。

## 实验协议

主实验只使用原始 `target_turn == 1` 且 `history == []` 的固定 192 个场景。各方法和 Judge 使用相同索引。每个场景最多五轮，任一轮接受目标计为成功。指标包括 Success、Win@1、Win-Full、平均成功轮数（失败计 5）及格式未解决数量。详细定义见 `TEST_SELECTION_192.md` 与各任务的协议配置。

当前 ECLP 实验输出只包含劝说话语，分析输出版本已取消。目录内部分旧配置与脚本保留用于追溯，不能将历史入口视为当前主实验入口。模型家族最新调度与结果汇总位于 `my_eclp_final/model_family_jobs/`。不同家族的目标/策略输入约定需依照其任务合同，不能默认完全相同。

## 数据与运行

数据使用 Git LFS：克隆后运行 `git lfs install` 和 `git lfs pull`。ISR 训练数据以无损 gzip 格式发布，随后运行 `python3 exp_code_ljj/data/restore_isr_dataset.py` 还原原始文件并校验 SHA-256；训练入口与数据内容不变。不包含模型权重、运行日志或模型生成的对话；`eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl` 是评估参考数据。

代码快照包含原实验服务器路径，运行前应配置本机模型、数据、输出路径及 GPU 分配。依赖文件位于各基线目录；不能直接执行历史调度脚本启动或停止现有任务。训练与评估的入口、提示词和方法说明见各目录 README。

上游项目的许可与署名保留在对应目录；本仓库不替第三方代码或数据重新授予许可。
