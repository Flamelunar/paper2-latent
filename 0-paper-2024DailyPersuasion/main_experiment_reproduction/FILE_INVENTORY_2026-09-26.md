# 当前主表对应的文件清单（2026-09-26）

范围：`results/main_experiment_table.md` 当前的 Qwen2.5 zero-shot / Dialogue-SFT、PersuGPT / Qwen3.5-9B 两组 Judge、五轮对话结果。这里只盘点；尚未移动或删除数据。

## 必须保留：可追溯数据和结果

| 文件或目录 | 用途 | 当前状态 |
| --- | --- | --- |
| `runs/paper10k/data/dialogue_sft.jsonl` | Qwen2.5 Dialogue-SFT 的 59,868 条训练样本 | 与根目录 `data/dialogue_sft.jsonl` 逐字节相同；训练脚本当前读取 runs 版本 |
| `runs/paper10k/data/evaluation_points.jsonl` | 1,000 条独立测试场景与 golden 回复 | 与根目录 `data/evaluation_points.jsonl` 逐字节相同；生成脚本当前读取 runs 版本 |
| `runs/paper10k/data/manifest.json`、`dataset_info.json` | 划分、行数、SHA-256、数据格式的记录 | manifest 的路径指向 runs 版本；根目录 manifest 路径指向根目录版本 |
| `runs/qwen25_dialogue_sft/generation_5rounds/` | zero-shot、Dialogue-SFT 各 1,000 条五轮轨迹及运行配置 | 当前主表使用的生成数据 |
| `runs/qwen25_dialogue_sft/evaluation_persugpt/` | PersuGPT 对两种方法各 1,000 条五轮成功判断 | Win-Rate 字段是旧的“仅首轮比较”，只能复用成功判断 |
| `runs/qwen25_dialogue_sft/evaluation_qwen3_5_9B/` | Qwen3.5-9B 对两种方法各 1,000 条五轮成功判断 | Win-Rate 字段也是旧协议，只能复用成功判断 |
| `results/main_experiment_table.md` | 当前主表 | 新 Win-Rate 四格均为 pending，尚未重评 |
| `runs/paper10k/run_configs/qwen2.5-7b-instruct-dialogue-sft-paper10k.json`、`run_configs/qwen25_dialogue_sft_paper10k.json` | 实际训练参数与计划记录 | 保留以追溯模型训练版本 |
| `/data1/liujianjian/Persuader/qwen2.5-7b-instruct-dialogue-sft-paper10k/` | 实际训练完成的 LoRA adapter | 位于项目外，不在本次目录清理范围 |

## 必须保留：现用入口和共享模块

- 训练：`run_qwen25_dialogue_sft_paper10k.sh`、`train_qwen25_dialogue_sft.py`、`repro/train_backend.py`、`repro/profile.py`、`repro/constants.py`。
- 生成：`run_qwen25_table2_generation.py`、`repro/persuadee_simulation.py`、`repro/simulation.py`。
- 评分和主表：`evaluate_qwen25_table2.py`、`finalize_qwen25_main_table.py`。
- 传递依赖：`repro/trajectory_evaluation.py` 的 `parse_winner` 仍被现用评分脚本导入；该模块又导入 `repro/evaluation.py`。删除 Llama 旧代码前需先把这一小段解析函数移入现用评分模块或共享工具。
- 数据重建与环境：`prepare_data.py`、`repro/data.py`、`requirements-paper-repro.txt`、`QWEN25_TABLE2.md`。数据已存在时并非每次运行都需要重建，但保留可追溯的数据生成代码。

## 可以迁移或清理，但不属于当前主表输入

| 文件或目录 | 判断 |
| --- | --- |
| 根目录 `data/dialogue_sft.jsonl`、`evaluation_points.jsonl` | 内容与 runs 版本相同，可作为迁移后的唯一副本；须同步修改训练、生成脚本和 manifest 路径 |
| 根目录 `data/isr_sft.jsonl`、`user_model_sft.jsonl`、`preference_seeds.jsonl` 与 runs 对应文件 | 两处各自逐字节相同；当前 Qwen Dialogue-SFT 主表不使用，可在确定不保留 ISR/DPO 路线后清理 |
| `runs/qwen25_dialogue_sft/generation_5rounds_bf16_autocast/` | zero-shot JSONL 与主生成目录逐字节相同，可清理重复副本 |
| `runs/qwen25_dialogue_sft/evaluation_reuse_zero_shot/` | zero-shot JSONL 与 Qwen3.5-9B 评估目录逐字节相同，可清理重复副本 |
| `runs/qwen25_dialogue_sft/evaluation_qwen3_5_9B_table2/`、`evaluate_qwen25_table2_metrics.py` | 374/1000 条旧首轮 Win-Rate/ROUGE 评估及其脚本；主表不使用 |
| `runs/qwen25_dialogue_sft/generation_bf16_smoke/` | 1 条冒烟样本；主表不使用 |
| `runs/qwen25_dialogue_sft/generation_final_bf16/`、`run_final_bf16_pipeline_resume.sh`、`finalize_qwen25_bf16_results.py` | 额外的独立 BF16 生成/汇总路线；不属于当前主表。目前后台生成进程已暂停，文件仍可能被恢复的进程继续写入；先不要迁移或删除 |
| `results/main_experiment_table.json` | 旧 Llama 3.1、500 条测试、ROUGE 的结果；与当前 Markdown 主表不配套 |

## 历史 Llama 3.1 路线，可在依赖拆开后移走或删除

- 入口：`01_Llama31_ZeroShot.py`、`02_Llama31_Dialogue_SFT.py`、`03_Llama31_ISR_SFT.py`、`04_Llama31_Ours_ISR_SimDPO_DPO.py`、`evaluate_main_table.py`、`save_merged_models.py`。
- 调度与文档：`run_card2_small5000.sh`、`run_paper10k_sft_3090.sh`、`run_table2_paper10k.sh`、`run_remaining_pipeline.sh`、`CONTINUATION.md`、`TABLE2_LOCAL_10K.md`、当前以 Llama 为主的 `README.md`、`0-任务运行.md`、相关 `scripts/` 和旧 `run_configs/`。
- 结果与运行数据：`data/small5000/`、`runs/paper10k/outputs/01_llama31_dialogue_sft/`（约 994 MB）、`results/persuadee_*`、根目录 `logs/`、旧 Qwen 首轮评估输出及大部分试运行日志。
- `repro/train_backend.py`、`repro/persuadee_simulation.py`、`repro/simulation.py`、`repro/constants.py` 虽包含 Llama 旧分支，但被当前 Qwen 流程导入，不能整文件删除。

## 迁移时的路径依赖

1. 将根目录 `data/` 作为 paper10k 的唯一数据目录；目前根目录已有与 runs 相同的 Dialogue-SFT 和测试数据，无须再复制大文件。
2. 修改 `repro/constants.py` 的 paper10k 数据目录、训练脚本的 `TRAIN_FILE` / `TEST_FILE` / manifest、生成脚本的 `TEST_DATA`，并更新当前 Qwen 文档和配置中的路径。
3. 将主表所需的 `generation_5rounds`、两套成功判断 JSONL 和配置归档到根目录下明确的结果位置，同时修改评分和主表汇总脚本的路径。旧 Win-Rate 不能写入新指标列。
4. 先处理已暂停的后台进程，或保留兼容路径；否则恢复进程可能写到迁移前的目录。

当前没有启动新评估。此清单没有删除任何文件。
