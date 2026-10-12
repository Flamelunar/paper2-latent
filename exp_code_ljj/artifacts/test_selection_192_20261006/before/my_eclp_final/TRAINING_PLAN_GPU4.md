# GPU4 当前训练与评估计划

当前方法为 ECLP + 离线隐式策略优化，详细研究依据和限制见
`OFFLINE_ADAPTATION_PLAN.md`，参数见 `offline_training_plan.json`。

1. 保留当前正在运行的 Qwen2.5-7B-Instruct SFT。使用原 59,868 会话、
   10,000 场景，一个完整 epoch，目标 7,484 次优化更新。保留 ISR
   reasoning 辅助监督（权重 0.1）和状态纠错；部署不输出推理文本。
2. 用固定的原 1,800 条 RL 场景跑一遍。每条 4 个高斯候选加 1 个
   冻结 SFT 参考；每 64 场景收集后更新小策略网络。全部使用已有
   公开历史和 gold_response 进行教师强制评分，不生成训练对话。
3. 默认奖励为 −NLL 加权重 0.1 的有界纠错增益。对照开关为纯 PPL、
   纯 NLL；详见研究方案。每个候选记录原始 NLL、分支 NLL、奖励、
   优势及信任域诊断；只优化策略模块，语言生成 LoRA 与状态网络冻结。
4. RL 完成后在固定 200 条 dev 上进行离线答案似然和纠错增益检查。
   若平均 NLL 比固定 SFT 参考恶化超过 0.05，则保存诊断并停止。
   最终 checkpoint 固定为全量一遍 RL 后的最后提交点，不用测试选模型。
5. 用该 checkpoint 对原 index 0–999 的 1,000 个测试场景生成完整五轮
   对话。此阶段使用原固定 Persuadee，最终 Judge 不参与生成。
6. 在同一物理 GPU4 上顺序运行 Selene-8B、Qwen3-14B，沿用既有
   success/Win-Rate 规则和解析协议。分别原子替换
   `exp_code_ljj/results/main_experiment_table.md` 对应分组最后的 Our 行。

模型、原数据、最终评估协议继续使用当前锁；网络 BF16、不量化。
SFT 核心源码保持原 SHA256，不为本次离线训练改动或重启其子进程。
监督进程换为 `run_gpu4_offline_pipeline.py`；其运行合同另存
`eclp_runs_gpu4/offline_pipeline_contract.json`，保留旧合同供复核。

入口：`my_eclp_final/run_gpu4_full_chain.sh`。
实时状态：`eclp_runs_gpu4/pipeline_status.json`。
独立存活监控：`my_eclp_final/watch_gpu4_pipeline.py`，
记录于 `eclp_runs_gpu4/watcher_status.json` 和 `watcher_events.jsonl`。
仅当监督进程意外消失且阶段未标记失败、完成或中断时重启监督进程；
若 RL、生成或评分的计算子进程仍存活，先等待它退出，避免在 GPU4
重复加载模型；
训练子进程仍由原监督进程按已提交 checkpoint 接续。
SFT 日志：`eclp_runs_gpu4/sft_full/train.jsonl`。
离线 RL 状态：`eclp_runs_gpu4/offline_rl/progress.json`。
每批奖励和更新：`eclp_runs_gpu4/offline_rl/blocks/`。
阶段日志：`eclp_runs_gpu4/logs/`。
完工复核：`my_eclp_final/audit_gpu4_completion.py`；仅在流水线
实际完成后由独立监控自动运行，逐项复核 checkpoint、RL 奖励、dev、测试、双 Judge
和表格，报告保存为 `eclp_runs_gpu4/completion_audit.json`。

从 SFT 已提交的 optimizer、RNG 和数据游标继续；RL 从每批已提交
checkpoint 继续。失败的未提交短批保留在独立 attempt 下，重做该批。
仅当训练、1000 条生成、两组真实 Judge 记录和两条表格结果实际完成，
才把整个实验报告为完成。

SFT 用近期真实步速估时，RL 用机制诊断和首批正式吞吐校准。
当前采用单条 NLL 前向，因 GPU 预检发现 BF16 批量路径存在约 0.013
的 NLL 差异；不为提速放宽奖励一致性要求。最终测试评估时长另计。
