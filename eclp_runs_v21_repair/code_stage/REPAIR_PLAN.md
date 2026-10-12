# ECLP 修复计划与进度（2026-10-03）

用户范围：提高当前 ECLP/ELPO 创新质量，修复问题；baseline 由用户另行运行。

## 当前步骤

1. 已完成：基于远端原文件适配双状态路径，使公开话语/策略损失能训练 correction、gate、posterior norm 和误差投影；保留 ISR reasoning 辅助损失和 PPL shaping。
2. 已完成：actor system prompt 与原 baseline 对齐，纳入配置和协议锁；远端 Python 3.11 编译及 CLI 注册通过。
3. 已完成：将 policy prior 改为冻结参数的真实可导计算，去掉 BF16 差值相加和错误的 identity Jacobian。
4. 已完成：进程启动时固定实现 SHA；锁/检查点核验版本及来源一致性，防止运行中换文件造成旧权重冒充新代码。
5. 已完成：可复用的 CPU/A100 BF16 梯度与数值回归、真实先验 Jacobian 和原数据审计；所有证据写入独立修复目录。
6. 已完成：独立目录完成 lock/prepare 和两轮真实 actor/Persuadee smoke；最终结果见 `eclp_runs_v21_repair/gpu_smoke_repaired.json`。task_effectiveness_verified=false，任务效果仍需对照实验。
7. 已修复、复验通过：embeds-only 生成遗漏 prompt token 的默认重复惩罚。读取原 actor generation_config 的 1.05，在真实公共 prompt 与生成 token 上按 HF 同一规则施加惩罚；连续槽位不伪造词表 ID。

## 当前运行

旧版 ECLP SFT：PID 872814，父进程 872813；输出在 `eclp_runs_gpu4/sft_full`。
2026-10-03 本次读取日志时为第 936/7484 步，最近提交 checkpoint 为第 500 步。
该进程的内存代码仍是 v2.0；不会自动获得新修复。保留进程、原数据、baseline 和旧结果。
新版本使用独立 `eclp_runs_v21_repair`，旧 lock/checkpoint 不自动迁移。
旧版保存时会重读磁盘脚本 SHA，文件替换后的 checkpoint 可能有错标的 implementation hash；核验时必须同时检查 version 和 provenance/run hash，不能只看单个 SHA 字段。

## 创新提升重点

- 主要贡献应是行为条件预测、真实反馈纠正及其对后续动作的可测影响。
- “多几个 latent/attention/SFT→RL/PPO”不能单独作为新颖性。
- ELPO Jacobian 衡量反馈特征敏感性，不保证成功方向；需在未参与拟合的训练场景上验证，再比较 isotropic/random/effect 的同预算实际回报。
- 原 ISR reasoning 是教师弱监督，不能把状态预测器说成仅由真实观察损失学习；公开梯度隔离与私有辅助梯度须分别报告。
- CCF-A 所需证据仍是参数/监督匹配对照、eclp/no_error/stateless、状态干预、多 seed 和任务效果。当前修复不能证明录用或性能增益。
