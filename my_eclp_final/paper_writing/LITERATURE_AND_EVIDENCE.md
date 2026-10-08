# 文献、代码证据与待补事项

核验日期：2026-10-07。这里记录的是已查阅来源与其能支持的主张，供写作agent复用。检索仅使用公开论文标题、方法名、arXiv编号及项目链接；未将用户内部稿或未公开方法文本上传到检索服务。

## 1. 当前参考文献

| 引用key | 工作与来源状态 | 已查阅证据 | 可用于论文的内容 | 边界 |
|---|---|---|---|---|
| `jin-etal-2024-persuading` | Persuading across Diverse Domains: a Dataset and Persuasion Large Language Model；ACL2024 Long Papers | 原始PDF、ACL项目链接、ACL官方BibTeX；摘要与方法 | DailyPersuasion、ISR、simulation-based preference optimization、PersuGPT | 原论文报告的全量数量与项目重建数量分开；不宣称复现相同split IDs |
| `zhang-etal-2026-ma2p` | MA2P: A Meta-Cognitive Autonomous Intelligent Agents Framework for Complex Persuasion；Findings ACL2026 | ACL官方页面、官方BibTeX、PDF §3/Algorithm1 | Configurator、Perception、World Model、Persuader、Short-term Memory、Evaluator、知识库更新 | 本地K=0三个角色是适配；不等于原论文完整知识库与记忆协议 |
| `zhang-etal-2026-latentr3` | Reinforced Latent Reasoning for LLM-based Recommendation；提供的PDF标明ICLR2026 | 原始PDF §3.1–3.2、官方R3仓库README | LatentRATT层、SFT→latent RL、Gaussian采样、PPL打分、reference advantage | 原目标使用target-token ratio；我们使用Gaussian action ratio；不能说逐行复现LR-GRPO |
| `hao-etal-2024-coconut` | Training Large Language Models to Reason in a Continuous Latent Space；arXiv2412.06769 | 公开arXiv metadata、提供的v4 PDF摘要/引言 | hidden state连续thought、再次作为input embedding的机制 | 引用初始2024年；阅读版本更新于2026。本文没有复现其顺序连续thought链 |
| `han-etal-2025-tomap` | ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind；arXiv2505.22961 | 公开arXiv作者/标题/摘要，PDF §3.1–3.3 | 反对意见、BGE-M3+MLP立场预测、模拟attitude-shift reward与对方意识训练 | 未做本地ToMAP baseline；原论文attitude shift不是本项目Success口径 |
| `schulman-etal-2017-ppo` | Proximal Policy Optimization Algorithms；arXiv1707.06347 | 公开arXiv元数据与已知原算法归属 | clipped likelihood-ratio objective的来源 | PPO工具本身不属于我们的贡献 |
| `hu-etal-2021-lora` | LoRA: Low-Rank Adaptation of Large Language Models；arXiv2106.09685 | 公开arXiv作者、标题与日期 | 低秩语言执行器适配的来源 | 当前Bib采用初始arXiv年份；如切换会议版本应先核实官方引用 |
| `zheng-etal-2023-judge` | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena；arXiv2306.05685 | 公开arXiv作者、标题、摘要，PDF中的position/verbosity bias与swapping位置段落 | 位置、verbosity等judge偏差及人工一致性校准动机 | 原任务的一致率不代表本任务两个judge的准确性 |

稳定链接：

- PersuGPT论文：<https://aclanthology.org/2024.acl-long.92/>；项目：<https://persugpt.github.io/>。
- MA2P论文：<https://aclanthology.org/2026.findings-acl.1550/>；DOI：`10.18653/v1/2026.findings-acl.1550`。
- LatentR3论文：<https://arxiv.org/abs/2505.19092>；代码：<https://github.com/xuwenxinedu/R3>。
- Coconut：<https://arxiv.org/abs/2412.06769>；代码：<https://github.com/facebookresearch/coconut>。
- ToMAP：<https://arxiv.org/abs/2505.22961>。
- PPO：<https://arxiv.org/abs/1707.06347>。
- LoRA：<https://arxiv.org/abs/2106.09685>。
- LLM-as-a-Judge：<https://arxiv.org/abs/2306.05685>。

`references.bib` 中ACL两篇引用直接来自官方导出的BibTeX；其他条目的标题和作者已核实，来源状态按上表标记。不要自动把arXiv年份改成猜测的会议年份。

## 2. 当前方法的代码依据

远程项目根目录：`/app/hzj/liujianjian/ljj-2-paper/`。本地快照保存在 `ccfa-workfiles/paper-handoff/eclp/evidence/current/`；快照中记录每文件hash和采集时间。

| 方法事实 | 文件与定位 |
|---|---|
| State预测、readout、error与gate | `my_eclp_final/eclp_elpo.py`：`EvidenceState.initial/prior/correct/policy_prior/policy_correct` |
| policy与虚拟token | 同文件：`StrategyPolicy`、`LatentHeads`、`ECLPRunner.context` |
| SFT四项loss | 同文件：`session_losses`、`train_sft`调用处；公开最后回复与私有ISR分别监督 |
| 冻结特征及背景初始化 | 同文件：`encode_public`、`initialize`、`consume` |
| Gaussian分布与动作比率 | 同文件：`exploration_cholesky`、`gaussian_log_prob`、`clipped_policy_loss` |
| 同历史、最新error干预 | `my_eclp_final/offline_latent_rl.py`：`paired_contexts` |
| 参考回复平均NLL | 同文件：`target_nlls` |
| 参照居中的修正reward | 同文件：`dense_rewards`、`score_group` |
| 仅policy训练与behavior KL guard | 同文件：`load_runner`、`behavior_guarded_mean_step`、`policy_step`、`train` |
| 最新GPU4训练参数 | `eclp_runs_meta_llama31_20261007/config.json`、`plan.json`；`model_family_jobs/offline_latent_rl_gpu4.py` |
| 最新family公开生成 | `my_eclp_final/model_family_jobs/generate_family.py`：fresh generation与goal-blind simulator接口 |
| 固定192选择 | `exp_code_ljj/data/test_selection_192.json` |
| Success规则、JSON校验 | `my_eclp_final/seven_methods_192/protocol.py`：`GOAL_RULE`、`goal_prompt`、`validate_goal` |
| judge流程、偏好积分与分母 | `my_eclp_final/seven_methods_192/evaluate.py` 与 `win_rules.json` |
| LatentR3适配预算 | `baseline5_latentRATT/README.md`、`run_latent_ratt.py` |

`CURRENT_METHOD_SUMMARY_ZH.md` 是辅助资料，部分描述沿用旧训练配置；若与最新GPU4 trainer和family plan冲突，以实际源码与plan为准。特别注意behavior hard KL和policy-update batch64。旧在线effect/Jacobian分支存在于大源码中，但不是当前Offline-LPO的active训练方案，不作为当前创新描述。

## 3. 主张—证据状态

| 主张 | 当前证据 | 稿件状态 |
|---|---|---|
| 代码实现误差驱动的连续状态修正 | 已检查模型实现 | 可描述方法机制 |
| reward比较同动作、同历史的完整/屏蔽error状态 | 已检查offline trainer | 可描述目标定义 |
| 后阶段不生成新对话、没有奖励judge调用 | 调用禁用与teacher-forced训练路径 | 可描述训练调用性质 |
| 后阶段总训练时间更少 | 缺同硬件预算实测对照 | 实验计划，不写实测收益 |
| error通道改善接受率 | 缺独立重训no_error对照 | TBD |
| 修正奖励优于NLL-only | 缺相同SFT起点的matched比较 | TBD |
| RL优于相同架构SFT | 缺ECLP-SFT-only matched比较 | TBD |
| 三基础模型均有效 | family训练/生成未全部完成 | TBD |
| 自动judge符合真人判断 | 缺盲人工校准 | TBD |
| 方法能解释真实心理状态 | 无独立心理标注或验证 | 当前不成立，不作此主张 |
| 达到CCF A录用水平 | 会议与完整证据未确定 | 不给录用概率或保证 |

## 4. 写作前必须解决的协议事项

1. 无目标Zero-shot与有目标训练方法的信息差异：披露并补goal-aware untuned控制。
2. 历史Qwen对话的模拟器是否与fresh goal-blind协议一致：未证明前不合并主实验。
3. MA2P为K=0 prompt adaptation：不能写成完整官方复现；补全原版是独立未来实验决策。
4. LatentR3适配的全59,868 RL数据与Our1800不同：列预算，区分native与matched比较。
5. SFT1536与推理6144的上下文差异：披露各actor实际预算及溢出处理。
6. A/B conflict较高：报告交换冲突及human agreement；格式正确不等于评分可靠。
7. 每个指标覆盖分母：保留raw unresolved，并补固定192上下界分析。

## 5. 后续文献补充范围

写作agent继续查阅近期latent-policy/continuous reasoning、反馈误差状态更新、对话opponent modeling与目标接受评价。检索问题应采用公开研究术语，不上传本项目内部未公开文本。主要目的是找最近邻和可区分的机制，而不是收集大量泛LLM引用。

当前文献工作是围绕方法和baseline的定向核验，不是完整新颖性证明。不要用“只找到了这些论文”推出“没有更接近工作”。用户提供的另一篇内部匿名 `AAAI2027_persuasion.pdf` 未用作公开引用，也未把其中架构或结果并入ECLP。
