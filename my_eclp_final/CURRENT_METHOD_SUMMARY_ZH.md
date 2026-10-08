# 当前 my_eclp 方法总结：ECLP-Offline-LPO

更新日期：2026-10-03。本文基于远端实际源码、配置、数据 manifest 和训练状态撰写。

## 1. 当前版本与核心思想

当前方法可表述为：**利用公开反馈的预测误差维护连续对话状态，以隐式策略向量控制公开劝说回复，并通过无需生成对话的离线连续奖励优化策略。**

研究名称采用 ECLP-Offline-LPO；离线训练器版本为 `eclp-offline-lpo-v1`，计划版本为 `gpu4_eclp_offline_lpo_v1`。核心 SFT/执行器源码仍使用 `eclp-elpo-bf16-v2.1` 标记，这是为了保持正在运行的 SFT 和 checkpoint 合同；名称中的 ELPO 不表示当前正在执行旧的在线 RL。

当前实际流程：

```text
原始 ISR 会话 → 1 epoch SFT → 冻结完整 SFT 执行器和参考策略
                                  ↓
1800 条已有参考回复 → 高斯隐式动作采样 → teacher forcing 奖励
                                  ↓
                         仅更新连续策略网络
                                  ↓
200 条 dev 代理退化检查 → 1000 场景五轮生成 → 两个独立 Judge 评分
```

以 `OFFLINE_ADAPTATION_PLAN.md`、`offline_training_plan.json` 和 `offline_latent_rl.py` 为当前 RL 依据。旧 README、FINAL_PLAN、training_plan 中的在线 effect 探测、effect predictor 和三轮真实交互奖励不进入当前流程。旧训练计划中的 SFT 参数仍被继承。

## 2. 模型、模块与参数

| 部件 | 当前设置 | 作用 |
|---|---|---|
| 劝说模型 | `/app/hzj/models/Qwen2.5-7B-Instruct` | 编码公开信息，生成公开回复 |
| LoRA | rank 16，alpha 32，dropout 0 | SFT 中适配语言模型 |
| 反馈状态 | 2 个 slot，每个 256 维 | 压缩公开交互并递推纠错 |
| 冻结特征 | 128 维 | 反馈预测和策略标签对齐 |
| 策略动作 | 64 维连续向量 | 表示隐式回复策略 |
| 策略虚拟 token | 2 个 | 将动作映射为语言模型可读的嵌入 |
| 注意力 | 4 heads | 状态读取与策略上下文聚合 |
| 固定对话对象 | `/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged` | 正式测试时回复劝说模型 |
| 最终 Judge 1 | `/app/hzj/models/Selene-1-Mini-Llama-3.1-8B` | 独立评估生成结果 |
| 最终 Judge 2 | `/app/hzj/models/Qwen3-14B` | 独立评估生成结果 |

网络采用 BF16，不使用量化；高斯密度、损失汇总及优化器主权重与动量采用 FP32。物理 GPU 为 4，通过 `CUDA_VISIBLE_DEVICES=4` 映射为代码内 `cuda:0`。seed 为 42。

配置文件中仍有旧默认 `judge_model=Qwen3.5-9B` 和 effect 参数。最终评估入口 `evaluate_our.py` 指定上述两个 Judge；离线 RL 使用各向同性高斯，不使用 effect 参数学习探索方向。

## 3. 公开反馈驱动的隐式状态

### 3.1 固定的文本特征

特征编码路径使用禁用 LoRA 的冻结骨干；随机投影矩阵固定，投影后进行归一化。公开话语同时提供 token 特征与聚合特征。特征提取不以训练中的私有 reasoning 作为对话历史。

背景初始化状态：以两个可学习 slot 对背景 token 特征作交叉注意力读取，得到初始状态 B₀。劝说目标与公开历史进入回复 prompt，状态初始化本身读取背景。

### 3.2 预测与纠错

模型说出公开劝说话语 y 后，以其特征 f(y) 更新先验状态并预测对方的下一次反馈特征：

```text
B_prior = LayerNorm(B + Transition([B, f(y)]))
f_hat = Predict(mean_slots(B_prior))
```

实际对方回复 u 到达后，读取该回复，并显式计算预测误差：

```text
obs = CrossAttention(B_prior, Project(token_features(u)))
e = ErrorProjection(f(u) - f_hat)
z = concat(B_prior, obs, broadcast(e))
B_post = LayerNorm(B_prior + sigmoid(Gate(z)) * Correction(z))
```

观察监督是 `MSE(f_hat, f(u))`。这里预测的是冻结编码空间中的反馈特征；不能将 B 直接解释成经过标注验证的“信念”“意图”或“接受概率”。

`no_error` 消融保留同一反馈的读取，只把预测误差通道置零；`no_correction` 跳过纠错；`stateless` 以当前回复重新读取状态。它们可用于区分误差通道、反馈读取和递推记忆的作用，当前主流程使用 `eclp`。

### 3.3 梯度隔离

实现维护观察路径 B 与策略路径 policy_B。公开策略损失在 policy_prior 中使用冻结的先验参数，但保留真实输入状态 Jacobian；policy_correct 对反馈读取与预测值截断梯度，允许纠错映射与误差投影接收策略监督。观察路径由预测损失学习；私有 reasoning 辅助路径允许状态梯度。

因此 `isolate_state=true` 表示按路径限制梯度传播，不能写成“SFT 中全部状态模块被冻结”。RL 阶段才冻结整个状态网络。

## 4. 隐式策略如何控制回复

公开 prompt 嵌入后拼接 2 个状态 token。禁用 LoRA 的骨干编码该前缀，最后的状态位置隐向量作为策略上下文 c。策略网络以 2 个查询 slot 聚合上下文，并通过 tanh 输出 64 维均值 μθ(c)。

动作 a 经 `action_expand → SiLU → reshape → action_readout` 映射为 2 个策略 token。状态和策略 token 的 RMS 被归一到基础词嵌入尺度。最终输入为：

```text
[公开 prompt 嵌入] + [2 个状态 token] + [2 个策略 token]
```

Qwen2.5 的 LoRA 执行器基于这些嵌入产生公开回复。隐式状态和策略是连续表示；它们不会作为文本思维链输出。

## 5. 数据：来源、划分与使用方式

### 5.1 原始数据与划分

原始数据目录：`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/data/`。

| 文件 | 规模 | 当前方法中的使用 |
|---|---:|---|
| `isr_sft.jsonl` | 59,868 条，10,000 场景 | 直接用于 SFT，含 ISR 监督 |
| `dialogue_sft.jsonl` | 59,868 条 | 同源公开对话资产；当前 SFT 入口读取 ISR 文件 |
| `user_model_sft.jsonl` | 59,868 条 | 对话对象相关数据资产；当前方法不重新训练 Persuadee |
| `preference_seeds.jsonl` | 2,000 场景 | 划分 1,800 RL train 与 200 dev |
| `evaluation_points.jsonl` | 1,000 场景 | 最终固定测试，原 index 0–999 |

原 manifest 的 profile 为 `paper10k`、seed 为 42。其说明指出来源论文未发布原始 split ID 和 seed，该划分是项目的固定种子重建，不能声称与论文未公开的原始划分完全一致。场景标识使用公开场景字段的 SHA-256 指纹，以避免原始 id 不唯一造成泄漏。

准备目录为 `eclp_runs_gpu4/prepared/`，保存 `sft.jsonl`、`rl_train.jsonl`、`dev.jsonl`、`test.jsonl` 及文件哈希。SFT、preference 和 test 场景集合互斥。dev 从 preference 场景按固定哈希排序选前 200 条，独立于训练 seed。

### 5.2 SFT 样本构造

每个原始记录的历史和当前 instruction/output 被恢复为完整会话。初始 instruction 提取 background 与 goal；assistant 输出拆成：

- `Intent to Strategy Reasoning`：私有教师推理；
- `Selected Strategy`：策略标签文本；
- `Utterance`：公开劝说话语。

历史回放只保留角色与公开 content。当前 `public_supervision=last`：每条 session 的最后一句公开 Utterance 是该样本的生成目标，之前的公开对话用于状态回放。策略和 reasoning 辅助目标也来自该最后一个劝说回合。不得将每个历史回合再计成一个新增生成监督样本。

准备 manifest 记录 59,868 个公开目标，249,720 次历史反馈预测监督，未合成末尾不存在的对方反馈。观察损失在每个 session 内取均值。

### 5.3 RL 样本构造

每条已有 preference seed 含 `scenario`、`history`、`gold_response`、场景/会话标识、`target_turn` 和 `strategies` 等字段。当前 RL 使用已有 `gold_response` 作为 teacher forcing 目标，history 仅使用公开 content。策略列表不作为离线奖励中的排序标签。

此处文件名 preference 不意味着运行 DPO：当前没有 chosen/rejected 文本对训练，也不生成新的 preference 对。gold_response 是项目提供的参考答案，不等同于经过当前两个 Judge 证明成功的最优话语。

RL train 中 1,455 条有历史反馈，345 条没有；dev 中 170 条有反馈，30 条没有。缺少反馈时禁用纠错奖励项。

## 6. 第一阶段：全量 SFT

训练 1 epoch，共 59,868 sessions，梯度累积 8，目标 optimizer updates 为 `ceil(59868/8)=7484`。学习率 1e-4，每 500 步保存 checkpoint，沿用当前训练的 optimizer、RNG 和数据游标。

损失：

```text
L_SFT = L_public_NLL + 1.0 L_observation_MSE
        + 0.1 L_strategy_alignment + 0.1 L_reasoning_NLL
```

公开 NLL 与 reasoning NLL 对目标 token teacher forcing，目标至多 511 个文本 token 加 EOS。策略对齐为策略 token 经投影后的特征与冻结编码策略标签特征的 cosine distance。

私有 reasoning 是训练辅助目标，不出现在公开部署输出中。它可能改善隐式路径初始化，但仍需要 reasoning_weight=0 的对照确定贡献。本方法使用了 CoT/ISR 教师监督，不能声称“无需 CoT 数据”。

SFT 训练 LoRA 与相应状态、策略、映射和对齐组件；基础语言模型权重与固定特征投影不更新。各损失对状态参数的梯度受上述路径隔离约束。

## 7. 第二阶段：离线隐式策略优化

### 7.1 高斯动作采样

完整 SFT 完成后，保存其策略为固定参考 πref，冻结语言骨干、LoRA、状态网络和动作执行映射。只训练 `heads.policy`。

当前行为策略在上下文 c 上产生 μθ(c)，按各向同性高斯采样 4 个候选：

```text
Σ = (0.5² / 64) I
ai ~ Normal(μθ(c), Σ), i=1…4
aref = μSFT(c)
```

每个上下文另评分 1 个固定 SFT 参考动作。采样噪声期望平方范数为 0.25，单维标准差 0.0625；候选采样不再经过 tanh 限幅。当前不学习协方差，也不利用旧 effect predictor 决定方向。

### 7.2 持续奖励

设 ℓfact(a) 为完整纠错状态下参考回复的平均 token NLL，含 EOS。固定公开历史，从最新对方反馈到达前的同一状态分支出发，仅令该次预测误差为零，仍读取同一反馈，构造 ℓcf(a)。后续若有历史回合则按同一公开内容继续回放。两个分支始终使用同一个候选动作 a。

```text
d(a) = ℓcf(a) - ℓfact(a)
R(a) = -ℓfact(a) + 0.1 * tanh(d(a) - d(aref))
```

NLL 项提供连续参考回复拟合信号；它等于 log PPL，避免直接 PPL 的指数放大。纠错项奖励比固定 SFT 参考更能利用预测误差通道的动作，幅度至多 0.1。参考动作 bonus 恒为 0；无历史反馈时 bonus 为 0。

实现提供 `ppl`、`nll`、`correction` 三种奖励模式，当前使用 `correction`。该奖励衡量参考答案拟合及模型内部误差通道利用，是代理目标。它不直接测量真实人的接受、不估计实际因果效果，也不保证最终劝说成功率提升。

### 7.3 优势与优化

优势先计算 `R(ai)-R(aref)`，除以当前收集批 delta 的标准差（下界 1e-3），再裁剪至 ±10。

策略比率使用 **64 维联合高斯动作密度**，log density 按维度求和。动作、协方差和 old_logp 在更新中固定并 detach。目标采用 PPO clipping，clip=0.2，KL 正则系数 0.02；KL 为相同协方差下当前策略与 SFT 参考的动作分布 KL，阈值 0.1。它不是文本输出分布 KL。

1800 场景完整通过一次，每收集 64 场景后更新策略，下一批使用更新后的策略重新采样。每批最多 2 个策略更新 epoch，update batch size=1，学习率 1e-5。KL 超限回滚权重和 optimizer，学习率减半重试，最多 7 次，最低 1e-7；参考 KL 已不满足约束时停止诊断。

### 7.4 计算预算与禁止在线调用

总答案评分次数为 `1800×5 + 1455×5 = 16275`，另有公开历史编码、状态回放和策略上下文计算。每个答案评分使用单条 teacher forcing 前向，不进行自回归回复生成。

`OfflineBridge` 与禁止 generate 的接口使训练中 Persuadee、Judge、自回归 generate 调用立即报错。省去真实对话采样能降低开销，但长历史回放与 7B 骨干前向仍有成本。

当前 `score_batch_size=1`。已有预检发现 BF16 批量评分路径与单条路径约 0.013 的 NLL 差异，可能影响很小的 latent 奖励差，因此暂未启用批量答案评分。

## 8. dev 检查与 checkpoint

RL 每批提交后保存策略、optimizer、RNG、数据游标和奖励分项。RL checkpoint.step 表示已提交场景数；SFT step 表示 optimizer 更新数，两者语义不同。

完整一遍 RL 的最后提交 checkpoint 是预先规定的最终模型，不用测试集选模型。200 条 dev 比较最终策略与固定 SFT 参考的平均 NLL；若恶化超过 0.05，则停止后续测试并诊断。该门槛只能排查代理任务明显退化。

应检查策略实际改变、冻结执行器哈希不变、参考策略不变、优势非零、训练在线调用次数为 0。数值保护、哈希、断点恢复和审计是实现可靠性措施，不能直接列作研究创新。

## 9. 推理使用的数据与完整过程

推理的模型输入仅为场景 background、goal、已有公开对话，以及随后真实生成的公开回合。测试点中的 reference 被保存在结果中用于后续评估对照，不用于选动作或构造回复；私有 reasoning、Selected Strategy、gold_response 不参与部署控制。

对每个测试场景：

1. 用背景初始化状态，回放原公开 history；不同场景重新初始化。
2. 根据公开 prompt 和当前状态得到策略均值，默认直接采用 mean action，不作多候选 Judge 筛选。
3. 将状态和动作转为虚拟 token，用 Qwen2.5 生成一句公开劝说回复。
4. 把刚生成的话语纳入状态，预测对方反馈特征。
5. 固定 Persuadee 根据实际公开对话产生回复。
6. 用真实回复特征计算预测误差并纠错状态。
7. 重复到完整 5 轮，保存 10 条新增角色交替话语及初始历史。

每轮动作依据当前状态重新计算。**mean action 不等于确定性文本解码**：公开回复仍以 temperature=0.95、top_p=0.7、top_k=50 采样，最多 512 个新 token。使用场景级种子以支持断点和顺序变化下的复现。

公开 prompt 最多 1536 token，过长时删除最旧成对交互，保留背景、目标和最近反馈；状态回放会消费已提供的公开历史。特征上下文上限 1536，特征回复跨度上限 256。正式生成阶段不调用 Judge，不提前因 Judge 成功而终止五轮。

## 10. 最终评估与保存位置

两个 Judge 分别评估同一批固定 1000 场景。Selene 使用 strict V17 v7 协议；Qwen3-14B 使用固定 V11 neutral JSON 协议。Selene 按轮前缀判断成功，Qwen3 的联合协议返回五轮判断和两个顺序下的胜者，具体解析遵循 `final_eval_protocol.py`。

- Success：5 轮内至少一次对方反馈满足目标接受标准。
- Win-Rate：当前方法与原 reference 比较，交换 A/B 两次，每条信用为两次“当前方法获胜”指示的平均值，以减轻位置偏差。
- 平均轮数：首次成功轮次；未成功计 5。

表格分母采用成功与胜负均解析完成的记录数；同时报告未解析数量及全部 1000 条分母的成功率上下界。Judge 输出格式失败不自动当作模型失败或成功。

结果写入 `/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/results/main_experiment_table.md` 中 Qwen2.5 对应的两个 Judge 分组的 Our 行。保留其他方法结果、原始生成和 Judge 输出。正式性能结论须等生成、双 Judge 和完整审计完成。

## 11. 相对参考论文的适配与创新点

参考论文：ICLR 2026《Reinforced Latent Reasoning for LLM-based Recommendation》，文件 `baseline5_latentRATT/ICLR-2026-reinforced-latent-reasoning-for-llm-based-recommendation-Paper-Conference.pdf`，主要借鉴 §3.2.2 公式 (5)–(8) 的高斯 latent、参考答案 PPL 连续奖励与冻结骨干的优化思路。

| 维度 | 来源思想 / 当前适配 |
|---|---|
| 任务 | 从推荐答案适配到公开多轮劝说 |
| 状态 | 引入对实际对方反馈的特征预测与误差驱动递推纠错 |
| SFT | 使用原 ISR 推理辅助蒸馏，公开部署只输出回复 |
| 奖励 | 负平均 NLL，加同历史误差通道干预下的参考中心化有界增益 |
| RL | 在连续动作联合高斯密度上优化固定执行器的策略 |

当前最值得研究的主张：

1. **预测误差驱动的交互状态**：把最新反馈的意外程度作为状态更新通道，而不是仅用额外 token 承载历史。
2. **面向反馈利用的连续奖励**：同历史、同答案、同动作，仅干预最新误差通道，结合固定参考中心化与有界增益。
3. **状态—策略—执行器分工**：SFT 学习状态及执行能力，离线 RL 专注连续策略，部署时无显式策略文本和思维链生成。

高斯 latent、PPL 奖励、PPO clipping、LoRA 和冻结骨干本身均有已有工作基础，不能单独作为原创贡献。当前联合动作密度优化与原论文 token-ratio LR-GRPO 不同，不能声称逐行复现。私有 ISR 蒸馏也是监督适配，须与已有蒸馏工作比较。

## 12. 当前证据、局限与后续研究

2026-10-03 最近读取快照：SFT 日志 step=2415/7484，epoch=0，next_position=19320；流水线 phase=sft_adopted。最近提交 checkpoint 为 2000。日志步数与已提交断点不同，本文不将未保存更新视为可恢复 checkpoint。此快照不是实时状态。

机制诊断与正式性能实验不同。目前完整 SFT、正式离线 RL、1000 场景生成及双 Judge 结果尚未全部完成。不能据此宣称方法有效、优于 SFT 或达到 CCF A 录用标准。

主要局限：参考回复 NLL 可能偏向已有措辞；模型内部干预可能被奖励利用；状态压缩或历史截断可能损失信息；RL 在已有单个参考回复上的提升未必迁移到五轮闭环；两个模型 Judge 的评分不能等同真实用户接受。

建议按优先级补足：

1. 相同 SFT checkpoint 的 SFT-only 与 RL 对照，确定 RL 是否增益。
2. 匹配监督、参数和预算的 `no_error`、`no_correction`、`stateless` 状态对照。
3. reasoning_weight=0 消融，分离 ISR 蒸馏收益。
4. PPL / NLL / correction 奖励比较，同时报告真实 Success、Win-Rate、NLL 和纠错分项。
5. 奖励增益与独立成功判断的相关性、多 seed、配对置信区间和泛化分析。

上述属于研究建议，不表示本文已经执行额外消融或修改当前正式训练。

## 13. 可追溯文件

远端方法目录：`/app/hzj/liujianjian/ljj-2-paper/my_eclp_final/`。

| 文件 | 作用 |
|---|---|
| `eclp_elpo.py` | 状态、策略、SFT、checkpoint 和生成 |
| `offline_latent_rl.py` | 离线奖励、策略优化、dev 检查 |
| `config_gpu4.json` | 模型与结构配置 |
| `offline_training_plan.json` | 当前离线实验规模和超参数 |
| `run_gpu4_offline_pipeline.py` | SFT 接续、RL、验证、生成和评估串联 |
| `evaluate_our.py` / `final_eval_protocol.py` | 双 Judge 协议与指标 |
| `audit_gpu4_completion.py` | 完工数据、奖励、哈希与结果一致性审计 |

核对时本地 remote_patch 与远端以下文件 SHA-256 一致：

```text
eclp_elpo.py
8c5402f92e5cbd8e1a7ed2e63fab175c5e00c91a983251fce4ddaadfde3b9d14
offline_latent_rl.py
a2786a828570a5767bf71c8a238edff0517795a38be6cb988a02f38f1863fb8c
offline_training_plan.json
77f7c6f064b1b6074d1e23f9e0f813c83593fa8556aafb62fcd97a05894d6dec
config_gpu4.json
de3c445815b0d15830d0eab501e6fda166c6089f57cb5959c434d8f4c763cb44
```

本文总结当前已部署实现，不新增训练数据、不修改运行中的模型代码或训练超参数。
