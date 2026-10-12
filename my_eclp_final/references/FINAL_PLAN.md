# 最终研究方案：ECLP + ELPO

**论文题目：Understand Before You Persuade: Evidence-Corrected Latent Reasoning for Dialogue**  
**中文定位：通过反馈纠正理解、通过交互效果学习策略的隐式劝说。**  
版本：v2.1，2026-10-03。本文是本代码包唯一的研究方案；旧文件保留作设计过程与审计来源，不再并行执行。

> 一个判断要能被对方的新反馈纠正，一项策略要按执行后的效果学习。我们将这两件事分别落实到 ECLP 的连续状态更新和 ELPO 的连续策略优化中。

当前交付包括实际核心代码、训练与评估命令以及测试。它不是已经取得性能提升的实验报告。CCF-A 投稿应由匹配对照、机制证据和实际效果支撑，不能由方法名称或未运行的预期数字代替。

## 1. 先把旧文件里的分歧定下来

本节是本次明确作出的整合决定，不将互相冲突的旧描述说成原本一致。

| 旧文件中的不同说法 | 最终决定 |
|---|---|
| `01_research_proposals(1).md`、`两个方案.md`：两条独立论文 | 合成一篇，围绕同一个多轮交互闭环，不再分别包装两套系统。 |
| `两个方案应该合成一个论文.md`：ELPO 可全部作为附录扩展 | 采用较新的《最终方案.md》定位：ECLP 是结构核心，ELPO 是完整方法的训练算法。若主表使用 ELPO，正文必须交代其采样、奖励和更新。附录仅放推导、扫描和更多实验。 |
| 早期 ELPO：严格低秩、tanh 有界扰动 | 采用《最终方案.md》的低维动作 + 全秩混合高斯协方差。不要同时混用两种动作定义或密度公式。 |
| 有的文字把 ECLP 等同于 SFT | ECLP 在 SFT、RL、推理阶段均存在；SFT 只是初始化，ELPO 改善其中的策略参数。 |
| 早期协议建议换 Persuadee 做迁移 | 服从之后确定的约束：训练、开发、测试均使用原 Persuadee，不另换模型。稳健性先靠固定模拟器下的不同场景、训练种子、独立 Judge/人工复核。 |
| 部分伪代码忽略参考 PPL 与实际交互的区别 | interaction success 是主奖励；参考 PPL 以默认 0.1 的有界 shaping 项辅助训练，仍单独报告 PPL control。 |
| 上版代码默认比较第五轮末成功 | 本次核对原 `evaluate_baselines.py`：原主指标是五轮内曾成功。因此主指标锁定为 **Success-within-5**，第五轮末仍成功作为稳定性指标。代码默认 compare=ever；RL 使用同定义的 within-H。 |

保留原名称，避免再次引入第三套缩写：**ECLP（Evidence-Corrected Latent Planning）+ ELPO（Effect-Aligned Latent Policy Optimization）**。外部叙述可称 ECLP 框架，方法章节包含 ECLP 结构与 ELPO 训练算法。

## 2. 两个直观问题，也是两个核心贡献

### 2.1 对方已经纠正了你，你还在使用旧策略

例子仅用于解释，不是数据集中的已验证案例。目标是“让对方参加一次社团活动”，模型以为对方害怕公开表演，于是不断鼓励。对方回复：“不是害怕，是周五要打工。”有效决策应更新为时间约束，在原目标允许的范围内协商安排，而不是继续鼓励。如果目标明确要求周五参加，不能擅自把目标改成周末以制造成功。

研究问题是：**连续隐式状态能否利用刚收到的反证修正，并让下一步策略发生相应改变？**

贡献一是**带行为条件反馈预测的跨轮状态校正**，而不是给几个隐藏向量命名为 belief/desire。状态有预测责任、时间更新规则以及与控制目标分开的梯度路径。

### 2.2 更像参考答案，未必更有效；向量不同，未必策略不同

原 LatentR³ 的负目标 PPL 适合评价“在这个 latent 下，参考答案有多容易被预测”，并免去逐个生成答案的开销。但劝说的评价对象是对方执行后的回应；同一问题通常允许多种有效话语。[R1]

研究问题是：**在对方当前状态相同的情况下，如何把有限的探索预算用在会改变交互效果的策略上？**

贡献二是**状态保持的效应导向连续策略优化**：固定本决策点的状态，学习反馈敏感的探索方向，实际执行候选，然后按同上下文增量更新已采样动作的概率。

两个贡献之间的接口只有一个：**ECLP 形成当前状态，ELPO 在该状态条件下学习策略。**不额外引入长程树搜索、另一套心智大模型、在线 Judge 候选择优或新的人工心理标注集。

## 3. 与来源论文、近邻工作的区别

来源论文给我们的是“先产生连续中间向量，再生成公开答案；先 SFT，再 RL”的技术接口。原文 LatentRATT 从 LLM 最后隐藏层经过 attention 产生连续 token；SFT 中已经有这条路径，并不需要人工给定正确 latent。[R1，§3.1–3.2]

本次外部核验还找到 **Factorized Latent Reasoning（FLR）**：它已经将推荐中的 latent 拆成多个因素，用多因素 attention 和正则化进行专门化。[R3] 这不是旧附件中已有的比较，是本次补充的相关工作。因此，不能再把“一个向量改成多个向量”当作主要新颖性。

| 已有技术 | 不单独申报为创新 | 本文必须证明的增量 |
|---|---|---|
| LatentR³ | attention latent、SFT→RL、高斯、PPL | 面向反馈变化的状态更新与交互策略学习，而不是任务名迁移。 |
| PersuGPT / DailyPersuasion [R2] | 意图到策略、用户模拟、偏好优化 | 连续状态能被证据纠正，策略优化对象与实际采样动作一致。 |
| FLR [R3] | 多因素 latent、多 attention、因素正则 | **时间上的预测—校正**与状态/控制的功能分工；不能由普通多槽位控制解释。 |
| GTS [R4] | 条件高斯 latent sampler | **由对方反馈确定探索方向**，并以真实执行回报区分好坏。 |
| MindZero [R5] | 观察行为学习无标注心理推断 | 反馈校正对后续策略的作用，以及与控制优化隔离的必要性。 |
| PPO [R6] | clipping、KL、固定行为动作比值 | 这些是正确实现的基础工具，不是新算法名称下的首创。 |

本次是针对性近邻核验，不是穷尽性“全球首次”证明。正式投稿前继续检查近邻，并将对应可适配方法纳入强对照。预期贡献有清晰研究对象，但是否成立取决于后面的证据矩阵。

## 4. 最终结构：一个骨干，两个连续计算阶段

```text
公开背景、已经观察到的对话
       ↓  冻结基础骨干，关闭生成 LoRA
上轮状态 + 上轮真正说出的话 → 预测对方反馈
       ↓  收到对方实际回复，计算预测误差
ECLP：校正当前状态 B_t（默认 2 个槽位）
       ↓  B_t 投影成连续输入，再进入基础骨干
目标 g + 当前状态 → 策略均值 μ_t
       ↓  SFT / 主推理使用均值；RL 从动作分布采样 a_t
策略投影 P_s(a_t) → S_t（默认 2 个槽位）
       ↓  同一骨干 + 公开生成 LoRA
公开劝说话语 y_t
       ↓
固定 Persuadee 的实际回复 → 下一轮状态校正
```

不是两个额外 7B 模型。新增部分是状态 attention、门控转移/校正 MLP、策略 attention、动作均值头和 embedding 投影。初始小网络宽度 256，状态 2 槽位、策略 2 槽位，动作维度 64，反馈特征维度 128；这些是配置起点，不是经过实验选出的最优参数。

B 和 S 不是可读思考句子，也不是词表中的特殊“思考词”。它们是连续向量，通过 `inputs_embeds` 进入骨干。从 SFT 第一次前向开始，这些中间计算已经存在。

**隐式计算存在是代码事实；它是否支持心理状态理解，是待检验的能力。**论文优先称它为“任务相关的预测性对方状态”，不声称恢复了真实人的心理。

## 5. ECLP 的具体定义

设 h_t 为已观察公共历史，g 为私有任务目标，u_t 为刚收到的对方话语，y_{t-1} 为上轮实际发言。E 是固定的基础骨干语义编码方式，仅池化回复区段。

先通过自己的实际行为预测状态与反馈：

\[
B_t^- = T_\phi(B_{t-1},E(y_{t-1})),\qquad
\hat e_t=D_\phi(B_t^-).
\]

再读入新反馈并校正：

\[
e_t=E(h_{t^-},u_t),\quad \delta_t=e_t-\hat e_t,\quad
O_t=\operatorname{CrossAttn}_\phi(B_t^-,H_t^{obs}),
\]

\[
B_t=\operatorname{LN}\left[B_t^-+
\sigma(W_g[B_t^-;O_t;W_\delta\delta_t])\odot
C_\phi([B_t^-;O_t;W_\delta\delta_t])\right].
\]

“预测不一致”通过显式残差通道提供给校正器；门控是否真的学会纠正仍需实验证据。首轮从公共背景初始化；中途测试点按已有公开历史重建，不将其伪装成对话第一轮。每条真实反馈消费一次，不把预测反馈写进真实历史。

然后把状态送回基础骨干，生成策略：

\[
H_t^{B}=F_0([Emb(x_t);M_B(B_t^{\pi})]),\quad
\mu_t=\operatorname{StrategyAttn}_\psi(H_t^{B}),\quad
S_t=P_s(\mu_t),\qquad B_t^{\pi}\overset{\mathrm{value}}{=}B_t.
\]

\[
y_t\sim p_\theta(\cdot\mid x_t,M_B(B_t^{\pi}),S_t).
\]

x_t 含公开生成需要的背景、目标与历史。状态编码不直接读取当前私有 g；策略与公开生成读取 g。背景可能已含目标相关事实，过去的劝说也已公开，因此不宣称 B 与所有目标信息统计独立。

### 5.1 梯度分工

实现中保留两条数值相同的状态路径：`B` 是观察与私有辅助监督路径，`policy_B` 是公开策略路径。训练时，transition、prior normalization、feedback predictor、slots、observation readout 和 `to_memory` 的参数不接收 public loss 梯度，而 correction、gate、posterior normalization 以及 ECLP 的 error projection 仍可接收公开话语/策略损失。policy prior 使用同一组冻结参数的真实计算，保留对历史校正状态的真实导数；不采用 BF16 差值相加或 identity straight-through 近似。观察 MSE 仍通过 `B` 训练预测分支；`no_error` 只移除误差通道，`no_correction`/`stateless` 不产生 correction 梯度。

远端版本保留 ISR reasoning 辅助 NLL，它通过完整 `B` 路径训练整个状态网络，包括反馈预测器。因此本版本的准确主张是“公开损失与反馈预测参数隔离”，不能声称预测器仅由观察 MSE 更新，或状态学习完全不使用教师心理分析。消融须保持相同 reasoning 权重，另报告 reasoning_weight=0 控制。

smoke 必须逐参数报告并检查这份契约：public loss 对 predictor/readout 前缀的梯度为零；ECLP 的 correction 核心和 error projection、`no_error` 的 correction 核心有非零梯度；观察损失有非零状态梯度。`B.detach()` 只是其中一部分。冻结骨干权重不意味着把依赖 S 的答案前向放到 `no_grad()`，否则会切断 S 的学习。

本实现仍让公开 decoder 读取历史，避免将全部语言信息硬压入几个槽位。由此也存在绕过 latent 的可能，必须用干预与匹配对照测量，而不是声称架构已强制完成所有心理推理。

### 5.2 本次补上的必要消融

上版 `no_correction` 完全不吸收新反馈，会把“有无误差信号”混成“看不看新信息”。保留它作为极端对照，但新增 **`no_error`**：仍有新反馈 attention 和门控更新，只将预测误差通道置零。

因此，本次优先比较 `eclp` 与 `no_error`，直接检验预测误差是否有超出普通反馈读入的贡献。`stateless` 则不保留递推先验，但其回复特征本身含公共上下文；不能误称“完全没有历史”。

## 6. Stage 1：监督初始化

采用固定 ISR 文件，公开主目标是每条 session 的最后一条 Utterance。数据提供 `Intent to Strategy Reasoning / Selected Strategy / Utterance`，并没有独立可信的 belief/desire 真值。[S2]

四个损失为：

\[
\mathcal L_{SFT}=\mathcal L_{utt}+\lambda_o\mathcal L_{obs}+\lambda_s\mathcal L_{strat}+\lambda_r\mathcal L_{reason}.
\]

| 损失 | 标签 | 更新对象 |
|---|---|---|
| 公开话语 NLL | 最后一条 Utterance | 策略 attention、动作投影、状态读出投影、公开 LoRA，以及 `policy_B` 的 correction/gate/error projection；不更新 feedback predictor/readout。 |
| 行为条件反馈预测 MSE | 同一真实会话中已经记录的下一条对方回复的冻结特征 | 状态读出、转移、反馈预测与校正参数。 |
| 策略余弦对齐 | 已有 Selected Strategy 文本的冻结特征 | 独立策略对齐头与策略 latent；不让公开 decoder 输出分析标题。 |
| 私有 ISR reasoning NLL | 已有 Intent to Strategy Reasoning | 完整 `B`、状态读出、策略与公开 LoRA；仅作为私有辅助目标，推理时不生成该字段。 |

观察监督在历史真实相邻话轮上计算；当前末尾 Utterance 没有后续回复就不补造。后续观察预测损失训练前面的校正器。不会用未来回复生成当前状态或策略。

默认 λ_o=1、λ_s=0.1、λ_r=0.1。只是起点，不靠测试集挑权重。第一阶段不做 RL，也不需要调用新的 Persuadee。策略与 reasoning 标签属于已有教师弱监督，因此不能宣称训练完全 CoT-free；准确说法是“无需新增心理标签，部署不生成显式思考文本”。

### 6.1 数据曝光与本代码的边界

代码保留 session-last 公开目标和 session-mean 观察损失；多条原 session 中重复出现的历史前缀仍可能重复受监督。`prepare` 同时报告总反馈曝光与唯一公共反馈前缀数，不把重复前缀当新独立样本。所有 matched-SFT 对照应复用同一视图与权重；本版没有悄悄重采样或重加权原数据。

## 7. Stage 2：ELPO，固定状态、探索策略

同一决策点的 B_t 固定，策略变量 a 可变：

\[
a_i\sim q_\psi(a\mid z_t),\qquad S_i=P_s(a_i).
\]

z_t 来自当前冻结公共上下文与 B_t。固定的是本组候选的初始状态值；分支真正执行后，新回复仍通过 ECLP 更新各自的状态。冻结状态网络权重不等于停止状态变化。

### 7.1 用对方反馈学习探索方向

在训练场景上执行候选，记录 a、公开 y、对方实际 u。训练小型 F：

\[
F_\eta(z,a)\approx
\mathbb E[\bar E(h,u)\mid z,a]-\mathbb E[\bar E(h,u)\mid z,a_{ref}].
\]

**本次对上一版作了一个明确修订：F 的目标编码固定为所有分支共同的动作前历史 h，只池化 u；不把候选 y 作为变化的编码上下文。**否则，即使对方说出同一句 u，编码器也可能因为刚才的 y 不同输出不同向量，将自己的措辞变化误当作对方反馈变化。

此修订只改变效应监督的测量，不改变 Persuadee 的输入。Persuadee、状态更新器和 Judge 仍读取各自完整的实际对话。代价是短句“可以/不行”在不含候选 y 的固定上下文下可能丢失部分指代，故该特征是保守的反馈代理，不代替完整轨迹的目标判断。

F 只决定从哪些方向探索，不给最终成功奖励，也不作为真实回复写进历史。

### 7.2 全秩、同动作能量的效应协方差

在固定 SFT 参考动作处，对小网络 F 求 Jacobian，取主要 k 个右奇异方向 U：

\[
J_t=\left.\frac{\partial F_\eta(z_t,a)}{\partial a}\right|_{a=a_{ref}},\qquad U_t^TU_t=I_k,
\]

\[
\Sigma_t=\epsilon_a^2\left[(1-\alpha)\frac{I_m}{m}+\alpha\frac{U_tU_t^T}{k}\right],\quad 0\le\alpha<1,
\qquad q_\psi=\mathcal N(\mu_\psi(z_t),\Sigma_t).
\]

默认 m=64、k=8、α=0.5、ε_a=0.5。保留全空间分量，避免只能相信当前不成熟的 F。代码同时实现 `isotropic / random / effect`。

同 ε_a 保证 a 空间的噪声能量一致，但经非线性 P_s 后不保证策略 token S 的扰动一致。因此实际记录 `strategy_noise_rms_vs_mean`；用开发集校准尺度后再冻结比较，不能只宣称“同 sigma 所以公平”。

局部恒等式 E||Jδ||²=tr(JᵀJΣ) 只解释反馈敏感方向，不证明策略最优、心理维度可解释或成功率必涨。随机方向、语法异常和有害反馈都可能产生大变化，最终仍由执行回报决定好坏。

### 7.3 与原评估口径对齐的奖励

原代码主成功率是五轮内曾明确达成目标，不是只看第五轮末。[S4] 本版训练在 H 轮固定窗口内采用相同聚合：

\[
R_i=V_i\cdot\max_{1\le r\le H}J_{accept}(g,h_{i,r}).
\]

V_i 当前代码只检查公开格式有效性；**它不是诚实性、隐私或尊重拒绝的完整安全检测器**。这些质量维度要单独盲审，不把 regex 通过写成安全通过。

每个 prefix 使用锁定的 Judge 与 rubric。若已知某轮成功，则 within-H 已知为真；若没有已知成功但有未解析 prefix，则结果是 unknown，不能强行写成失败或丢弃。所有 H 轮都实际生成，不通过提前停止改变与原基线的生成协议。

这相对于上一版“只 Judge 第 H 轮末”增加了 Judge 调用；是本次任务口径修订，必须在预算中明确统计。另报 FinalRound5Success，检查早期接受是否在后续撤回，避免把短暂接受解释成稳定效果。

### 7.4 同上下文配对，真实连续动作归因

每个候选重复执行 M 次，同时执行固定 SFT 参考分支：

\[
\Delta_i=\frac1M\sum_j(R(a_i;\omega_j)-R(a_{ref};\omega_j)),\qquad A_i=\Delta_i/s_{past}.
\]

动作采样与对话执行使用独立 RNG；配对执行种子不含候选编号。同一场景在分片、排序和续跑后保留相同种子。重复用于估计策略效果对措辞/模拟器波动的敏感性；配对是否降低方差需测量，不能自动保证。

首步使用 a_i，之后使用同一冻结 continuation policy，分支状态随实际反馈更新。这里归因的是首步决策的 H 轮结果，不假装已经实现完整多步轨迹 policy gradient。

优化固定记录下来的动作：

\[
\rho_i=\exp(\log q_\psi(a_i\mid z_t)-\log q_{old}(a_i\mid z_t)),
\]

\[
\mathcal L_{ELPO}=-\mathbb E\min(\rho_iA_i,\operatorname{clip}(\rho_i,1-\epsilon_c,1+\epsilon_c)A_i)
+\beta D_{KL}(q_\psi\|q_{ref}).
\]

密度对动作维度求和，不平均。a_i 在更新时 detach，不能随新均值重采样。P_s 不必可逆：密度定义在 a 空间，动作执行映射必须固定。KL 是动作分布 KL，不等于公开文本 KL。

一次 rollout/update 中，基础骨干、公开 LoRA、状态网络、B→embedding、P_s、F、Σ、continuation 都固定，只训练策略均值网络。代码检查执行器哈希与行为密度；同时在实际 step 之后检查 KL，越界恢复策略权重和 Adam 状态。该回滚是实现保护，不是新的最优性定理。

### 7.5 PPL 对照必须准确命名

`collect --reward ppl` 使用负参考 NLL 的指数得到 -PPL，作为独立 action-density PPL control；正式 interaction 分支则在 Judge success reward 上加入默认 0.1 的 tanh 归一化 PPL shaping。两者都使用本框架的连续动作密度，因此不是原 LR-GRPO 的逐行复现。

原 LR-GRPO 对照应继续使用原项目的 `latent_model.py` / `run_latent_ratt.py`，先完成独立格式修复和依赖恢复。不能用新动作密度算法冒称原论文实现。

## 8. 实验设计：什么证据才能支撑高水平投稿

### 8.1 先完成主因果区分，不靠模块数量撑贡献

| 阶段 | 必须比较 | 排除的解释 |
|---|---|---|
| 修复层 | 原 v0、格式修复 v0、统一 loss mask/cropping 的 v0 | 收益只是修复原系统。 |
| 结构层 | 普通同容量 latent + 同观察/策略监督；ECLP；no_error；stateless | 收益只是更多参数、更多标签、额外历史。 |
| 回报层 | 同一 ECLP 的 PPL 控制、isotropic interaction RL | 收益是否来自实际反馈，而非另一种模仿。 |
| 几何层 | 同回报、同调用量、同尺度的 isotropic、random、effect | 收益是否来自效应方向，而非单纯噪声或更大扰动。 |
| 强对照 | 已复现 MA2P、ISR-SFT+SimDPO；同预算文本策略 RL | 不能只赢一个有缺陷的旧 latent。 |

保留核心 2×2：普通匹配 latent/ECLP × PPL/交互效果优化。全文主结论不能只靠“原单 token vs 全部升级完整模型”的一次比较。

代码已实现 ECLP 主方法与 no_error/no_correction/stateless、PPL action control 和三种探索。**严格同参数普通多槽位 + 同辅助监督、文本 RL、PCA 方向及完整人工干预评估不是本包已经实现的基线；不能将 stateless 偷换成它们。**这些是正式论文下一阶段必须补齐的实验对象，原基线项目仍为其实现起点。

### 8.2 ECLP 的三个机制问题

1. **预测是否有增量？**留出会话上对方回复预测优于无状态的行为条件预测器、同容量历史编码器，不能只看训练 MSE 下降。
2. **遇到反证是否纠错？**预先标注“澄清旧误解/提出新约束”子集，盲审下一句是否回应新约束；不是事后只挑完整模型赢的案例。
3. **状态是否影响策略？**固定公开历史与 RNG，比较校正前后状态；使用语义兼容的状态干预，而非任意乱向量导致分布外崩坏。单纯 goal-change 下 B 不变属于接口测试，不属于心理能力结果。

### 8.3 ELPO 的三个机制问题

1. F 在**完全不参与 F 拟合与选 epoch 的训练内部场景**上能否预测实际回复变化？`check-effect` 对源场景交集直接报错。
2. 同样幅度的策略扰动，effect 是否优于随机方向？先比较真正有意义的策略差异，再比较目标结果，不能把“反馈越不同越好”当成功标准。
3. 同累计调用预算下是否提高 Success-within-5？效应探测、参考分支、重复、失败尝试、Judge 重试全部计入，不能只比较 epoch。

### 8.3.1 当前创新质量的提升优先级

预测—校正与残差门控已有滤波、预测状态和递归网络思想基础；仅给现有模块命名不足以支撑 CCF-A。本文应将增量限定为连续推理中“行为条件的反馈预测、按真实反证纠正、对后续策略产生可验证作用”的组合，并以匹配对照证明必要性。

1. 在当前实现内固定公开历史与真实回复，单独置零/打乱预测误差、交换状态，检查下一步动作、参考 NLL 与真实交互回报；区别直接历史读取与 latent 的额外作用。
2. 使用真实对话中约束/拒绝原因发生变化的场景，另配未发生反证的场景；报告错误先验的纠正方向及后续策略适配，不把大 residual 当作心理理解证明。
3. 明确区分教师 reasoning 蒸馏和反馈建模：reasoning_weight=0 控制、同标签的普通 latent 对照，保留同公开 token 和观察曝光。
4. 在新训练场景中检查 F 的预测与方向稳定性，再以相同探测和 Judge 预算比較 isotropic/random/effect 的真实回报。PPL shaping=0 控制用于检验收益是否依赖参考答案模仿。

上述实验尚未完成；本轮只完成代码缺陷修复与集成验证，不对中稿概率或成功率给出虚构数字。

### 8.4 固定项与统计

保持原 1,000 测试点、五轮、公开上限 512、temperature=0.95、top_p=0.7、top_k=50、同 Persuadee 权重/tokenizer/prompt。主 Judge/rubric 在运行前锁定，结果未知不能丢弃后换分母。主指标 Success-within-5；补充 FinalRound5Success、格式泄漏、token、延迟、显存和首次成功轮数。

当前 1,000 点已在前期开发中被看过，不宣称全程盲测。开发调参只用预先固定的开发场景；尽可能三个训练 seed，给场景级配对区间与 seed 波动。多个 Judge 评同一场景不是多个独立数据点。独立 Judge/人工复核对同一生成轨迹执行，**不更换 Persuadee**。

不预填新结果。支持论文主张的标准是：匹配强对照有一致效果，预测—纠错—策略作用证据方向一致，质量代价可接受。若只在一个 Judge 提升或几何消融无差异，就收缩相应主张，而不是再堆模块掩盖归因。

## 9. 执行路线与预算

### P0：文件、环境、协议

2026-10-03 远端只读审计通过，5 个数据文件均为完整内容且 SHA 匹配 manifest。结果见 `eclp_runs_v21_repair/source_audit.json`。首次新 lock 哈希完整模型和 tokenizer；只改资源放置不改模型身份。

### P1：真实 GPU 冒烟

执行 smoke：BF16 实际前向/反向、逐参数验证公开损失只训练校正分支、观察及私有 reasoning 损失有状态梯度、一次真实生成、一次固定 Persuadee 回复、反馈状态更新。输出真实日志与显存。CPU 测试不能替代这一步。

### P2：ECLP 小训练

固定 32 个 SFT 场景，先 100 optimizer steps，最多续至 200；不以这个 pilot 的成功率估计全量收益。检查公开输出、latent 梯度、状态预测、断点恢复及开发集 NLL。dev 先固定 32 场景，正式共 200 场景。

### P3：结构是否有独立价值

逐步扩大训练子集，先跑 eclp、no_error、stateless，再补同容量同监督普通 latent。seed/公开监督 token/反馈曝光一致；必要时单独校准各结构学习率并记录相同搜索预算。

### P4：先探测，再启用效应方向

在 rl_train 的固定前 128 个上下文采集各向同性探测，G=4、M=2、H=1。拟合 F，内部按场景划分训练/选择集。再用 offset=128 的新 64 场景做 check-effect；这些场景不能出现在 F 的 fit 或选择集。

之后在预先固定的新场景做等尺度 random/effect 实际比较，才决定是否扩大 effect RL。零预测器 gate 是最低可训练检查，不足以单独证明方向有效。

### P5：同预算 RL 与正式测试

同一 SFT checkpoint 分别产生 isotropic、random、effect 分支；未更新的行为 checkpoint 与一次 rollout 必须匹配。先 H=1 后 H=3。对每一条已分配训练路径预先设定上限，不给效果差的分支无限补预算。

采集 N 个上下文、G 个候选与一个参考、M 次重复、H 轮：Persuadee 调用为 N(G+1)MH。当前 within-H Judge 对每轮 prefix 评分，基础 Judge 调用也为 N(G+1)MH，另外加解析重试。

例：N=128、G=4、M=2、H=3 时，每类各 3,840 次；首次 H=1 时每类各 1,280 次。F 拟合与 check-effect 本身不调用 Judge，但其输入探测需要真实 Persuadee 调用。账本先记 attempted 再发请求，中断未完成记录不能按零成本处理。

### 数据划分的具体实现

原 SFT 10,000 场景和原 test 1,000 点不变。代码从已有 preference 2,000 场景固定预留 200 作 dev，余下 1,800 作 rl_train。这是**训练内部派生视图**，未修改原始文件；因此实际 RL 不应写成用了全部 2,000 场景。SFT 59,868 的数字目前来自源清单，完整内容恢复后 prepare 会逐条核验。

## 10. 实现、版本与不承诺的部分

全核心在 `eclp_elpo.py`，无必须补写的模型 stub。支持数据校验、SFT/续训、配对交互、离线 Judge、F 拟合/新场景检查、连续动作 PPO、生成、离线评估与配对 bootstrap。依赖原项目的模拟器与提示词实现，刻意不重新发明一个不同的 Persuadee。

本轮新增或修订的实现：公开目标可训练的双状态校正、真实先验导数、启动时实现指纹、公开 prompt 的重复惩罚对齐；no_error 控制；固定动作前上下文的效应目标；within-H 奖励与原主指标对齐；按场景独立随机流；跨场景状态复用拒绝；有限值/密度检查；step 后 KL 越界回滚；先记请求的成本账本；F 外部新场景检查；输出目录保护。

这些程序性修复与接口控制不是额外论文贡献，但让贡献更可验证。

旧 v1 checkpoint/rollout/F 不自动兼容。若已有 v1 SFT 训练结果，需单独完成参数映射与输出等价核验再迁移；不能绕过代码哈希验收。旧 F 的监督含候选 y，本版含义已变，必须重新拟合；旧 rollout 不可伪造行为密度继续 PPO。

训练/推理网络使用 BF16、无量化；损失汇总、连续概率/线性代数、FP32 optimizer master/moment 是数值稳定计算，不是把网络偷偷改成 FP32。生产命令拒绝 CPU/FP16/INT8/INT4 退化运行。

当前修复版通过 CPU/A100 BF16 机制回归、原数据审计及真实 Qwen2.5 actor/Persuadee smoke；最终代码的复验结果见 `eclp_runs_v21_repair/gpu_smoke_repaired.json`。这些检查没有测量任务成功率。当前可复用回归脚本为 `check_eclp_mechanism.py`，不沿用未在本轮重跑的历史 76 项统计。完整证据见 `references/VALIDATION_REPORT.md`。

## 11. 正文组织与贡献表述

正文只写一条主线：问题 → ECLP 反馈校正 → 三类监督初始化 → ELPO 效果学习 → 固定预算与机制证据。

建议贡献表述：

**(1) 可反馈校正的连续状态—策略推理。**让内部对方状态具有行为条件的反馈预测与跨轮更新责任，而不仅是参考分析的压缩。

**(2) 状态保持的交互效应策略优化。**在相同当前证据下探索策略，以反馈敏感方向组织候选，用同上下文实际执行回报优化固定采样动作。

**(3) 与主张对应的证据体系。**联合检验预测、反证纠正、策略干预与同预算任务效果；第三点是需要通过实验建立的证据，不是尚未运行就成立的新方法。

研究应用限定在非欺骗、尊重明确拒绝与事实依据的互动。模拟器上的接受不是人的真实信念，也不是一切质量维度。正式结果同时呈现失败、格式和行为质量，不把“只要生成 yes”当成功。

## 来源

[S1] 用户上传《最终方案.md》，尤其 §4–8、§11–12；本方案继承其主结构与全秩动作定义。  
[S2] 用户上传《02_snapshot_audit(1).md》，数据真实字段、LFS 状态、通道问题及历史结果边界；本次另做 source_audit.json 只读核验。  
[S3] 用户上传《03_experiment_protocol(1).md》《04_results_template(1).md》，固定协议与匹配对照；独立两篇/换 Persuadee 等被后续约束覆盖的部分以 §1 为准。  
[S4] 项目快照 `exp_code_ljj/evaluate_baselines.py`：`strict_success_within_5_rounds`、first_success 及主表汇总；原数据与源文件不修改。  
[R1] Zhang et al. Reinforced Latent Reasoning for LLM-Based Recommendation. 用户上传 ICLR 2026 PDF，§3.1–3.2，公式2–8。  
[R2] Jin et al. Persuading across Diverse Domains: a Dataset and Persuasion Large Language Model. ACL 2024, `2024.acl-long.92`，`https://aclanthology.org/2024.acl-long.92/`。  
[R3] Gao et al. Factorized Latent Reasoning for LLM-based Recommendation. `arXiv:2604.26760`，本次补充的近邻，`https://arxiv.org/abs/2604.26760`。  
[R4] Wang et al. GTS: Inference-Time Scaling of Latent Reasoning with a Learnable Gaussian Thought Sampler. `arXiv:2602.14077`。  
[R5] Zhang et al. MindZero: Learning Online Mental Reasoning With Zero Annotations. `arXiv:2606.00240`。  
[R6] Schulman et al. Proximal Policy Optimization Algorithms. `arXiv:1707.06347`。  
实现参考：Hugging Face PEFT 官方 Models/Troubleshooting 文档中的 `autocast_adapter_dtype=False`；Transformers 官方 Qwen3.5 模型文档。外部近邻用于新颖性核验，不替代上传来源的实验事实。
