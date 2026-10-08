# 给论文写作 agent 的任务书

请基于这个目录的 `CCFA_PAPER_HANDOFF_ZH.md`、`PAPER_DRAFT_EN.md`、`LITERATURE_AND_EVIDENCE.md`、`references.bib` 和 `EXPERIMENT_CHECKLIST.csv`，完成 ECLP 的正式英文论文。当前目标是把真实实现组织成连贯论文，并按已验证证据填入结果。不要把实验计划写成已完成结果。

## 研究主线

只围绕一个核心问题：**对方真实反馈与先前预测不一致时，如何学习修正潜在状态并调整下一轮劝说策略？**

方法名暂用 ECLP-Offline-LPO；ECLP 英文展开暂用 Evidence-Corrected Latent Persuasion。推荐标题为 `Learning to Revise Latent Strategies from Feedback for Multi-Turn Persuasion`。全文统一名称后再润色，不能各节另造名字。

两项待验证的机制贡献是：

1. 预测反馈特征，结合实际回复的 readout 与预测误差，用门控更新连续状态。
2. 固定语言执行器与状态模块，用既有回复的 NLL 加上相对固定 SFT 动作的修正增益奖励，优化连续动作 policy。

Gaussian sampling、PPL、LoRA和PPO不是我们的独创。修正分支属于模型内部通道干预，不是对真人的因果效果估计。latent state是任务特征，不是经过验证的心理状态概率。

## 按这个顺序写

1. 阅读方法源码与快照说明，标出训练模块、冻结模块和输入边界；解决文字与代码冲突后再定公式。
2. 写方法与实验设置，继而写引言和相关工作；最后根据实际结果写摘要与结论。
3. 引言遵循任务—已有路线—反馈修正问题—机制—证据五段结构，不把所有模块包装成独立大创新。
4. 相关工作使用已核验的 PersuGPT、MA2P、Coconut、LatentR3 和优化工具引用。补充文献时逐篇核验 primary source 与支持的具体主张。
5. 主结果及消融暂用TBD；拿到同合同完整结果后再填，计算配对差值、95%区间、覆盖率与多seed统计。
6. 按T1–T5与F1–F4组织证据。正文保留主机制和最关键实验，其余放附录。
7. 完成后检查每个贡献是否有充分证据，删除没有结果支持的“显著”“最好”“首次”等主张。

## 必须保持的实现事实

- SFT使用10,000场景、59,868 session目标；249,720是反馈预测暴露次数。
- 离线RL 1,800 contexts，dev200；当前测试仅固定192，来自原1,000测试集。
- 三个backbone是Qwen2.5-7B-Instruct、Meta-Llama-3.1-8B-Instruct、DeepSeek-R1-Distill-Qwen-14B。
- BF16、不量化；LoRA rank16/alpha32/dropout0；state2×256、feature128、action64、strategy tokens2。
- SFT一epoch，public NLL +pred MSE +0.1strategy cosine +0.1private ISR NLL。
- 私有ISR是训练辅助标签；公开推理仅劝说语句。用户取消了公开心理/策略分析实验，不要重新写进结果或消融清单。
- 离线RL冻结backbone、LoRA、state与action映射，只更新policy；4Gaussian候选+1固定SFT action。
- covariance=(0.5²/64)I；均值tanh，但样本不clip。
- factual/cf两分支读相同公开历史，cf仅屏蔽最新反馈error；同动作、同参考回复打分。
- reward=-NLL+0.1tanh[(cfNLL-NLL)-(cfRefNLL-refNLL)]；无反馈时bonus0。
- 动作密度比为64维联合Gaussian；soft KL参照固定SFT；hard KL≤0.1参照本收集批次behavior，不能混写。
- 最新policy collection64、update batch64、update epochs2、LR1e-5；一个完整pass后取最后committed checkpoint，不按test选。
- RL不生成新对话、不调用模拟器或reward judge。不要引用源码其他旧在线分支来描述当前训练。
- SFT prompt预算1536，最新family Our inference6144；feature context1536/response256。附录披露并核查可见历史差异。

## 主实验边界

- 固定子集 `target_turn==1` 且原始 `history==[]`，192个original index。每方法从空历史开始，后续保存本次生成公开历史。
- 完成五轮；任一轮明确接受同一目标即成功。不能在成功后提前停止生成再与五轮golden比较。
- 被劝说者模拟器只接收背景和公开历史，不接收明确goal、心理/策略标签、golden或未来回复。
- 用户指定Zero-shot不看goal；训练方法看goal。保留并披露，建议增加goal-aware untuned作公平控制，不悄悄改旧baseline定义。
- MA2P是本地K=0三角色prompt adaptation，不是完整官方系统；LatentR3适配的数据/预算与Our不同，要列出并设置matched control。
- ECLP-SFT-only必须来自同结构同checkpoint，是RL归因控制；Dialogue-SFT不能替代它。
- 历史Qwen对话与新goal-blind模拟器对话可能协议不同；不能不核实就混合成统一主结果。
- 原始结果在远程 `exp_code_ljj/artifacts/`；主表在 `exp_code_ljj/results/`。完整逐条结果及合同优先于截图和旧Markdown分数。

## 评价与统计

- Success看被劝说者是否接受精确goal；不能把礼貌、考虑、询问信息或预备活动当成更大目标成功。
- Judge输出 `evidence/progress/success` 且success是JSON boolean。引用必须来自当前被劝说者回复。
- Win@1是第一句对golden第一句；Win-Full是全部生成对话对全部可用golden，不叫Win@5。
- A/B和B/A映射回同一生成候选；credit=win1/tie0.5/loss0。两次是位置控制，不是独立n。
- 所有指标报告有效分母与未解析数；补固定192分母的成功上下界。
- Success与Win衡量不同构念，不强行使两者一致。
- 成对bootstrap以场景为单位，不能把轮次或交换判断当独立样本。主要差值用百分点。
- 主模型至少计划3训练seed；当前未跑完的seed不能补虚构均值±方差。

## 优先实验与交付要求

最低实验包：同合同主结果、ECLP-SFT-only、重新训练的no_error、NLL-only RL、同1800数据的continued-SFT/likelihood control、judge人工校准与交换冲突分析。其次是三backbone、多seed、逐轮与成本分析。所有新增运行另由实验agent安排，本任务书不授权自动抢占GPU或启动网格。

关键归因要求：no_error的修正bonus会退化为零。因此状态贡献用error-on/off都做NLL-only RL的比较；奖励贡献用相同error-on SFT checkpoint的NLL-only vs完整reward比较。不要用full模型对no_error的一个差值同时证明两项创新。

请交付正式manuscript、完整BibTeX、表格数据与plot脚本、附录提示词/配置、claim-evidence映射、以及仍缺证据的明确TBD列表。先采用会议无关草稿结构；投稿前再确定具体CCF A会议及当年官方模板、页数、匿名与AI使用政策。

不要复制用户另外的 `AAAI2027_persuasion.pdf` 内部稿的架构、结果或贡献到ECLP；它不属于当前方法的证据或公开引用。
