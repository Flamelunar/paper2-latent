# Qwen2.5 Zero-shot、MA2P 与 Our 的固定 192 条审计

结论：确认有影响结果可信度的生成和判断问题，但没有证据证明 Zero-shot、MA2P 的优势来自直接读取隐藏策略、golden 或心理标注。Our 自身有更明显的公开回复污染；不能将 baseline 比 Our 高作为错误判据。本次保留主表，仅保存审计代码、统计、源码副本和逐条证据，没有启动新的模型推理。

## 审计范围与真实结果

读取各方法 Selene unified v2 合同指向的生成文件，核对 SHA256、同一固定 192 个原始索引、空初始历史、场景与 golden 对齐、逐条判断的来源哈希与换序 credit。基线旧 summary.json 包含原始 1000 条，本报告所有指标均从固定 192 条逐条记录重算，不能直接读取旧全量汇总作为子集分数。

| 方法 | N | Success (%) | Win@1 (%) | Win-Full (%) | 检出额外角色续写的场景数 | 首次成功前缀已包含续写的场景数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot | 192 | 72.92 | 62.24 | 42.45 | 0 | 0 |
| MA2P | 192 | 91.15 | 78.91 | 53.91 | 2 | 2 |
| Our | 192 | 81.77 | 60.16 | 34.11 | 39 | 17 |
| Our + MA2P | 192 | 81.77 | 65.10 | 29.43 | 0 | 0 |

续写统计使用行首 user/assistant/system/persuader/persuadee 标签作为可核查检测条件，并查看具体实例；不是对所有无标签角色切换的完整检测。17 条不等于 17 条都为假成功；它表示这些判断需要复核。不能删掉污染场景，再把不同的子集当作公平主结果。

## 1. Zero-shot 没有检出直接输入泄露

当前 zero_shot_contract.py:10 只取 background、goal 与公开历史。此前归档的 dialogue_prompt 同样调用 scenario_header(..., include_strategies=False)，虽然构造 item 时带有 strategy 字段，真正字符串渲染没有使用它。未发现向 Zero-shot 输入 reference、心理标签或未来对话的代码路径。背景中自然描述的顾虑与隐藏标注需要区分。

192 条初始历史均为空；960 条对话发言无 think 标签；首条回复与 reference 完全相同的样例为 0；未检出额外角色续写。上述统计不能证明训练数据无预训练污染，但没有支持直接泄露的证据。

当前 Zero-shot 代码修订晚于已保存生成，旧文件没有保存最终输入及源码哈希；因此当前分数评的是旧保存对话，不能声称是修改后代码重新生成的结果。历史 /data1 模型目录已不可访问，不能仅靠路径名或 adapter 文件不存在，证明它与当前 /app 原始 instruct 权重完全相同。

## 2. MA2P 的优势有实际预算基础，也有评测问题

baseline4_ma2p.py:59 每轮依次运行 Perception、World Model、Persuader；常规路径 3 次 Qwen2.5 调用，五轮共 15 次，而 Zero-shot 与 Our 常规劝说路径各 5 次。Perception 的顾虑和动机、World Model 的策略是自身推断结果，不是读取外部心理标签或策略。agent_trace 没有进入模拟对象或 Judge 的公开对话输入。

当前实现是 MA2P prompt adaptation，K=0，无知识库和在线更新。它不等于已经核验的完整原论文复现。额外计算本身是方法组成部分，需要报告预算，而不是删掉分析角色来压低分数。

MA2P 每条劝说发言平均 52.11 词，Zero-shot 35.64，Our 39.63；首轮分别 50.77、32.96、42.33。更长、更具体的输出可能帮助回应顾虑，但这些长度差异不证明 Judge 偏爱长文本。

已确认部分 MA2P 具体事实缺少场景或检索依据。例如 index 66 宣称 Microsoft 的多样性倡议使收入增加 10%、离职率下降 30%、创新增加 25%；index 26 宣称独立经济分析已经准备好。不能将未核实的断言当作可靠证据。该问题需要对所有方法统一约束与核查；Our 也有“已准备案例”等未获背景支持的说法。

## 3. Success 存在语义误判，JSON 合法不代表判断正确

核查保存的五轮对话后，至少发现下列场景未明确接受精确目标，却被判场景成功：

| index | 目标 | 方法与被判 true 的原话 | 违反的判断边界 |
| ---: | --- | --- | --- |
| 4 | 参加 workshop | Zero-shot: "I'm more open ... tell me more"；MA2P: "I'll seriously consider attending" | 开放、考虑和索取信息不等于决定参加 |
| 24 | 写志愿工作主题 | MA2P: "I'll think about it more and let you know my decision" | 尚待决定，后续也未明确选定主题 |
| 26 | 支持建设 sports complex | Zero-shot: "attend the meeting and consider"；MA2P: "willing to learn more" | 开会了解不等于支持建设 |
| 66 | 公司采用 D&I program | MA2P: "If the numbers back it up, I'm willing to consider it"；Our: "let's set up that meeting" | 关键条件尚未满足，或仅同意开会 |
| 106 | 购买 Tesla Model Y | MA2P: "I'd like to test drive one" | 同意试驾不等于购买 |

Zero-shot index 66 也仅表示愿意考虑。现有 success_rule 已明文排除这些情形，说明问题是 Judge 执行规则的语义能力，不能归结为没有写规则。仅这些已核查实例也不足以解释所有方法间差距；没有进行完整人工重标注，不提供所谓纠正后的真实成功率。

模拟 Persuadee 提示词将精确目标提供给它；这是各方法共用的环境，可能增加目标导向的顺从，但本次没有做对照实验，不能断言因果，更不能只给基线换一个更难的模拟对象。

## 4. 生成器没有可靠限制为一次角色回复，Our 受影响更多

public_answer.py:public_turn 和 persuadee_simulation.py:clean_utterance 清理 think、Utterance 与开头标签，但不截断后续角色对话。eclp_elpo.py:BaselineBridge.respond 也只调用 clean_utterance。只验证 trajectory 外层十个对象，不能保证每个 content 真的是单条该角色发言。

MA2P index 700 第二轮，模拟对象原本仍担心房地产风险，却续写了多轮 Persuader/Persuadee 内容并包含投资建议。index 757 同样夹入后续劝说。这些文本进入历史后，后续模型输入已经改变。

Our index 32 第三轮，模拟对象原本只是询问精神 retreat 如何帮助业务，随后输出 user、assistant、多轮角色对话及策略说明，其中自行续写了接受计划。这些内容既污染状态更新和之后的生成，也污染 Success 与 Win-Full。

Our 的 39 条污染场景，Win-Full 平均为 25.64%；其余 153 条为 36.27%。这是输出依赖的事后分组，只表明相关性，不是修复后分数或因果实验。

Our + MA2P 已有第一回复截取：192 条中保存了 370 个截取标记，公开 content 未检出相同续写。但它与旧 MA2P/Our 的边界不同，不能声称严格同条件下仅替换 checkpoint 的比较。

## 5. Our 有公开回复输出内部分析的问题

使用窄模式查找并查看全部 30 条候选，确认 29 个场景中出现策略/心理分析式 Persuader 发言；Zero-shot、MA2P 与 Our + MA2P 未被该窄模式检出同类输出。它不是穷尽检查。

例如：

- index 0，第 3 轮："Sarah seems grateful but hasn't fully committed ... Offering her guided meditation resources ..."
- index 66，第 4 轮："David has agreed to the meeting. It's important to end the conversation ..."
- index 127，第 1 轮："Jane is concerned ... I will use expert opinions and statistics ..."

这些文本描述应该如何劝说，没有自然地对对方说出劝说内容。format_valid/private_marker_leak 只检查特定标记，未拒绝此类未加标签的分析。

需要排查 SFT 中的共享解码器目标：eclp_elpo.py:session_losses 使用同一 utterance_loss/LoRA 分别监督公开话语与 reasoning，reasoning_weight=0.1；sft_full/run.json 确认该配置，源码哈希与训练和 Our 生成一致。没有明确的文本任务模式区分两种输出。公开与 private 状态路径有差异，但在空历史初始化时状态值相同，存在相同条件训练不同输出风格的风险。这是合理的潜在原因，尚未通过开发集消融证明它导致上述公开分析；不应在没有实验的情况下声称调整后一定提升性能。

## 6. Win 存在较强位置敏感性

固定 192 条换序冲突：Zero-shot Win@1 为 109/192=56.77%，Win-Full 为 89/192=46.35%；MA2P 分别 67/192=34.90%、87/192=45.31%；Our 分别 61/192=31.77%、77/192=40.10%。A/B 映射和胜负 credit 正确，冲突来源不是已发现的胜负反算。

Zero-shot 与 Our 的首轮平均得分仅差 2.08 个百分点。对相同场景 credit 差做 10000 次配对 bootstrap，95% 分位区间为 [-4.17, 8.33] 个百分点；完整对话差 8.33，区间 [1.82, 14.58]。这些区间只估计已保存 Judge 得分的场景抽样不确定性，不包含 Judge 偏差、生成污染、模型来源差异；不能据此把完整对话优势视为已经验证的真实说服优势。

新提示词共同 20 条小测已完成，它未消除 Our 低于 Zero-shot 的差异。完整比较的换序一致性改善，首轮一致性未整体改善。因此单纯调整偏好措辞不能替代生成与判断修复。

## 7. 代码与复现风险

- Zero-shot/MA2P 的历史模型路径为 /data1，Our 为 /app。旧目录已不存在，完整权重哈希无法核实；采样参数虽相同，也不能据此证明权重、模板、实际精度和库版本完全相同。
- 历史 baseline 记录没有源码哈希与最终 prompt；当前 baseline 生成器也存在将 cleaned content 写回 raw_output 的路径，破坏原始输出存档。
- MA2P agent_json 可把 None/列表/对象转成字符串，非法 JSON 被静默降级为空字段或默认策略；这是鲁棒性缺陷，不是高分来源的已证实原因。
- MA2P 旧生成入口硬编码原始 1000 条，--limit 192 是前 192 条，不能用于未来固定 192 条；已有另一个 from_start192 入口，但协议、来源和公共回复边界仍需要统一冻结。
- 断点合同只检查部分路径/协议，缺少完整源码、采样参数、模板和数据内容哈希；不能将旧保存结果称为修订代码的新结果。

## 修复顺序

1. **统一单轮公开回复边界。**新版本对所有方法、双方角色采用相同截取与角色检查，保留原始输出；在历史、状态更新及 Judge 前应用。已污染对话必须从污染处连同后续重生成，最稳妥是固定 192 条全程新生成，不能只删尾巴后复用旧后续对话或判断。
2. **修复 Our 的公开输出目标。**在开发集检查共享 reasoning loss 的影响，采用独立辅助头或明确私有分支，保证公开解码器学习对人说话。保留旧 checkpoint 和结果，修订训练协议后做消融；不能用测试集挑选权重。
3. **校准成功判断。**在训练/开发来源上构建目标接受证据与反例，用人工标注核查考虑、试用、了解与精确目标的区别。新 Judge 输出可严格包含 success 和证据摘录；对所有方法统一使用，报告与人工一致率。修订提示词不保证修复模型语义能力。
4. **冻结来源与预算。**统一实际模型权重、tokenizer/template、模拟对象、精度、采样、场景与源码哈希；MA2P 报告常规 15 次 actor 调用及 token/耗时/重试。原始 JSON 与公共回复分别存档，严格验证 MA2P 分析 schema，避免静默降级。
5. **重新比较时保留配对诊断。**在新统一环境评估固定 192 条，同时统计换序冲突，并增加 Our 与 Zero-shot 的直接比较。所有 Win 仍须明确对手与平局计分，不能只修改单个 baseline 的 Judge 或人为降低其分数。

当前主表未修改。证据文件：evidence/statistics.json、continuation_examples.json、selected_sources_and_judgments.json、our_eclp_offline_lpo_narrative_candidates.json、paired_score_uncertainty.json；代码副本在 source/。统计不是人工标注后的替代主结果。
