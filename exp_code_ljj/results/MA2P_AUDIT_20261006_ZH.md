# MA2P 高分审计（2026-10-06）

## 结论

目前没有证据说明 MA2P 高分来自直接输入 golden、外部策略标签或私有心理状态。但确实找到 Selene unified v2 将“考虑、了解、试驾”判成更大目标成功的实例。MA2P 的 91.15% 是当前 Judge 下的观测结果，不能直接当作已经验证的精确目标接受率。

MA2P 也确有额外计算和更具体的提示词：每轮同一个冻结 Qwen2.5 分别进行 Perception → World Model → Persuader 三次生成。推断心理状态和策略是该方法自行产生的中间结果，不能视作外部标注泄露，也不能为了让 Our 胜出而删除这个方法的组成部分。

本次仅审计并保存证据；没有调整主表、替换 Judge、启动模型推理或停止 GPU 任务。

## 数据及结果核对

使用固定清单中的同一批 192 个原始 history 为空、target_turn=1 的场景。原始 1000 条文件保留作来源档案。

- Success-within-5：175/192 = 91.15%。
- Win@1：78.91%；Win-Full：53.91%。两者都以同场景 golden 为对照，不能解释为直接击败 Our 的比例。
- 第五轮接受：169/192 = 88.02%。
- 首次成功轮次人数：第1轮6、第2轮20、第3轮67、第4轮57、第5轮25、五轮均未成功17。
- 平均首次成功轮数，失败计5：3.568。Our 在相同子集为2.651；高成功率与更早成功是不同指标。

## 输入、模型与评估边界

实际 `baseline4_ma2p.py:context` 读取 background、goal、persuader、persuadee 名称。Perception 读取此上下文及当前公开对话；World Model 另外读取本方法生成的 Perception 和过去计划；Persuader 读取本方法生成的顾虑、动机和策略。没有传原始 strategies、reference、私有 persona、未来对话或 Judge 结果。

当前代码使用共享 `user_prompt` 调用固定 Persuadee，未将 agent_trace 发给对方。统一 Judge 的 `success_prompt` 和 `win_prompt` 只展开公开 role/content，不展开 agent_trace。Golden 只用于胜负比较，不进入生成；Success 请求也不含 golden。

核验结果：

1. 固定192条 × 5轮，对实际 ma2p_turn 使用 CPU 假端点记录调用。向 reference、strategies、psychological_state、persona、future_dialogue、judge_outcome 等字段注入标记后，全部模型输入不变。每轮正常路径确实调用3次。
2. 192条 ×（5个Success请求 + 4个Win换序请求）=1728个已保存请求，其 prompt_sha256 与当前重建提示词完全匹配。
3. 污染 agent_trace 不改变 Judge 提示词；已保存公开对话1920条均无 think标签。
4. 现有完整评估的源文件、golden、逐条判断和汇总校验通过，当前192条汇总确为175成功。

上述检查证明当前代码边界及现存 Judge 提示词，不能替代过去生成时未保存的最终输入日志。历史 MA2P 轨迹不包含当前新版过滤标记，说明当前源码并非完整的历史执行快照。

MA2P 和 Zero-shot 的历史记录都使用 `/data1/liujianjian/model/Qwen2.5-7B-Instruct`，固定模拟器都记录 `/data1/liujianjian/Persuadee/Qwen3.5-9B-persuadee/merged`。解码参数一致：temperature=0.95、top_p=0.7、top_k=50、max_new_tokens=512。当前默认 `/app` 路径不同；没有完整权重哈希证据证明所有路径下权重一致。本审计未加载或修改模型。

## 确认的语义判定问题

已查看下面5个场景的全部五轮公开对话，按照当前 success_rule，它们不应因现存文本认定为接受精确目标。

| 原始index | 精确目标 | 被判成功的对方原话 | 问题 |
| --- | --- | --- | --- |
| 4 | 参加home organization和presentation workshop | `I'll seriously consider attending the workshop and see if it's something I'd enjoy.` | 认真考虑尚不等于决定参加；第5轮是唯一true。 |
| 24 | 写animal shelter志愿工作 | `I'll think about it more and let you know my decision.` | 后续第4、5轮仍表示会思考后决定，没有决定写该主题。 |
| 26 | 支持sports complex建设项目 | `I'm willing to learn more about this project and see if it can truly benefit our city.` | 愿意开会了解，不等于支持建设；第5轮是唯一true。 |
| 66 | 公司采用diversity and inclusion program | `Send me the cost-benefit analysis and the case studies, and let's see what we can come up with.` | 第4轮仍等数据后决定，第5轮`If the numbers back it up, I'm willing to consider it.`，关键接受条件未解决。 |
| 106 | 买Tesla Model Y | `maybe test drive one before making a decision.` | 第5轮同意试驾，五轮中未决定购买。 |

现有 rubric 第3条已排除“考虑或索取信息”，第4条已区分试用与购买/长期采用，第5条已排除决定尚待证据的关键条件。故这些是模型语义判断没有遵守已有规则，不能仅以JSON合法、解析失败为0，认定判断正确。

这是已发现的实例，不是192条完整人工重标注。没有根据关键词自动修改分数，也不报告所谓“纠正后真实成功率”。例如“try meditation”本身就是目标时，`give it a try`可以成功；不能统一把所有try判false。

同类问题也在Zero-shot出现：index4仅表示更开放并询问workshop内容、index26仅愿意开会考虑项目，都曾判true。Our的index66首轮开会了解也被判true。不能单独收紧MA2P评分，否则失去可比性。首轮误判可能影响平均轮数；只要后续真正接受，整个场景成功计数也可能仍然正确，必须逐场景区分。

## MA2P生成内容及比较预算

- 每轮常规3次Persuader模型调用，相较Zero-shot常规1次；五轮分别15次与5次。双方模拟器各5次相同。需要报告实际token量、耗时及重试，不能据此声称相同推理预算。没有发现根据Judge奖励选最佳候选或额外模拟分支的代码路径。
- 192条中，MA2P劝说发言平均52.11词，Zero-shot35.64词，Our39.63词；首轮分别50.77、32.96、42.33词。更长、更具体的表达可能影响模拟器和Judge，但长度统计不能证明偏见或解释全部优势。
- MA2P会将缺少可观察依据的顾虑推测交给下游。它是内部推断，不是标注泄露；同时需保留推断的不确定性，避免把猜测当事实。
- 存在未由场景背景支持的具体事实陈述。例如index66宣称Microsoft的diversity initiatives使收入增加10%、离职率下降30%、创新增加25%；未提供来源或检索证据。index26宣称有现成的独立经济分析。不能在未查外部资料的情况下断言全部为假，但当前记录不能支持这些断言。模拟器接受这些说法、Judge又未充分惩罚时，可出现“无依据说法促进高分”的现象。若引入事实依据约束，必须统一对所有方法应用。

## 实现与复现限制

- 当前是“MA2P prompt adaptation, K=0”：同一冻结模型扮演不同agent、无知识库、无在线更新。项目内未找到MA2P原论文和官方实现的可核对副本，不能认定为完整原版复现。表格/论文应明确版本和计算预算。
- `agent_json`不严格校验字段类型，可将列表/对象/None转成字符串，解析失败时悄悄空值/默认策略；这是鲁棒性问题，没有证据证明它导致高分。
- 断点续跑只核对部分模型路径、protocol及scenario_id，未冻结源码、全部采样参数和场景上下文哈希；建议未来新运行使用完整不可变合同。
- 旧 `verify_generation_file` 硬编码当前/app路径，与现存/data1历史记录不一致；可能错误拒绝合法历史文件，应按明确的历史合同验证而不能直接改元数据伪装一致。
- 旧MA2P生成入口仍硬编码原1000条来源。未来只运行192条时需单独适配固定清单并保留原始索引，不得将`--limit 192`当成固定子集；本次未启动入口。

## 建议处理顺序

1. 保留当前观测结果和审计记录，不手工只压低MA2P分数。
2. 对固定192条的所有方法，统一抽查/标注目标接受，优先核对“了解/考虑/试用≠更大目标”，明确真正接受的轮次；同样保护以考虑或试用本身为目标的场景。
3. 如需改Judge协议，用对所有方法一致的、按具体目标给出的正反例或显式目标满足证据，严格JSON仍可保留；在新的结果目录统一重评。新增证据字段可帮助核查，但也须记录新增解析失败和token成本，不能假设提示词改动必然修复语义。
4. 给MA2P补足版本来源、正常与重试调用数、实际token量和耗时。论文同时报告成功率和推理成本，必要时增加同预算显式策略对照。
5. 如修改生成中的事实约束、模拟器或目标可见性，所有方法应采用同一环境并重新生成，不应仅给MA2P更难的对手。
