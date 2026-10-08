# Qwen2.5 Zero-shot 输入与评估审计（2026-10-06）

## 结论

没有在现存 Zero-shot 生成代码中发现显式策略、额外心理状态标注、golden 回复或未来对话进入 Persuader 提示词。不能因为 Zero-shot 高于某些训练方法，就判定数据泄露或降低它的评分。

但当前实验是 **从测试集给定的对话位置继续五轮**，不是每个场景都从背景和目标开始。原始前置对话已经包含人的顾虑、此前劝说理由，甚至接受倾向。这是当前结果解释上的重要限制；不等同于未来答案泄露，也不能把已有劝说成果全部归因于新模型。

## 核对证据

- `exp_code_ljj/repro/persuadee_simulation.py:dialogue_prompt` 调用 `scenario_header(..., include_strategies=False)`。虽然构造临时字典时带了 strategy，最终字符串并不包含它。
- `exp_code_ljj/repro/data.py:scenario_header` 实际提供 background、goal；`transcript` 仅取历史的 role 和 content。原始测试 history 使用 `session[:target_index]`，reference 使用目标位置回复，未将目标回复拼入历史。
- 历史实现 `0-paper-2024DailyPersuasion/main_experiment_reproduction/repro/persuadee_simulation.py` 的 zero-shot 路径也是 dialogue_prompt；带策略的是 ISR 分支，zero-shot 的 with_isr 为 false。
- 当前 1000 条生成轨迹与 legacy/qwen25_zero_shot.jsonl、原始复现实验 generation_5rounds/qwen25_zero_shot.jsonl 的轨迹逐条相同。生成文件 SHA256 与 Selene v2 run.json 记录相同。
- 生成结果 JSON 中存了 strategies、reference，但存储字段不等于实际模型输入。历史没有逐请求保存最终渲染 prompt，现存代码与轨迹一致性不能完全替代历史运行时证据。

## 前置历史的影响

| 分组 | 场景数 | 五轮内成功（Selene v2） | 新生成首轮成功 | 第五轮接受 |
| --- | ---: | ---: | ---: | ---: |
| 无前置历史 | 192 | 140/192 = 72.92% | 3/192 = 1.56% | 135/192 = 70.31% |
| 有前置历史 | 808 | 735/808 = 90.97% | 296/808 = 36.63% | 695/808 = 86.01% |
| 全部 | 1000 | 875/1000 = 87.50% | 299/1000 = 29.90% | 830/1000 = 83.00% |

两组并非随机对照，不能将差值直接解释成历史的因果效果。前置历史长度为 0/2/4/6/8/10/12/14 条，数量分别为 192/182/210/204/155/51/5/1。

例如 index=1 的前置历史最后一句已说：`You're right, Alice. I guess I should prioritize my career development first. I just need to find the right startup for me.` 当前协议仍要求新生成的回复表达、重申或继续接受，但无法证明接受是本方法首次促成。没有对全部初始历史做人工/模型接受标注，不报告“已经成功”的精确数量。

## 模型、解码与评估

- 历史轨迹记录的 Persuader 路径为 `/data1/liujianjian/model/Qwen2.5-7B-Instruct`，并非表格所写的 `/app/hzj/models/Qwen2.5-7B-Instruct`。两处均未发现 adapter_config.json、trainer_state.json、training_args.bin；未做完整权重指纹比对，因此不能断言两处权重完全相同或从未合并微调。
- 记录解码参数：temperature=0.95、top_p=0.7、top_k=50、max_new_tokens=512、batch=2、五轮。当前本地加载路径使用 BF16、无量化。历史精度来自协议/配置声明，不能从生成文本反推精度。没有证据显示 Zero-shot 使用最佳候选筛选。
- Selene v2：Success=87.50%，Win@1=50.82%，Win-Full=53.10%。两项 Win 分数接近 50%，不是 87.5% 的胜率。Qwen3 对同批轨迹 Success=75.80%，说明 judge 差异也较大。
- Success 取五轮任一轮成功，第五轮仍接受仅 83.00%。旧表的 Selene 66.80% 使用旧 rubric；不能直接与新表作相同量尺比较。
- Persuadee 模拟器也收到目标，提示词允许自然抵抗/接受，没有强制接受指令；但目标可见、模拟器训练分布都可能影响顺从程度。此问题属于共享实验环境，未单独为 Zero-shot 更换对手或提示词。
- Win-Full 比较五轮模拟对话与 golden 全部剩余对话；长度及对话对象来源可能不同。即使 rubric 提醒不偏爱长度/顺从，仍不能保证消除偏差。

## 本次改动（仅 Zero-shot 生效）

1. 新增 `exp_code_ljj/repro/zero_shot_contract.py`，明确白名单读取 background、goal 和公开历史 role/content，禁止其他标注字段经提示词构造进入输入；验证历史角色及文本类型。
2. `persuader_prompt` 只在 baseline1_zero_shot 分支调用独立构造器；其他基线继续原分支。
3. 生成入口及 Zero-shot 历史复用入口检查模型目录的显式训练/adapter 文件，避免误把可识别的微调目录当 Zero-shot。这是防误用检查，不是完整权重来源证明。
4. 1000 条现有输入的新旧提示词逐条相同；向策略、reference、心理状态、future_dialogue 等字段注入 canary 后提示词不变；Baseline2/3 提示词相同；模型拒绝检查、语法和生成源哈希检查通过。没有 GPU 推理或新评估。
5. 修改前文件保存在 `exp_code_ljj/artifacts/zero_shot_audit_20261006/before/`。表格数字、现有轨迹、其他基线、评估提示词及 GPU 任务没有修改。

## 后续建议

默认保留共同前置对话，以保持与当前其他基线及 golden 的起点一致。若“只给背景和目标”要求连原始前置对话也不能给，则应单独建立从零开始的 Zero-shot 结果，生成端双方都从空历史开始，并将 golden 改为原始完整对话起点；不能只清除 Persuader 历史而让模拟器保留，也不能替换旧表数字继续声称同起点评测。该变更尚未执行，等待用户对历史范围的明确选择。

科研结论建议额外区分“初始未接受 → 新接受”与“维持既有接受”，并报告终轮接受率；共享指标应对所有方法一致应用。本次未更改其他方法或通过收紧 Zero-shot 独有评分来压低分数。
