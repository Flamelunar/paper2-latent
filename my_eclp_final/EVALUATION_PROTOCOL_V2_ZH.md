# 统一评估协议：Win@1、Win-Full 与说服成功

## 当前固定测试约定（2026-10-06 生效）

用户明确要求：今后主实验仅测试原测试集中 **192 个无原始前置对话的场景**。筛选条件为 `target_turn == 1` 且原始 `history == []`，使用 `exp_code_ljj/data/test_selection_192.json` 固定原始索引，数据文件为 `exp_code_ljj/data/evaluation_points_from_start192.jsonl`。各方法与 Selene/Qwen3 必须使用相同子集；禁止重新随机抽取 192 条或重新编号后误配 golden。

第一轮只有背景和目标，无原始对话、候选策略或额外心理状态标注；后续保留本次生成的公开对话。最多五轮，任一轮达到目标即场景成功。Win@1 比较双方首条完整劝说回复；Win-Full 比较模型全部五轮与同场景原始 golden 全部可用对话。换序独立评分及平均轮数（失败计5）沿用统一 v2。

训练数据、SFT/RL 数据和开发集不变。`data/evaluation_points.jsonl` 的原 1000 条及其 manifest 保留作来源档案，不作为今后全量主测试入口。当前 192 条分数通过筛选已有逐条判断获得，不启动训练、生成或 Judge。旧的 1000 条运行命令/合同属于历史记录；未来启动前必须确认生成和评估仅覆盖清单中的原始索引，结果目录另建，禁止将 192 条任务追加进原 1000 条合同。GPU4 上其他任务不得停止。

当前主表 `exp_code_ljj/results/main_experiment_table.md` 的 unified v2 部分使用 192 条；上方旧协议 1000 条表仅作历史档案。格式失败不能从分母中删除，未完成方法显示 `待评估 / 192`。

版本：`matched_win1_winfull_success_v2_20261005`。

## 1. 测试数据与对齐

固定使用原 1,000 个测试点中的 192 个无原始前置对话场景；当前复用已保存的五轮生成及逐条评估，不重新训练或生成。
从原始 `DailyPersuasion_full_version.json` 追溯 golden，按公开场景指纹、session_id、target_turn、完整前置历史和 reference 精确匹配。
原始 id 不唯一，不能单独用 id 匹配。Golden 仅用于独立评分，不反馈给生成或训练。

Win@1 比较测试点之后双方第一条完整 Persuader 回复，包含完整公共背景、目标和共同历史；“第一条”不是只截取一句话。

Win-Full 比较 Our 已生成的全部十条消息（五个完整交互轮）与从同一测试点起 golden 的全部可用消息，保留双方的 Persuader 和 Persuadee 回复。
不限定 golden 必须五轮，不截断较长 golden，不补齐较短 golden；golden 最后只有一条 Persuader 回复时也按实际记录展示，缺失反馈不能视为接受或拒绝。

这是用户选定的全部可用对话比较，双方长度可能不同，并不是等轮数的 Win@5。每条结果保存实际轮数、golden 回复数量和最后是否有 Persuadee 回复；汇总按 golden 劝说回复数量少于、等于、多于 Our 分类报告，帮助观察长度影响。
Golden 对方回复来自原始发布会话，Our 对方回复来自当前固定模拟器，因此 Win-Full 还存在对话对象来源差异。相同背景和历史不能消除这一局限。

## 2. 公平的胜负判断

Selene-8B 与 Qwen3-14B 使用完全相同的 rubric、输入字段、输出 schema 和解码约束，保留各自原生 chat template；Qwen3 使用非 thinking 模式。
Win@1 与 Win-Full 各自执行两次独立请求：第一次 A=Our/B=golden，第二次 A=golden/B=Our。一次 prompt 只展示一个排列，只输出一个 winner；不告诉 Judge 哪个候选是 Our 或 golden，也不共享前次判定。

胜者输出仅允许 `{"winner":"A"}`、`{"winner":"B"}`、`{"winner":"Tie"}`。
每次 Our 胜得 1、平得 0.5、负得 0，两次取平均。显式平局与顺序不一致分别统计；两次矛盾不能解释成 Judge 判了平局。
不因为多轮、较长或更强硬而直接加分，不假设未提供的事实、回复或结果。

## 3. 说服成功

逐轮分别给出截至该轮的前缀，实际只传该轮可见对话，不传未来轮、golden 或前一轮 Judge 输出。
判断当前生成的 Persuadee 回复是否明确接受、承诺、或在语境中明确延续对精确目标的接受。
行动目标要求决定或承诺；态度目标允许明确认同；不要求执行已经完成、固定关键词或正式承诺句式。

礼貌、兴趣、索取信息、愿意考虑本身不算接受。目标为试用/了解/参加时，明确接受该活动可算成功；目标为长期采用/购买/加入时，仅试用后再决定不算。
已经决定后的普通时间安排不取消接受；关键前提未满足且决定仍待确认不算接受。多目标需要接受全部必需部分，不添加目标之外的要求。
对方撤回接受时，该轮可为 false，不把首次成功之后的未评轮次自动补为 true。

主指标为五轮内至少一次成功，保持 Success-within-5；补充第五轮当前接受率和首次成功轮数（未成功计五轮）。

成功输出仅允许 `{"success":true}` 或 `{"success":false}`。JSON boolean 不允许转成字符串、数字或 null。

## 4. 严格格式与复现

提示词保存在 `matched_judge_prompts_v2.json`。通过有限 JSON token grammar 限制解码为合法 schema，不靠宽松正则、bool(string) 或修补内容获得结果。
输出结构合规不代表语义判断正确；保留每次原始输出、prompt hash、独立请求 ID、token 数和实际耗时。
重复 key、额外字段、围栏和尾部文本均被拒绝。技术失败不默认算模型失败，不用剔除失败样本的方式虚增分数。
保存输入、golden、提示词、评估代码与模型元数据的版本约束，断点恢复遇到变化时拒绝混合。

## 5. 运行和结果

只评 Selene-8B 与 Qwen3-14B，不评 Gemma。Judge 使用 BF16、无量化；运行器按物理 GPU 设置 `CUDA_VISIBLE_DEVICES` 并记录显卡编号。
当前主评估覆盖五个 baseline 和 Our，各 192 条；成功判断按五轮前缀执行，Win@1 与 Win-Full 各做两次独立换序判定。已中断的小规模试评已归档，不计入新版结果。
运行器在 GPU 0、1、4 上并行调度；启动每个模型前检查可用显存并设置进程显存上限，显存不足时等待，不终止现有任务。请求按批次严格 JSON 解码，逐请求保存原始结果，可从已保存的请求与场景记录断点续跑。
Baseline 5 的生成 JSONL 未记录生成精度或量化设置；新版结果只评分已保存的回复，不推断该组生成时使用 BF16。Baseline 1–4 的生成协议标记 BF16。

代码入口：`prepare_matched_golden_v2.py`、`evaluate_matched_v2.py`、`run_matched_eval_v2.py`。
`evaluate_matched_v2.py` 的 `--method` 支持表中六种方法，`--judge` 支持 Selene 与 Qwen3 两个 Judge；baseline 输入严格验证原测试上下文、五轮生成和 BF16 协议。结果目录按 Judge 和方法分开。
数据和输出：`eclp_runs_gpu4/evaluation_v2/`。
结果表：在 `exp_code_ljj/results/main_experiment_table.md` 原表下面新增 v2 表，包含 Selene 与 Qwen3 两个 Judge 分组、六种方法和 Our 每组末行；新增独立 Win@1 与 Win-Full 列。
`update_matched_table_v2.py --template-only` 只绘制待评估表格，不调用模型。后续不加此选项时，仅填入完整且可追溯的 v2 结果，不读取旧评估分数或未完成试评。
任何旧指标结果或不同 prompt 的 baseline 都不能直接与此版本分数混排比较，基线需要采用同一 v2 协议复评。
