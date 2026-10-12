<!-- baseline-five-rounds:start -->
## Main experiment: five-round evaluation

This table uses Qwen2.5-7B-Instruct as the Persuader (`/app/hzj/models/Qwen2.5-7B-Instruct`) and the fixed merged Qwen3.5-9B Persuadee (`/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged`). All generation runs use `/app/hzj/anaconda3/envs/ljj/bin/python` and disable thinking/reasoning output. Separate tables are maintained for Meta-Llama-3.1-8B-Instruct and DeepSeek-R1-Distill-Qwen-14B.

Completed rows use the same 1,000 held-out contexts and five-round protocol. The Judge models use BF16. `-` means no complete result is available.

| Judge | Method | Persuader model | N | Strict acceptance within 5 rounds (%) | Win-Rate (vs golden) (%) | Mean rounds to success (failures = 5) | Success parse unresolved |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Selene-1-Mini-Llama-3.1-8B (BF16, strict V17 v7) | Baseline 1: Zero-shot | Qwen2.5-7B-Instruct | 1000/1000 | 66.80 (668/1000) | 67.90 | 3.469 | 0 |
|  | Baseline 2: Dialogue-SFT | qwen2.5-7b-instruct-dialogue-sft-paper10k | 1000/1000 | 62.50 (625/1000) | 65.55 | 3.554 | 0 |
|  | Baseline 3: ISR-SFT + SimDPO | - | - | - | - | - | - |
|  | Baseline 4: MA2P | Qwen2.5-7B-Instruct | 1000/1000 | 75.60 (756/1000) | 58.85 | 3.147 | 0 |
|  | Baseline 5: LatentRATT | - | - | - | - | - | - |
|  | Our | Qwen2.5-7B-Instruct | 1000/1000 | 70.80 (708/1000) | 61.00 | 2.883 | 0 |
| Gemma 4 E4B-it (BF16) | Baseline 1: Zero-shot | Qwen2.5-7B-Instruct | 969/1000 | 53.25 (516/969) | 47.42 | 3.990 | 31 |
|  | Baseline 2: Dialogue-SFT | qwen2.5-7b-instruct-dialogue-sft-paper10k | 939/1000 | 56.98 (535/939) | 48.08 | 3.962 | 61 |
|  | Baseline 3: ISR-SFT + SimDPO | Qwen2.5-7B-Instruct | 940/1000 | 49.68 (467/940) | 51.33 | 3.935 | 60 |
|  | Baseline 4: MA2P | Qwen2.5-7B-Instruct | 892/1000 | 54.48 (486/892) | 62.44 | 3.988 | 108 |
|  | Baseline 5: LatentRATT | rl | 941/1000 | 58.77 (553/941) | 48.99 | 3.829 | 59 |
|  | Our |  |  |  |  |  |  |
| Qwen3-14B (BF16) | Baseline 1: Zero-shot | Qwen2.5-7B-Instruct | 1000/1000 | 66.00 (660/1000) | 46.70 | 3.775 | 0 |
|  | Baseline 2: Dialogue-SFT | qwen2.5-7b-instruct-dialogue-sft-paper10k | 1000/1000 | 66.70 (667/1000) | 45.30 | 3.880 | 0 |
|  | Baseline 3: ISR-SFT + SimDPO | Qwen2.5-7B-Instruct | 1000/1000 | 60.90 (609/1000) | 42.10 | 3.739 | 0 |
|  | Baseline 4: MA2P | Qwen2.5-7B-Instruct | 1000/1000 | 73.60 (736/1000) | 49.00 | 3.584 | 0 |
|  | Baseline 5: LatentRATT | rl | 1000/1000 | 66.30 (663/1000) | 38.05 | 3.740 | 0 |
|  | Our | Qwen2.5-7B-Instruct | 1000/1000 | 67.60 (676/1000) | 30.35 | 2.955 | 0 |

Mean rounds uses the first successful round; a trajectory without success counts as round 5.
Win-Rate (vs golden) compares all five generated Persuader replies with the single held-out golden reply using two order-swapped Judge decisions; a split vote contributes 0.5.
A missing Win-Rate requires new five-round Judge evaluations; previous first-turn Win-Rate scores are not reused.
The original Selene-1 row uses the compact v17 prompt; the additional strict V17 v7 row uses explicit goal-acceptance criteria. Gemma 4 retains prompt v11, and Qwen3-14B uses the same v11 decision criteria with an example-free strict JSON output contract. All rows use the same Win-Rate comparison. Success rate, Win-Rate, and mean rounds use only parsed cases; N and unresolved show parsed and unresolved counts out of 1,000. Each case allows up to three additional generations after a parse or evidence validation error.
<!-- baseline-five-rounds:end -->

<!-- unified-evaluation-v2:start -->
## Main experiment: unified v2 evaluation (Win@1 / Win-Full)

本表采用最新统一评估协议，使用 Selene-8B 与 Qwen3-14B 两个 Judge，沿用上表的方法顺序、Qwen2.5 劝说模型和固定 1,000 个测试场景。`-` 表示尚未完成此版本评估；上表旧分数不填入本表。

Prompt / calculation revision: `matched_win1_winfull_success_v2_20261005`。
提示词：[matched_judge_prompts_v2.json](../../my_eclp_final/matched_judge_prompts_v2.json)；协议说明：[EVALUATION_PROTOCOL_V2_ZH.md](../../my_eclp_final/EVALUATION_PROTOCOL_V2_ZH.md)。

| Judge | Method | Persuader model | N | Goal acceptance within 5 rounds (%) | Win@1 (%) | Win-Full (%) | Mean rounds to success (failures = 5) | Success format unresolved |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selene-1-Mini-Llama-3.1-8B (BF16, unified v2) | Baseline 1: Zero-shot | Qwen2.5-7B-Instruct | 1000/1000 | 87.50 (875/1000) | 50.82 | 53.10 | 2.790 | 0 |
|  | Baseline 2: Dialogue-SFT | qwen2.5-7b-instruct-dialogue-sft-paper10k | 待评估 / 1000 | - | - | - | - | - |
|  | Baseline 3: ISR-SFT + SimDPO | Qwen2.5-7B-Instruct | 待评估 / 1000 | - | - | - | - | - |
|  | Baseline 4: MA2P | Qwen2.5-7B-Instruct | 1000/1000 | 95.60 (956/1000) | 61.30 | 59.10 | 2.347 | 0 |
|  | Baseline 5: LatentRATT | Qwen2.5-7B-Instruct | 待评估 / 1000 | - | - | - | - | - |
|  | Our | Qwen2.5-7B-Instruct | 待评估 / 1000 | - | - | - | - | - |
| Qwen3-14B (BF16, unified v2) | Baseline 1: Zero-shot | Qwen2.5-7B-Instruct | 1000/1000 | 75.80 (758/1000) | 49.05 | 60.65 | 3.284 | 0 |
|  | Baseline 2: Dialogue-SFT | qwen2.5-7b-instruct-dialogue-sft-paper10k | 待评估 / 1000 | - | - | - | - | - |
|  | Baseline 3: ISR-SFT + SimDPO | Qwen2.5-7B-Instruct | 待评估 / 1000 | - | - | - | - | - |
|  | Baseline 4: MA2P | Qwen2.5-7B-Instruct | 待评估 / 1000 | - | - | - | - | - |
|  | Baseline 5: LatentRATT | Qwen2.5-7B-Instruct | 待评估 / 1000 | - | - | - | - | - |
|  | Our | Qwen2.5-7B-Instruct | 待评估 / 1000 | - | - | - | - | - |

Success：逐轮使用截至当前轮的前缀，只判断 Persuadee 是否明确接受精确目标，不提供未来对话或 golden。只允许输出 `{"success":true}` 或 `{"success":false}`。礼貌、兴趣或愿意考虑不自动算成功；不要求已经执行、固定关键词或正式承诺。实施安排与尚待满足的关键接受条件按统一规则区分。五轮内任一轮接受计成功；首次成功轮次用于平均轮数，未成功计 5。第五轮当前接受率另存 summary.json。
Win@1：双方同一测试起点后的第一条完整 Persuader 回复对比，提供共同背景、目标和前置历史。
Win-Full：Our 全部已生成五轮对话与同一起点后 golden 的全部可用对话对比，包含双方角色；golden 不截断、不补齐。双方长度可能不同，缺失回复不当成接受或拒绝；因此本指标不是等轮数的 Win@5。原始 golden 与当前模拟对话对象的来源差异也需保留在结果解读中。
两项 Win 指标均执行两次独立匿名 A/B 换序请求，每次只返回 A、B 或 Tie。每次胜=1、平=0.5、负=0，两次取平均，再对相同 1,000 个场景取平均×100%。顺序冲突与显式平局分别统计。
Baseline 5 的生成文件未记录生成精度或量化设置；本表评估已保存的回复，不推断该组生成时使用 BF16。
所有 Judge 使用相同 rubric、字段、严格 JSON schema 与有限 token 解码约束。格式/技术失败不得转换成假成功或直接剔除后当作完整结果；完整行要求 1,000 个场景完成。
<!-- unified-evaluation-v2:end -->
