# DailyPersuasion small5000 数据清单

目标模型：`/data1/liujianjian/model/Qwen3.5-9B`，训练方式：LoRA。

## 直接训练 persuadee 的数据

| 文件 | 样本数 | 输入内容 | 输出内容 | 用途 | 是否用于 persuadee LoRA |
| --- | ---: | --- | --- | --- | :---: |
| `sft_persuadee.jsonl` | 5,000 | 场景背景、persuadee 身份、persuader 历史话语 | 真实数据中的下一句 persuadee 回复 | 学习 persuadee 的角色、观点、犹豫、接受和拒绝行为 | **是** |
| `sft_dialogue.jsonl` | 5,000 | 场景目标和对话历史 | 下一句 persuader 话语 | Dialogue-SFT persuader | 否 |
| `sft_add_isr.jsonl` | 5,000 | 场景、候选策略和对话历史 | 意图分析、策略选择和下一句 persuader 话语 | ISR-SFT persuader | 否 |

`sft_persuadee.jsonl` 每行是一个完整多轮会话的训练样本。5,000 个场景的
persuadee 目标轮数分布如下：

| persuadee 目标轮数 | 场景数 |
| ---: | ---: |
| 5 | 3,090 |
| 6 | 1,564 |
| 7 | 294 |
| 8 | 52 |
| **合计** | **5,000** |

它不是只训练最后一句：历史中的前面轮次也保留在 `history`，训练模板会把
这些历史轮次作为完整对话上下文使用。该文件中的 `system` 字段类似：
`You are Emily, the persuadee in the quoted scenario...`，因此它正是当前
Qwen persuadee LoRA 的起始数据。

注意：现有 `train-user` 入口是为 Llama 3.1 适配的，内部会强制设置
Llama-3 chat template。Qwen3.5 应保留自身的 Qwen chat template，并确认
训练入口使用 `Qwen3_5ForCausalLM`；因此当前文件可以直接作为数据源，但不
建议不改模板就直接启动 Qwen3.5 训练。

## 其他数据的用途

| 文件 | 场景/样本数 | 用途 | 是否直接训练 persuadee |
| --- | ---: | --- | :---: |
| `preference_seed.jsonl` | 1,000 | 用 ISR persuader + API persuadee 模拟两条路径，生成后续偏好对 | 否 |
| `test.jsonl` | 500 | 四种 persuader 方法共用的最终测试上下文 | 否 |
| `manifest.json` | 元数据 | 记录 profile、样本哈希、场景划分和数据文件信息 | 否 |
| `dataset_info.json` | 元数据 | 记录 Alpaca 字段映射 | 否 |
| `dpo_preferences.jsonl` | 当前尚未生成 | ISR-SFT 完成后由模拟流程生成，主要给 persuader DPO 使用 | 否 |

## 三组场景规模

| 划分 | 场景数 | 是否进入 persuadee SFT |
| --- | ---: | :---: |
| SFT | 5,000 | 是，通过 `sft_persuadee.jsonl` |
| Preference | 1,000 | 否，仅用于后续模拟偏好/对齐 |
| Test | 500 | 否，严格留作最终测试 |

## 两个目录的关系

当前以下两个目录中的小规模运行数据哈希完全一致，属于同一份数据的两份
物理副本：

| 目录 | 角色 |
| --- | --- |
| `main_experiment_reproduction/data/small5000/` | 便于直接查看和指定文件 |
| `main_experiment_reproduction/runs/small5000/data/` | `--profile small5000` 运行时使用的隔离运行目录 |

因此，查看或训练 persuadee 时可以使用：

```text
/home1/liujianjian/2-paper-CoLING/0-paper-2024DailyPersuasion/main_experiment_reproduction/data/small5000/sft_persuadee.jsonl
```

如果走统一入口 `--profile small5000`，训练代码会自动读取对应的
`runs/small5000/data/sft_persuadee.jsonl`，两者内容相同。
