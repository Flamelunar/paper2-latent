# 双 Persuader 扩展实验计划

## Summary

在不覆盖现有 Qwen2.5-7B-Instruct 结果的前提下，新增两套完整实验：

- 任务 1：GPU 0 使用 Meta-Llama-3.1-8B-Instruct。
- 任务 2：GPU 1 使用 DeepSeek-R1-Distill-Qwen-14B。
- 每套运行 Baseline 1--5，并在固定测试集上各生成 1,000 条五轮对话。
- 首轮只使用 Qwen3-14B 作为 Judge。
- 所有 Persuadee 统一使用：`/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged`。
- 所有代码统一使用：`/app/hzj/anaconda3/envs/ljj/bin/python`。
- 所有模型关闭 thinking/reasoning，只把最终答案文本送入后续对话和 Judge。

## 输出与目录隔离

生成结果按 Persuader 分层保存，避免覆盖当前 Qwen2.5 结果：

```text
/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/artifacts/
├── generation_5rounds/
│   ├── qwen25/
│   ├── meta_llama31/
│   │   ├── baseline1_zero_shot.jsonl
│   │   ├── baseline2_dialogue_sft.jsonl
│   │   ├── baseline3_isr_simdpo.jsonl
│   │   ├── baseline4_ma2p.jsonl
│   │   └── baseline5_latentratt.jsonl
│   └── deepseek_r1_qwen14b/
│       ├── baseline1_zero_shot.jsonl
│       ├── baseline2_dialogue_sft.jsonl
│       ├── baseline3_isr_simdpo.jsonl
│       ├── baseline4_ma2p.jsonl
│       └── baseline5_latentratt.jsonl
└── success_judgments/
    ├── qwen3_14b_meta_llama31/
    └── qwen3_14b_deepseek_r1_qwen14b/
```

训练 adapter 按 Persuader 和 baseline 分层保存：

```text
/app/hzj/liujianjian/persuader_atapter_save/
├── meta_llama31/
│   ├── baseline2_dialogue_sft/
│   ├── baseline3_isr_sft/
│   ├── baseline3_simdpo/
│   └── baseline5_latentratt/
└── deepseek_r1_qwen14b/
    ├── baseline2_dialogue_sft/
    ├── baseline3_isr_sft/
    ├── baseline3_simdpo/
    └── baseline5_latentratt/
```

每个 adapter 目录只保存 adapter、训练状态和 provenance，不合并成完整 Persuader 模型。

## 固定环境、模型和数据

### 运行环境

- Python：`/app/hzj/anaconda3/envs/ljj/bin/python`
- PyTorch：2.6.0+cu118
- Transformers：5.12.1
- PEFT：0.19.1
- GPU 0：Meta-Llama 任务
- GPU 1：DeepSeek 任务
- GPU 2、GPU 3：不参与这两套新实验

### 模型路径

- Meta-Llama Persuader：`/app/hzj/models/Meta-Llama-3.1-8B-Instruct`
- DeepSeek Persuader：`/app/hzj/models/DeepSeek-R1-Distill-Qwen-14B`
- Persuadee Base：`/app/hzj/models/Qwen3.5-9B`
- Persuadee Adapter：`/app/hzj/liujianjian/Qwen3.5-9B-persuadee/adapter`
- Persuadee Merged：`/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged`
- Qwen3 Judge：`/app/hzj/models/Qwen3-14B`

### 数据拆分

- 原始 DailyPersuasion：13,000 个场景。
- 随机种子：42。
- Persuader SFT：10,000 个场景、59,868 个 turns。
- Preference：2,000 个场景。
- Test：1,000 个场景。
- 三个场景集合不得重叠。
- 所有方法使用相同的 1,000 个测试场景和相同顺序。
- 测试场景顺序 SHA256：
  `41ad0be519824efe135fb46170de409297e110a81f61096a2c9071884f9f4856`

## 统一 Persuadee 设置

所有 Baseline 和 Our 在生成测试对话时使用同一个冻结 Persuadee：

- 训练目标：`scenario + dialogue history + current persuader utterance -> persuadee response`
- 训练数据：5,000 条记录、27,308 个 assistant turns。
- 训练方式：监督式 user-model SFT；不使用 ISR target，不使用 DPO。
- 训练参数：2 epochs，max length 2048，batch size 1，gradient accumulation 16，learning rate `1e-4`，warmup ratio 0.03，seed 42。
- LoRA：r=16，alpha=32，dropout=0.05。
- LoRA 目标模块：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。
- Persuadee 在所有 Persuader 方法中冻结，不因方法变化。

## 统一生成协议

每个 baseline 在固定测试集上生成 1,000 条完整轨迹：

- 5 rounds。
- 每条轨迹最多 10 条双方发言。
- `temperature=0.95`。
- `top_p=0.7`。
- `top_k=50`。
- `max_new_tokens=512`。
- `enable_thinking=False`。
- 禁止 `<think>...</think>` 内容进入 trajectory、Persuadee 输入或 Judge 输入。
- 每条结果保存 index、scenario ID、方法名、模型路径、Persuadee 路径、采样设置、GPU 和完整轨迹。

## 任务 1：Meta-Llama-3.1-8B-Instruct

基础模型：`/app/hzj/models/Meta-Llama-3.1-8B-Instruct`。

运行 GPU：`cuda:0`。

执行 Baseline 1--5：

1. Baseline 1：Meta-Llama-3.1-8B-Instruct zero-shot。
2. Baseline 2：Dialogue-SFT，使用 10,000 场景、59,868 turns；adapter 保存到 `persuader_atapter_save/meta_llama31/baseline2_dialogue_sft/`。
3. Baseline 3：ISR-SFT + SimDPO。ISR-SFT 使用 59,868 turns，SimDPO 使用 2,000 preference 场景；ISR 和 SimDPO adapter 分开保存。
4. Baseline 4：MA2P，`K=0`。
5. Baseline 5：适配后的 LatentRATT 隐式推理；沿用全量 ISR 数据口径。

输出目录：`artifacts/generation_5rounds/meta_llama31/`。

## 任务 2：DeepSeek-R1-Distill-Qwen-14B

基础模型：`/app/hzj/models/DeepSeek-R1-Distill-Qwen-14B`。

运行 GPU：`cuda:1`。

执行流程与任务 1 完全一致，输出目录为 `artifacts/generation_5rounds/deepseek_r1_qwen14b/`，adapter 保存到 `persuader_atapter_save/deepseek_r1_qwen14b/`。

DeepSeek-R1 额外要求：

- 明确关闭 thinking/reasoning。
- 正式生成前先做 5 条样本 dry run。
- dry run 必须确认无 `<think>` 内容、五轮结构正确、Persuadee 回复正常。
- 若仍返回 thinking 标记，生成器必须剥离后再写入 trajectory，并记录过滤计数。
- provenance 记录 `thinking_policy=disabled` 和实际过滤结果。

## Baseline 5 适配要求

当前 LatentRATT 实现默认绑定 Qwen2.5-7B，不能只替换模型路径。对 Llama 和 DeepSeek 必须：

- 从模型 config 读取 hidden size、层数和 attention 结构。
- 使用目标模型自身 tokenizer 和 special tokens。
- 根据目标模型实际模块名校验 LoRA target modules。
- 校验 latent attention 输入输出维度。
- 先用少量数据完成 forward/backward smoke test，再进行完整训练。
- 独立保存 adapter、latent attention 权重、训练状态和 `run_config.json`。
- 记录数据 hash、GPU、dtype、seed、训练步数和恢复信息。

如果某个模型的 Baseline 5 smoke test 失败，该模型的 Baseline 5 停止，不得复制其他模型的结果。

## Qwen3-14B Judge 评测

Judge 模型：`/app/hzj/models/Qwen3-14B`。

评测环境：`/app/hzj/anaconda3/envs/ljj/bin/python`。

新评测统一使用：

```bash
--prompt-version v18
```

结果目录：

```text
meta_llama31/* -> artifacts/success_judgments/qwen3_14b_meta_llama31/
deepseek_r1_qwen14b/* -> artifacts/success_judgments/qwen3_14b_deepseek_r1_qwen14b/
```

v18 Judge 输出必须严格是一个 JSON 对象：

```json
{
  "round_success": [false, false, true, true, true],
  "win_ab": "A",
  "win_ba": "B"
}
```

格式约束：

- 只能有 `round_success`、`win_ab`、`win_ba` 三个字段。
- `round_success` 必须是长度为 5 的原生 JSON 布尔数组。
- `win_ab`、`win_ba` 必须是大写 `A` 或 `B`。
- 禁止额外解释、Markdown、代码块、`<think>`、`null` 和额外字段。
- 成功序列必须单调；成功后不能回退为失败。
- 最多额外重试 3 次。
- 三次重试仍失败时记录为 parse unresolved，不自动转换为 False。

成功率规则：

- Persuadee 明确接受、承诺或坚定决定执行完整目标，才算成功。
- 多动作目标必须全部接受。
- “我考虑一下”“我想了解更多”“听起来不错”“也许可以”“先试试看以后再决定”不算成功。
- 如果目标本身是了解、尝试、参加或考虑，明确接受该目标可以算成功。
- 只使用 Persuadee 的话判断，Persuader 的陈述不能证明成功。
- 每一轮只使用该轮及此前的对话。

Win-Rate 规则：

- 将五轮 Persuader 回复作为一个整体，与单条 golden 回复比较。
- 判断相关性、具体可信的理由、尊重语气和事实责任。
- 忽略 Persuadee 回复和消息长度。
- `win_ab`：生成回复为 A，golden 为 B。
- `win_ba`：golden 为 A，生成回复为 B。
- 一胜一负计 0.5。

计算指标：

- Strict acceptance within 5 rounds。
- Win-Rate vs golden。
- Mean rounds to success；失败记为 5。
- Parse unresolved。

Judge 也关闭 thinking，只输出规定 JSON。

## 训练和 GPU 调度

- GPU 0 和 GPU 1 的两个任务可以并行。
- 同一 GPU 上训练任务按单 worker 排队。
- 训练、生成和 Judge 不在同一 GPU 上同时常驻不必要的模型。
- 生成完成后释放 Persuader 和 Persuadee 显存，再加载 Qwen3-14B Judge。
- 每次启动前记录 GPU、显存余量、模型 dtype 和量化设置。
- 使用 BF16、无量化；显存不足时保存恢复状态后等待，不改变精度。

## 验收标准

开始完整实验前：

- 两个基础模型 config、tokenizer 和 causal-LM 类在 `ljj` 环境加载成功。
- Persuadee merged 模型在同一环境加载成功。
- GPU 0、GPU 1 显存预检通过。
- Llama 和 DeepSeek Baseline 1 各完成 5 条 dry run。
- DeepSeek dry run 不产生 thinking 内容。
- Baseline 5 的两个架构 smoke test 通过。
- 输出目录为空，或只包含同一实验的可恢复 checkpoint。

每个生成文件完成后：

- 恰好 1,000 行。
- index 完整覆盖 0--999，无重复、无缺失。
- scenario ID 集合和顺序与固定测试集一致。
- 每条 trajectory 最多 5 rounds。
- `persuadee_model` 始终为 `/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged`。
- 不包含 thinking 内容。
- provenance 含模型、adapter、数据 hash、采样参数、GPU 和 thinking policy。

每个 Judge 文件完成后：

- 恰好 1,000 条记录。
- 与生成文件 index 一一对应。
- Judge 为 `/app/hzj/models/Qwen3-14B`。
- prompt revision 为 `joint_strict_v18_json_schema`。
- 保存原始输出、重试输出、parse errors 和最终 verdict。
- unresolved 数量单独统计。

## 默认口径

- SFT 使用 10,000 场景、59,868 turns。
- SimDPO 使用 2,000 preference 场景。
- Baseline 5 使用全量 ISR 的 LatentRATT 隐式推理口径，不改成显式 2,000 场景 RL。
- Persuadee 不重新训练，继续使用现有 5,000 条记录、27,308 assistant turns 的 merged 权重。
- 新实验首轮只评测 Qwen3-14B；Selene、Gemma 和 gpt-5.6-luna 后续再评测。
- 现有 Qwen2.5 主表、历史 Gemma/Selene artifact 和旧协议结果保持不变。
