# 当前版本的提示词索引

这里列出实际运行时构造提示词的代码位置。所有占位变量由每条 JSONL 场景填入，函数里的字符串是准确的运行版本。Judge 提示词统一按“角色说明、输入内容、判断标准、输出要求”分段。Selene V17 的严格版提高了成功判定门槛；Win-Rate 规则不变。旧首轮 Win-Rate 记录只用于复用五轮成功判断。

| 用途 | 函数或文件 | 核心内容 |
| --- | --- | --- |
| Gemma 4 V11 五轮成功率与 Win-Rate | `evaluate_joint_prompt_probe.py` 的 `joint_prompt()`；规则见 `prompts/joint_judge_prompts.json` 的 `gemma4_v11` | `joint_gemma_v11_strict_json_v2`；V11 判定尺度、固定 JSON schema、逐轮原文证据和双顺序 Win-Rate 保持不变 |
| Selene 五轮成功率与 Win-Rate | `evaluate_joint_prompt_probe.py` 的 `selene_round_prompt()`、`selene_win_prompt()`；规则见 `prompts/joint_judge_prompts.json` 的 `selene1_v17` | `joint_selene_success_reference_v17_v7_strict_goal_acceptance`；只把明确接受目标或作出确定承诺计为成功，考虑、询问、试用后再决定等不计，按轮评估与 JSON 布尔值输出保持不变，Win-Rate 规则不变 |
| Dialogue-SFT 训练系统提示词 | `repro/constants.py` 的 `SYSTEM_PERSUADER` | `You are a skilled persuader.` |
| 训练数据首轮场景提示词 | `repro/data.py` 的 `scenario_header`，及 `data/dialogue_sft.jsonl` 的 `instruction` / `history` | 背景、目标、`Give the next persuader utterance only.`；后续轮次接入真实历史 |
| Qwen assistant-only loss 模板 | `repro/train_backend.py` 的 `_qwen_assistant_mask_template`；已训练模型保存的 `models/dialogue_sft_adapter/chat_template.jinja` | 使用 Qwen 原生 chat template，并用 `{% generation %}` 标记 assistant 回复用于损失计算 |
| 严格统一 Judge v18 | `evaluate_joint_prompt_probe_main_table.py` 的 `joint_prompt()` / `parse_joint()`；规则见 `prompts/joint_judge_prompts.json` 的 `strict_v18` | `joint_strict_v18_json_schema`；五轮布尔数组、双顺序 A/B、严格 JSON、拒绝额外字段和非单调成功序列 |
| 五轮劝说者生成 | `repro/persuadee_simulation.py` 的 `dialogue_prompt` | 场景背景、目标、历史对话、只写下一句劝说者回复 |
| 固定 persuadee 回复 | `repro/simulation.py` 的 `user_prompt` | 以场景中的 persuadee 身份自然回复；可拒绝、犹豫、询问或接受，不得虚构个人事实 |
| 五轮严格成功率 | `evaluate_baselines.py` 的 `strict_success_prompt` | 每轮前缀独立判断；只有 persuadee 明确且无条件接受准确目标才算成功，含糊情况为失败 |
| 新 Win-Rate (vs golden) | `evaluate_baselines.py` 的 `all_persuader_replies`、`all_replies_judge_prompt` | 五轮劝说者回复合成一个候选，与单条 golden 交换 A/B 顺序各比较一次；分歧记 0.5 |

Baseline 3 的训练与偏好模拟提示词位于 `repro/preference_optimization.py`；其正式测试生成与 Judge 提示词和 Baseline 1/2 共用上述代码。`with_isr` 仅用于从模型原始输出抽出公开发言，不改变 `dialogue_prompt`。

Judge 的原始回答和重试回答保存在 `artifacts/success_judgments/`。新版 Win-Rate 输出目录为 `artifacts/win_judgments/`，目前为空。
