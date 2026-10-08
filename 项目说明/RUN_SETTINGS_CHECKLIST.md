# LJJ 主实验统一运行设置清单

状态：已确认，2026-10-02。

扩展计划：保留当前 Qwen2.5-7B-Instruct Persuader 主表；新增 Meta-Llama-3.1-8B-Instruct 和 DeepSeek-R1-Distill-Qwen-14B 两张独立 Persuader 结果表。三张表不混合数值。

## 1. 项目目录

- `0-paper-2024DailyPersuasion/`：原始 DailyPersuasion 论文、数据与复现材料。
- `exp_code_ljj/`：Baseline 1--4 的训练、五轮生成、Judge 评测、结果与主表。
- `baseline5_latentRATT/`：Baseline 5 LatentRATT 隐式推理方法的复现代码、训练脚本与评测脚本。
- `exp_code_ljj/artifacts/generation_5rounds/`：各方法固定 1,000 个测试场景的五轮生成结果。
- `exp_code_ljj/results/main_experiment_table.md`：主实验结果表。
- `exp_code_ljj/results/main_experiment_table_llama31.md`：Meta-Llama-3.1-8B-Instruct Persuader 表。
- `exp_code_ljj/results/main_experiment_table_deepseek14b.md`：DeepSeek-R1-Distill-Qwen-14B Persuader 表。

## 2. 数据来源与固定拆分

- 原始数据：DailyPersuasion，共 13,000 个场景。
- 拆分随机种子：42。
- Persuader SFT：10,000 个场景，共 59,868 个对话 turn。
- Preference 数据：2,000 个场景，供需要 preference/RL 数据的方法使用。
- Test：1,000 个场景。
- 三个场景集合不得重叠；所有方法必须使用相同的 1,000 个测试场景和相同顺序。
- 已核验 `dialogue_sft.jsonl`、`isr_sft.jsonl`、`user_model_sft.jsonl` 均为 59,868 行、10,000 个唯一场景。
- 已核验 `preference_seeds.jsonl`、`simulation_preferences.jsonl` 均为 2,000 行、2,000 个唯一场景。
- 已核验 `evaluation_points.jsonl` 为 1,000 行、1,000 个唯一场景。
- 现有五个五轮生成文件的场景顺序一致，场景序列 SHA256：
  `41ad0be519824efe135fb46170de409297e110a81f61096a2c9071884f9f4856`。

## 3. Persuader 方法设置

模型路径：Qwen2.5-7B-Instruct（`/app/hzj/models/Qwen2.5-7B-Instruct`，当前主表）；Meta-Llama-3.1-8B-Instruct（`/app/hzj/models/Meta-Llama-3.1-8B-Instruct`，新增表 A）；DeepSeek-R1-Distill-Qwen-14B（`/app/hzj/models/DeepSeek-R1-Distill-Qwen-14B`，新增表 B）。

模型设置合理：Llama 3.1 是与当前 Qwen2.5 接口最接近的标准 causal LM，建议先运行；DeepSeek 14B 是 Qwen2 架构，BF16 显存和延迟更高。所有模型均关闭 thinking/reasoning 输出，只保留最终答案文本；不得把 `<think>...</think>` 内容送入对话或 Judge。

统一实验环境（所有训练、生成、本地评测和 API 请求脚本）：`/app/hzj/anaconda3/envs/ljj/bin/python`。环境版本：PyTorch 2.6.0+cu118、Transformers 5.12.1、PEFT 0.19.1。旧环境 `/app/hzj/.conda/envs/verl/bin/python` 不再作为入口。

| 方法 | Persuader 设置 | 训练数据口径 |
| --- | --- | --- |
| Baseline 1: Zero-shot | Qwen2.5-7B-Instruct | 不微调 |
| Baseline 2: Dialogue-SFT | Qwen2.5-7B-Instruct Dialogue-SFT | 10,000 场景、59,868 turns |
| Baseline 3: ISR-SFT + SimDPO | ISR-SFT 后执行 SimDPO | ISR-SFT 使用 59,868 turns；SimDPO 使用 2,000 preference 场景 |
| Baseline 4: MA2P | Qwen2.5-7B-Instruct + MA2P prompt，K=0 | 不微调 |
| Baseline 5: LatentRATT | LatentRATT 隐式推理 checkpoint | SFT 与 LR-GRPO 使用全部 59,868 条 ISR 数据；不要求改为 2,000 场景的显式 RL 口径 |
| Our | 待填写最终方法和 checkpoint | 沿用相同拆分，并完整记录实际训练数据 |

Baseline 5 应在论文和表格中描述为“LatentRATT 隐式推理方法”。主表的 Persuader model 字段必须填写可追溯的模型或 checkpoint 名称，不能只写 `rl`。

## 4. 统一 Persuadee

所有 Baseline 与 Our 在生成测试对话时必须使用同一个冻结的 Persuadee：

- Base：`/app/hzj/models/Qwen3.5-9B`
- Adapter：`/app/hzj/liujianjian/Qwen3.5-9B-persuadee/adapter`
- Merged：`/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged`
- 训练目标：`scenario + dialogue history + current persuader utterance -> persuadee response`
- 训练数据：5,000 条记录、27,308 个 assistant turns。
- 训练方式：仅监督式 user-model SFT；不使用 ISR target，不使用 DPO。
- 训练参数：2 epochs，max length 2,048，batch size 1，gradient accumulation 16，learning rate `1e-4`，warmup ratio 0.03，seed 42。
- LoRA：r=16，alpha=32，dropout=0.05；目标模块为 q/k/v/o、gate/up/down projection。
- Persuadee 在所有对比方法中冻结，不因 Persuader 方法而变化。

注意：Persuadee 的 5,000 条训练记录与 Persuader SFT 的 10,000 场景属于不同实验角色，不要求数量相同。

## 5. 五轮生成协议

- 每个场景最多 5 rounds，一条完整轨迹最多包含双方 10 条发言。
- 测试场景数：1,000。
- `temperature=0.95`
- `top_p=0.7`
- `top_k=50`
- `max_new_tokens=512`
- 各方法只替换 Persuader；Persuadee、测试输入、轮数与采样设置保持一致。
- 每条结果应保存 index、scenario ID、方法名、实际模型路径、生成协议及完整五轮轨迹。
- Baseline 5 主生成文件：
  `exp_code_ljj/artifacts/generation_5rounds/baseline5_latentRATT.json`。
  该文件扩展名为 `.json`，文件内容仍采用一行一条记录的 JSONL 格式，共 1,000 条。

## 6. Judge 与主表指标

Judge 分组：

- Selene-1-Mini-Llama-3.1-8B，BF16，strict V17 v7。
- Gemma 4 E4B-it，BF16，prompt v11。
- Qwen3-14B，BF16，v11 判定标准与严格 JSON 输出协议。
- gpt-5.6-luna，通过 API 评估；API 模型名、版本、请求参数和完整响应写入 artifact provenance。

Selene 当前下载目标为（`/app/hzj/models/Selene-1-Mini-Llama-3.1-8B`），模型未完整落盘前不得启动本地 Selene 评测。Gemma 4 E4B-it 继续保留在当前 Qwen2.5 主表；新增两张表预留相同 Judge 行，但不复制旧数值。

统一指标：

- Strict acceptance within 5 rounds。
- Win-Rate vs golden：比较全部五条 Persuader 回复与单条 held-out golden 回复；交换 A/B 顺序评判两次，split vote 计 0.5。
- Mean rounds to success：使用首次成功轮次；失败记为 5。
- Success parse unresolved。
- 每个 parse 或 evidence validation error 最多额外重试 3 次。
- Success rate、Win-Rate 与 mean rounds 仅基于成功解析的样本；N 和 unresolved 同时报告。

## 7. 主表填写规则

- 主表位置：`exp_code_ljj/results/main_experiment_table.md`。
- 每个 Judge 分组固定依次列出 Baseline 1--5 和 Our。
- `Our` 在结果产生前保留空值，不使用占位实验数字。
- 同一方法在三个 Judge 分组中的 Persuader model 名称必须一致。
- 缺失结果用 `-`，不能混用旧的一轮 Win-Rate 结果。
- 对旧结果保留原始 provenance；路径迁移不改写历史 Judge JSONL 和日志内容。

## 8. 复现与验收检查

- [ ] 训练前记录数据文件、行数、唯一场景数和哈希。
- [ ] 生成前记录 Persuader checkpoint 与统一 Persuadee merged 路径。
- [ ] 验证 1,000 个测试 ID 的集合、顺序与固定 SHA256。
- [ ] 验证每个样本最多 5 rounds，并检查中断或重复样本。
- [ ] Judge 使用固定模型、精度、prompt revision 和重试规则。
- [ ] 主表数值可以从对应 artifact 重新计算。
- [ ] 新实验不得覆盖历史日志和历史 Judge 输出。
- [ ] 三个 Persuader 表使用相同 1,000 个测试场景和统一 Persuadee（`/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged`）。
- [ ] DeepSeek-R1 和其他模型均确认关闭 thinking，记录最终答案过滤规则。

## 9. Llama / DeepSeek 新实验执行补充（2026-10-04）

本节覆盖新增两张表的旧 Judge 占位设置，不改变历史 Qwen2.5 结果：

- 仅 GPU 0（Meta-Llama-3.1-8B-Instruct）和 GPU 1（DeepSeek-R1-Distill-Qwen-14B）运行这两套实验，不借用 GPU 2/3/4。
- 训练、生成和 Judge 均使用 BF16、无量化；入口（`/app/hzj/anaconda3/envs/ljj/bin/python`）。
- 新评测只运行 Qwen3-14B（`/app/hzj/models/Qwen3-14B`）和 Selene（`/app/hzj/models/Selene-1-Mini-Llama-3.1-8B`），均使用 `--prompt-version v18`。Gemma/API 不运行，API 表格占位保持空值。
- Persuadee 始终冻结使用（`/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged`）。
- 固定数据：SFT（`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/data/dialogue_sft.jsonl`）、ISR（`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/data/isr_sft.jsonl`）、preference 场景（`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/data/preference_seeds.jsonl`）、测试（`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/data/evaluation_points.jsonl`）。数量与拆分仍为 10,000 场景/59,868 turns、2,000 preference 场景和 1,000 测试场景。
- Baseline 5 的 SFT checkpoint 单独保存在对应 family 的 `baseline5_latentratt/sft/`；完成全量 LR-GRPO 后，最终 adapter、attention 和训练状态保存在 `baseline5_latentratt/` 根部。两阶段分别核验数据 hash、模型、GPU、精度与完成步数，不允许把 SFT 文件存在误判为整套训练完成。
- Baseline 5 对每个架构先分别进行一步 SFT 和一步 LR-GRPO forward/backward smoke test；失败就停止该 family 的 Baseline 5。恢复时选择同阶段的有效 checkpoint，并加载 optimizer、RNG 和数据游标。
- DeepSeek-R1 的模板实际忽略 `enable_thinking=False`；统一使用空 thinking-block 的 assistant 前缀进入 final-answer 分支，并在公开对话写入前清理 thinking。事后 normalization 同时覆盖 trajectory、final_history 和 MA2P agent trace。
- Judge 显式启用 `ALLOW_JUDGE_EVAL=1`，解除继承的 CUDA mask 后使用物理 GPU 0/1。每个 Judge/family 的 5 个完整评测文件通过核验后，由（`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/report_new_baselines_v18.py`）更新对应新表；不修改原 Qwen2.5 主表，也不填写 Our/API 数值。
- 表中 N 为成功解析数，unresolved 单列；成功率、双顺序 Win-Rate 和平均轮数只用成功解析样本计算。失败轨迹计 5 轮，未解析样本不能当作失败。

## 10. 偏好模拟恢复与执行验收补充（2026-10-04）

- 两个 family 的偏好模拟文件独立存放：（`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/artifacts/preference_simulations/meta_llama31/`）和（`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/artifacts/preference_simulations/deepseek_r1_qwen14b/`）。SimDPO 只消费本 family 的 `simulation_preferences.jsonl`，并保存数据 hash 与模拟配置。
- 模拟协议为 `persugpt_simulation_table5_reconstruction_v4_final_answer`。在公开回复进入另一方下一条提示词之前清理完整、closing-only 或未闭合的 thinking；没有公开回复时由同一个模型重写，最多额外 3 次，仍为空则停止。保存真实过滤计数，不将清理后的旧污染路径冒充重新模拟。
- Llama 旧模拟共 1,024 条，其中保留 356 条完全干净的候选和路径；隔离 668 条受污染场景并重新完整模拟。隔离位置（`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/artifacts/preference_simulations/meta_llama31/quarantine_v3/`），审计（`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/artifacts/preference_simulations/meta_llama31/recovery_audit.json`）。仍覆盖同一固定 2,000 个 preference 场景。
- 当前安装的 TRL 对已加载的 ISR PEFT policy 自动复制 `default` adapter 为冻结的 `ref` adapter，SimDPO 参考行为是初始 ISR policy，不是禁用 adapter 后的裸 base model。训练前的源 adapter 与 preference 数据必须属于同一 Persuader。
- MA2P 的 Perception、World Model、Persuader 与 Persuadee 均在内容转交下一角色之前清理 thinking；agent trace 只保存最终公开内容和真实过滤标记。已审计原 Llama 93 条部分生成，未发现 thinking 标记或被 private JSON 改变的 agent 输入，可继续复用。
- LatentRATT 的 SFT、辅助损失、LR-GRPO loss/reward 与梯度必须有限；NaN/Inf 立即失败。checkpoint 记录 `numerical_validation=finite_loss_reward_gradient_v1`、最后 loss/reward/gradient norm；小样本 smoke 不能仅凭文件存在而通过。
- 两个 family 独立等待自己的五份完整生成；评测前核验固定场景、golden、初始和最终历史、五轮结构、采样参数及公开内容，再等待本 GPU 至少 50,000 MiB 空闲。每张卡先 Qwen3-14B v18，再 Selene v18；不等待另一 family 完成，也不与残留生成模型争用显存。
- 运行清单的实际路径（`/app/hzj/liujianjian/ljj-2-paper/项目说明/RUN_SETTINGS_CHECKLIST.md`）；主计划（`/app/hzj/liujianjian/ljj-2-paper/1_plan_llama+ds_baselines.md`）。恢复补丁不改变固定数据或原 Qwen2.5 主表。
- 严格历史一致性验收发现 Llama B2 的场景 index 455、478、618 存在 trajectory/final_history 文本不一致。这 3 条不能靠重写历史冒充有效生成；在本 family 的模拟完成、释放模型后，SimDPO 入口先加载原 B2 adapter 和统一 Persuadee，在 GPU 0 重新生成各自完整五轮，保存原始隔离记录和可恢复缓存；验证成功后原子替换这 3 条，其余 997 条保持原样。代码（`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/repair_baseline2_history.py`）。三条实际重生成并通过验收前，Llama B2 仅算文件齐全、不算最终验收完成。
- 五轮生成的单场景和 batch 路径均先剥离 private thinking，再提取 ISR Utterance；private 块中的 Utterance 不得被误提取为公开答案。空公开回复与强制 direct-answer 重写最多额外 3 次。固定 initial_history 属于源数据，不得在 normalization 中被裁剪或改写。

## 11. 训练配置隔离与历史记录审计（2026-10-04）

- 原训练后端对项目外的 adapter 目录只使用 stage 名生成中央配置名，导致两族的 `baseline2_dialogue_sft.json`、`baseline3_isr_sft.json` 中央记录发生覆盖。模型权重和 generation 目录本身仍按 family 独立，不因此混用结果；旧中央记录保留，不冒充完整历史记录。
- 后续新启动的 SFT/SimDPO 进程使用完整解析后的 output 路径 SHA256 隔离中央配置名，并在对应 adapter 根目录原子保存 `run_config.json`、`training_provenance.json`。记录实际 CUDA mask、可识别的物理 GPU、进程、时间、BF16/无量化和实际数据 hash；哈希与已有值不一致时拒绝继续。不重启当前正在执行的 worker。
- 对此前缺少原始 per-run 配置的 SFT，可运行（`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/audit_saved_sft_metadata.py --write`），入口（`/app/hzj/anaconda3/envs/ljj/bin/python`），训练依赖（`PYTHONPATH=/tmp/ljj_train_env`）。它只读保存的 `training_args.bin`、adapter config、metrics、checkpoint state 并核验当前固定数据，保存 `training_metadata_audit.json`；不加载模型，不初始化 CUDA，不修改权重、checkpoint 或原始 runtime config。
- 该历史审计必须标为 `retrospective_audit_not_original_runtime_record`：当前数据 hash 与 manifest 一致不等于找回原始运行 hash；计划指定 GPU 不等于恢复历史 CUDA mask。无法恢复的事实保留为缺失，不编造。训练完成状态仍结合实际训练日志、完整步数、metrics 与输出验收判断。
