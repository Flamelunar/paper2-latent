# ECLP–ELPO v2.1：运行说明

> 2026-10-03 当前执行方案已改为 **ECLP + 离线隐式策略优化**。
> SFT 保留现有 reasoning 辅助监督、只跑全量 1 epoch；后续使用 1800 条
> RL 数据和连续 NLL/纠错增益奖励，不进行训练对话生成或 Judge 评分。
> 当前入口是 `run_gpu4_full_chain.sh` → `run_gpu4_offline_pipeline.py`。
> 以 `OFFLINE_ADAPTATION_PLAN.md` 和 `offline_training_plan.json` 为当前
> 训练依据；下文真实交互 effect/ELPO 的章节保留为旧研究方案，未作为
> 当前训练路线，也不能把其“真实交互效应”主张写成当前方法的已实现收益。

方案只读 `references/FINAL_PLAN.md`，核心只读 `eclp_elpo.py`。这不是新的七份并行方案。

代码为主方法研究实现：包含实际模型计算、ISR心理推理辅助SFT、交互采集、打分、PPL shaping、效应拟合、连续策略更新与评估。2026-10-03 修复版已通过 CPU 与远端 A100 BF16 机制检查、原数据审计；真实模型 smoke 与方法效果实验分别报告。现有运行证据见 `references/VALIDATION_REPORT.md`，当前修复步骤见 `REPAIR_PLAN.md`。

## 1. 放置位置与依赖

从原项目根目录运行，目录并列，不覆盖原基线：

```text
exp-2024DailyPersuasion-main/
  exp_code_ljj/                # 原数据、Persuadee/prompt 与评估实现
  baseline5_latentRATT/               # 原始来源方法与修复对照
  my_eclp_final/
    references/FINAL_PLAN.md
    README.md
    eclp_elpo.py               # 全部核心功能
    check_eclp_mechanism.py
    config.json
    requirements.txt
    references/VALIDATION_REPORT.md
    validation.log
    source_audit.json
    manifest.json
```

只复制本文件夹到原项目根目录，不需要复制此包中的测试路径或旧 /mnt/data 路径。请使用**已经能加载原 Persuadee 的环境**。requirements.txt 是兼容范围，不是经过本机 GPU 测试的锁定环境；不要盲目降级已有 Transformers，也不要为通过测试换一个用户模拟器。

```bash
# 使用你已有的 Python 环境；记录当前环境。
python -m pip freeze > before_eclp_environment.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 仅在缺少依赖时补装，并先确认固定 Persuadee 仍能加载：
# python -m pip install -r my_eclp_final/requirements.txt
```

默认 actor 是原 Qwen2.5-7B-Instruct。固定 Persuadee：

```text
/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged
```

`actor_system_prompt` 已显式写入两个配置，当前值与原 baseline 的 `You are a skilled persuader.` 逐字一致；改动 prompt 必须重新创建 protocol lock。

网络 BF16，无 INT8/INT4/FP16。高斯联合概率、Cholesky/SVD、loss 汇总及 Adam 的主副本与矩使用 FP32。PEFT 禁止自动将 adapter 提升为 FP32，且会检查实际网络参数 dtype。

默认 actor=cuda:0、Persuadee=cuda:1、Judge=cuda:1；采集与打分分进程，不同时驻留三模型。单 GPU 可将设备指向同一张卡，但必须通过真实显存冒烟；这里**不承诺 40GB 能同时容纳两模型、长上下文 KV 和激活**。OOM 不允许通过量化或替换 Persuadee 解决。Actor 只支持单 GPU；本版不含 DDP/FSDP 或 CPU offload 训练。

## 2. 首次配置与 P0

`config.json` 的 `judge_model` 故意留空：请填入原评估实际使用且确定为本轮主 Judge 的本地 checkpoint 路径，然后锁定。旧主表有多个 Judge，不能把源脚本的默认路径自动当成你当前主表所用的模型，更不能选择最有利于新方法的那一个。

从项目根目录执行以下命令：

```bash
set -euo pipefail
CFG=my_eclp_final/config.json
WORK=eclp_runs_v2

# 可复用机制回归：四种模式、三个种子、四轮反馈、先验 Jacobian 与来源保护。
python my_eclp_final/check_eclp_mechanism.py --device cuda:0 --output "$WORK/mechanism.json"

# 只读审计；遇到 LFS 指针返回非零。这是明确阻塞，不是假数据训练。
python my_eclp_final/eclp_elpo.py --config "$CFG" audit --output "$WORK/source_audit.json"
```

2026-10-03 远端 5 份原始数据的完整内容与 manifest SHA 已匹配。旧上传快照曾包含 LFS 指针，不能将历史快照状态当成当前服务器状态。不同环境仍需只读审计，不修改原 JSONL 或评估点。

审计通过、模型路径确认后：

```bash
# 首次完整哈希耗时取决于模型大小；不覆盖旧 lock。
python my_eclp_final/eclp_elpo.py --config "$CFG" lock
python my_eclp_final/eclp_elpo.py --config "$CFG" prepare --output "$WORK/prepared"

# 真 GPU + 真模型测试：一次 SFT 前向反向和一次固定 Persuadee 回复。
python my_eclp_final/eclp_elpo.py --config "$CFG" smoke \
 --prepared "$WORK/prepared" --output "$WORK/gpu_smoke.json"
```

GPU 上另有运行时，可在 `smoke` 增加 `--sequential-models`：actor 暂存到 CPU 内存后调用 Persuadee，再回到 GPU 编码真实反馈。所有神经前向均为 CUDA BF16；该选项仅改变 smoke 的模型驻留，不用于训练。修复版现有独立配置是 `config_repair.json`，输出目录为 `eclp_runs_v21_repair`，不要续用 `eclp_runs_gpu4` 的旧 lock/checkpoint。

生产命令始终先验证协议/数据/所需权重。没有 CUDA 时明确退出，不自动用 CPU toy 模型代替。

## 3.1 ISR心理推理与PPL shaping

SFT 读取 ISR 样本中的 Intent to Strategy Reasoning、Selected Strategy 和公开 Utterance。心理推理作为私有辅助目标，通过状态/latent路径计算 reasoning NLL；部署和测试时不会输出该字段。总SFT损失为 utterance + obs_weight*observation + strategy_weight*strategy + reasoning_weight*reasoning。

交互式 ELPO 的主奖励仍是固定 Judge 判定的 success-within-H。采集阶段同时记录参考 Utterance 的 NLL，评分阶段加入有界的 PPL shaping：reward = interaction_reward + ppl_shaping_weight*tanh(reference_nll-candidate_nll)。默认 ppl_shaping_weight=0.1，因此 PPL 只作小幅辅助信号；--reward ppl 仍保留为不调用 Persuadee 的独立控制实验。旧 checkpoint/lock 与新权重不兼容，必须重新 lock、prepare 和训练。

## 3. P2：32 场景 SFT pilot 与开发集

从 SFT 场景中稳定选择 32 个场景，保留对应 session。不是随手取 32 行。

```bash
python my_eclp_final/eclp_elpo.py --config "$CFG" sft \
  --prepared "$WORK/prepared" --output "$WORK/sft_pilot" \
  --limit-scenarios 32 --max-steps 100 --epochs 30 \
  --accumulation 8 --lr 0.0001 --save-every 50

CP=$(python -c 'import json; print(json.load(open("eclp_runs_v2/sft_pilot/last.json"))["checkpoint"])')

# 无新交互的 DEV 参考 NLL 与历史真实反馈预测。
python my_eclp_final/eclp_elpo.py --config "$CFG" validate \
  --prepared "$WORK/prepared" --checkpoint "$CP" \
  --output "$WORK/validate_pilot" --limit 32

# 固定 dev 32 场景、全部五轮；此命令不调用 Judge。
python my_eclp_final/eclp_elpo.py --config "$CFG" generate \
  --prepared "$WORK/prepared" --checkpoint "$CP" \
  --split dev --limit 32 --method ECLP-SFT --output "$WORK/gen_pilot"

# 独立进程只加载 Judge。
python my_eclp_final/eclp_elpo.py --config "$CFG" evaluate \
  --generation "$WORK/gen_pilot" --output "$WORK/eval_pilot"
```

上述 CP 命令使用默认 WORK；改变 work_dir 时相应改变路径。不要通过用 test 替代 dev 来解决 dev 不够理想的问题。

续训到**绝对第 200 步**，并保持原 epochs/lr/accumulation/scenario 子集一致：

```bash
python my_eclp_final/eclp_elpo.py --config "$CFG" sft \
  --prepared "$WORK/prepared" --output "$WORK/sft_pilot" \
  --checkpoint "$CP" --resume --limit-scenarios 32 \
  --max-steps 200 --epochs 30 --accumulation 8 --lr 0.0001 --save-every 50
```

新 checkpoint 的评估输出写新目录。不要覆盖第100步的结果。`--max-steps` 是上限，不是强制在 epochs 耗尽后继续；pilot 的 epochs=30 是为了覆盖100/200步的机械验证，**不是全量30 epoch建议**。

## 4. 结构消融

复制配置文件并仅改变 `state_mode`，分别从同一基础模型初始化：

```bash
python - <<'PY'
import json
from pathlib import Path
base=json.loads(Path('my_eclp_final/config.json').read_text())
for mode in ['no_error','no_correction','stateless']:
    c=dict(base); c['state_mode']=mode
    Path(f'my_eclp_final/config_{mode}.json').write_text(json.dumps(c,indent=2)+'\n')
PY
```

`no_error` 是关键控制：仍看新回复、仍更新状态，只去掉预测误差。`no_correction` 完全不吸收新回复，是极端对照。`stateless` 无递推状态，但观察特征包含历史上下文。它们均不是已经完成的“严格同容量普通四槽位 latent”或“纯自身记忆”基线。

对比实验的训练数据视图、公开监督 token 和反馈曝光相同。不要把不兼容结构的 checkpoint 强行加载，或更换 state_mode 后直接评价旧权重。

## 5. ELPO 第一轮：探测 → F → 未见场景核验

以下命令必须在 ECLP-SFT 通过 P2/P3 后执行，不以随机初始化网络跑昂贵 RL。CP 改为通过开发集选择的 SFT checkpoint，并固定。

```bash
# 使用 rl_train 稳定序列前128个上下文；仅训练池，H=1先探测。
python my_eclp_final/eclp_elpo.py --config "$CFG" collect \
  --prepared "$WORK/prepared" --checkpoint "$CP" \
  --output "$WORK/probe_iso" --offset 0 --limit 128 \
  --horizon 1 --candidates 4 --repeats 2 \
  --exploration isotropic --reward interaction

# F 拟合只使用实际对方反馈特征，不需要 Judge。
python my_eclp_final/eclp_elpo.py --config "$CFG" fit-effect \
  --rollouts "$WORK/probe_iso" --output "$WORK/effect" \
  --epochs 30 --batch-size 32 --lr 0.001

# 新上下文：不参与F拟合，也未用于F选择epoch。
python my_eclp_final/eclp_elpo.py --config "$CFG" collect \
  --prepared "$WORK/prepared" --checkpoint "$CP" \
  --output "$WORK/probe_holdout" --offset 128 --limit 64 \
  --horizon 1 --candidates 4 --repeats 2 \
  --exploration isotropic --reward interaction

python my_eclp_final/eclp_elpo.py --config "$CFG" check-effect \
  --rollouts "$WORK/probe_holdout" --effect "$WORK/effect" \
  --output "$WORK/effect_holdout_check.json"
```

`fit-effect` 的零预测器检查失败时保留失败文件并退出，不擅自加载。通过零基线不代表方向已有效；必须继续做未见场景实际 random/effect 干预，并结合 `strategy_noise_rms_vs_mean` 检查尺度。

F 的反馈编码只用共同动作前上下文与对方回复；网络本身、Judge 和 Persuadee 不因此缺少真实对话。F 的效果指标不是成功率。

## 6. ELPO 训练：独立采集、打分、更新

正式几何对照必须使用同一 SFT 起点、相同总预算。下面只演示 effect 分支。效果方向未通过验证时不要执行。

```bash
python my_eclp_final/eclp_elpo.py --config "$CFG" collect \
  --prepared "$WORK/prepared" --checkpoint "$CP" \
  --output "$WORK/elpo_r0" --offset 256 --limit 128 \
  --horizon 3 --candidates 4 --repeats 2 \
  --exploration effect --effect "$WORK/effect" --reward interaction

# within-H：每个分支对所有H个prefix打分，与主指标定义一致。
python my_eclp_final/eclp_elpo.py --config "$CFG" score \
  --rollouts "$WORK/elpo_r0" --output "$WORK/elpo_r0_scores"

python my_eclp_final/eclp_elpo.py --config "$CFG" update \
  --checkpoint "$CP" --rollouts "$WORK/elpo_r0" --scores "$WORK/elpo_r0_scores" \
  --output "$WORK/elpo_r0_update" --epochs 2 --batch-size 1 --lr 0.00001
```

`update` 的 checkpoint 必须是生成该 rollout 的行为 checkpoint。F/P_s/状态/公开 LoRA 固定，不能对旧数据篡改动作或 old_logp。下一轮从更新后的 checkpoint 重新 collect，不把上一轮 buffer 当成无限离线数据循环训练。

相同起点的 `isotropic / random` 分支也需要相同调用总额；完整 effect 的 F 探测预算算进总额。**方法对照不是各跑128上下文就默认同预算，因为 effect 还用了先前探测。**先分配统一预算，必要时把额外预算用于 isotropic 的真实训练，再比较学习曲线。

`collect --reward ppl` 则不调用环境，使用原训练参考话语；它是连续动作密度下的 PPL 控制，不是原论文 LR-GRPO。正式原方法请保留原工程脚本。

请求账本在 `calls.jsonl` 中先记 started，再记 completed/failed。summary 的 `total_persuadee_calls` 是完整轨迹的推导值，**总成本优先看 `request_ledger` 的 attempted 与 unresolved**，包含中断后可能已执行但未提交成完整 episode 的请求。

## 7. 固定测试与配对统计

先从独立 dev 冻结结构、checkpoint、seed、单动作规则、主 Judge/rubric，再执行测试。默认 action=mean，不在测试中调用 F/Judge 搜索。

```bash
# --limit 0 才是全部固定1000点；有显式 --allow-test 闸门。
python my_eclp_final/eclp_elpo.py --config "$CFG" generate \
  --prepared "$WORK/prepared" --checkpoint "$FINAL_CP" \
  --split test --limit 0 --allow-test --method ECLP-ELPO \
  --output "$WORK/test_eclp_elpo"

python my_eclp_final/eclp_elpo.py --config "$CFG" evaluate \
  --generation "$WORK/test_eclp_elpo" --output "$WORK/test_eclp_elpo_eval"

# 对同协议、同数据、同seed、同Judge的另一已完成方法。
python my_eclp_final/eclp_elpo.py --config "$CFG" compare \
  --a "$WORK/test_eclp_elpo_eval" --b "$MATCHED_BASELINE_EVAL" \
  --metric ever --output "$WORK/paired_success_within5.json"

# 稳定接受补充指标：
python my_eclp_final/eclp_elpo.py --config "$CFG" compare \
  --a "$WORK/test_eclp_elpo_eval" --b "$MATCHED_BASELINE_EVAL" \
  --metric final --output "$WORK/paired_final_round5.json"
```

FINAL_CP / MATCHED_BASELINE_EVAL 是你实际选定的产物路径，不存在时不能运行。未知 Judge 结果保留上下界；比较命令不会删掉不完整样本后给显著性。

本包内部以一个锁定 Judge 完成采集回报/主评估，**独立 Judge 复核需通过原项目评估接口对同一轨迹执行并明确记录另一身份**；不要改现有 lock 伪装成同一 Judge。本包没有自动实现双 Judge 一键混评或人类心理真值评估。

修复版最终证据：`eclp_runs_v21_repair/gpu_smoke_repaired.json`。actor 的重复惩罚读取被模型 SHA 锁定的 generation_config，并对真实 prompt 与新 token 同时生效；连续槽位不计作词表 token。

## 8. ECLP 训练梯度契约

代码为每个 live state 保留 `B` 和 `policy_B` 两条路径。`B` 用于真实反馈预测 MSE；`policy_B` 在数值上复现同一状态，但 public loss 只允许通过 correction、gate、posterior normalization 和 ECLP 的 error projection 更新校正路径。transition、prior normalization、feedback predictor、slots、observation readout 和 `to_memory` 不接受 public loss 梯度。`no_error` 去掉 error projection 输入，`no_correction` 与 `stateless` 不产生 correction 梯度。

`smoke` 会逐参数检查这项隔离，并报告 `public_state_gradient_norms`、correction 梯度和 error projection 梯度。这项检查是当前创新闭环的必要验收，不应再用“public loss 对整个 state module 梯度全为零”作为旧断言。

私有 ISR reasoning 仍走完整 `B` 路径，可更新预测器；因此这里的隔离仅针对 public loss。policy prior 冻结参数且保留真实的状态输入导数，不使用直通近似。实现 SHA 在模块加载时固定，checkpoint 同时核验版本和来源；旧进程在文件更换后保存的 SHA 可能错标，不能据此迁移为修复版。

## 9. 代码导航与实际覆盖

| 对象 | 核心实现 |
|---|---|
| 原始 ISR → 公开时间视图 | `parse_isr`, `parse_session`, `prepare_data` |
| BF16与身份固定 | `cuda_bf16`, `precision_report`, `create_lock`, `verify_lock` |
| 状态预测/校正 | `EvidenceState`, `ECLPRunner.consume/replay` |
| 连续策略与输入注入 | `StrategyPolicy`, `LatentHeads`, `ECLPRunner.context/utterance_loss/speak` |
| SFT/断点续跑 | `command_sft`, `FP32MasterAdamW`, `save_checkpoint` |
| 效应方向 | `EffectModel`, `exploration_cholesky`, `command_effect`, `command_check_effect` |
| 固定动作与RL | `gaussian_log_prob`, `group_advantages`, `guarded_mean_step`, `command_update` |
| 真实配对交互 | `BaselineBridge.respond`, `ECLPRunner.rollout`, `command_collect` |
| 成本与unknown | `CallLedger`, `within_h_success`, `judgment_summary` |
| 评估/统计 | `command_generate/evaluate/compare`, `paired_bootstrap` |

真实预训练 CUDA smoke 已通过；仍未完成的项目：分布式训练、严格参数匹配的 flat-latent 基线、文本空间 RL 基线、PCA 方向、人工反证子集与安全质量自动评分、所有方法正式三 seed 大训练。主方法代码存在不代表这些实验已完成。

## 10. 给 Codex 的本轮执行任务

把下面内容作为本轮 Goal 的任务正文；不要要求它一次性“做到超过 MA2P 才停止”。

```text
阅读 my_eclp_final/references/FINAL_PLAN.md、my_eclp_final/README.md、my_eclp_final/references/VALIDATION_REPORT.md 和 my_eclp_final/eclp_elpo.py。
以 FINAL_PLAN.md 为唯一方案执行依据，旧文件仅作来源，不重新拆分两篇论文。

本轮目标只完成 P0–P2：真实 BF16 GPU 集成验收，以及固定32个SFT场景、最多200 optimizer steps的pilot。
Persuadee 必须保持原 Qwen3.5-9B-persuadee/merged 的权重、tokenizer、system/user prompt、清理及采样协议。
任何FP16/INT8/INT4、替换模拟器、修改原始数据/旧结果、测试集选checkpoint、放宽Judge的行为均不允许。

步骤：
1. 只读核验原始数据SHA、LFS对象、已有模型路径/环境和GPU；主Judge从现有评估配置确认，不自行选择新模型。
2. 运行测试并记录PASS/SKIP/FAIL；运行真实smoke，证明连续输入梯度、状态/公开梯度分工、生成和固定Persuadee实际回复。
3. P1通过后按README训练100steps，固定dev32做离线预测和五轮生成/评估；最多续至200steps。
4. 修复任何集成错误需附回归测试与差异说明；不要修改科学机制和协议来让测试通过。
5. 保存 PLAN.md、RUN_STATUS.json、GPU_SMOKE.json、PILOT_REPORT.md、真实日志、checkpoint、配置与环境指纹。

注意：当前回归使用 `check_eclp_mechanism.py`；真实 smoke 与成功率评估分别记录。历史 76 项测试统计不在本轮重新声称已执行。
显存不足、原数据缺失、模型身份不一致、未来信息泄漏、参数梯度串路时报告BLOCKED，说明精确阻塞。
保留所有失败/未知案例与成本，不填写尚未运行的主表数字。
本轮达到P2预算即停止，不擅自进行全量训练或大规模ELPO。
```

之后另开 P3 结构对照 Goal，再开 P4/P5 效应验证与同预算 RL Goal。代码缺陷修复和科学结论要分开记录。
