# 修复版验证报告（2026-10-03）

对象：远端项目 `/app/hzj/liujianjian/ljj-2-paper/my_eclp_final` 的 v2.1 修复。
本次结果与旧 v2.0 历史记录分开；不沿用没有在本轮重跑的“76 项测试通过”结论。

## 已实际完成

| 检查 | 结果 | 范围 |
|---|---|---|
| Python 编译、CLI、配置 JSON | 通过 | 远端 Python 3.11.15；14 个 CLI 子命令 |
| CPU BF16 机制回归 | 通过 | 4 种模式 × 3 seeds × 4 轮真实张量计算；双路径逐位相等 |
| 远端 A100 BF16 机制回归 | 通过 | public 梯度不进入预测/readout 参数；ECLP/no_error 校正梯度存在；误差投影仅 ECLP 有梯度 |
| 多轮与先验导数 | 通过 | 最后 public loss 可达首轮校正；policy prior 的输入 Jacobian 与真实 prior 一致 |
| 实现指纹与旧版本拒绝 | 通过 | 导入后替换磁盘文件不改变进程指纹；错标 SHA 的旧版 checkpoint 仍被拒绝 |
| 原数据只读审计 | 通过 | 5 个原 JSONL SHA 均与 manifest 匹配；不是 LFS 指针 |
| 真实 actor/Persuadee smoke | 两轮均通过，最终含 prompt 重复惩罚修复 | 真实 Qwen2.5-7B BF16 前向/反向、优化器、公开生成、固定 Persuadee 回复及状态更新通过 |
| 任务成功率、ELPO 效果、多 seed 对照 | 未执行 | 当前不能报告提升幅度、显著性或 CCF-A 录用概率 |

环境：Python 3.11.15、PyTorch 2.6.0+cu118、Transformers 5.12.1、PEFT 0.19.1。远端为 NVIDIA A100 80GB，机制检查仅使用小张量；它不等于真实大模型验收。

## 修复与证据

- 公开目标现在直接训练校正器，反馈预测参数与 public loss 隔离。
- policy prior 冻结参数，保留实际状态导数；去掉 BF16 差值相加和 identity straight-through 近似。
- 私有 ISR reasoning 仍通过完整 B 路径训练，不能将该版本描述为“预测器仅由观察 MSE 更新”。
- actor system prompt 为原 baseline 的 `You are a skilled persuader.`，进入 protocol lock。
- 模型身份按内容 SHA 锁定，绝对挂载路径记录为元数据；不将路径字符串当成身份。
- checkpoint 使用进程启动时的实现 SHA，同时核验 version 与 provenance；不自动继承旧 lock/权重。
- 修复 inputs_embeds 生成的 prompt 重复惩罚遗漏：使用 actor 原 generation_config 的 1.05，对真实公共 prompt 和生成 IDs 按 HF 同一规则处理；不更换 Persuadee 或原 baseline 采样。
- generic config 修正 repo_root 和实际 actor 路径；修复版使用独立 config_repair.json。

远端证据目录：`eclp_runs_v21_repair/`。

- `source_audit.json`：完整原数据 SHA 检查。
- `mechanism_a100.json`：第一轮 A100 机制检查。
- `mechanism_final.json`：第一轮 smoke 对应源码检查。
- `mechanism_repaired.json`：包含生成协议补丁的最终源码检查。
- `repair_smoke.log`、`repair_smoke_final.log`、`RUN_STATUS.json`：锁定、prepare、真实 smoke 的日志与实时状态。
- `gpu_smoke.json`：第一轮真实 smoke。
- `gpu_smoke_repaired.json`：生成协议补丁的最终真实 smoke，仅成功完成后生成；不会用旧 `eclp_runs_gpu4/gpu_smoke.json` 冒充。

## 旧训练的限制

PID 872814/872813 继续运行 v2.0 ECLP SFT，输出仍在 `eclp_runs_gpu4/sft_full`。基线数据和旧结果没有改写。该进程不获得已上传的修复。
旧实现保存 checkpoint 时会重新读取磁盘脚本 SHA；上传后保存的旧 checkpoint 可能有 implementation SHA 错标。新版本会通过版本与训练 provenance 联合检查拒绝它们，不能把这些权重当成修复版实验结果。

## 科学判断

目前可以验证机制与集成是否按定义工作，不能验证 ECLP/ELPO 是否提高劝说成功率。高水平投稿仍需同容量/同教师监督对照、no_error 与状态干预，以及 isotropic/random/effect 同预算真实交互比较。现有教师 reasoning 和 PPL shaping 必须在论文中明确披露并消融；ELPO 目前优化的是首动作、后续策略冻结，不能写成完整多轮 policy gradient。

## 最终源码验收

最终实现 SHA256：`8c5402f92e5cbd8e1a7ed2e63fab175c5e00c91a983251fce4ddaadfde3b9d14`。

真实 public 梯度范数：correction.0.weight=0.452840、gate.weight=0.096022、error_projection.weight=0.017022；predictor/readout/transition/slots 全部为 0。

`state_values_bitwise_equal`、真实 prior 输入 Jacobian、prompt 重复惩罚与 HF 一致性及启动 SHA 保护均通过。新版真模型 smoke 与最终机制检查记录同一个实现 SHA；task_effectiveness_verified 仍为 false。
