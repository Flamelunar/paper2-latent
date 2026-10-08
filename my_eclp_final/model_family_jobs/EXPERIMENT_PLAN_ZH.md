# 2026-10-07 两项任务安排

本项目不使用 GPU0。GPU0 当前有 ljj-1-paper 的其他训练，是否暂停等待用户明确回复；不得停止 GPU4 其他任务。

## 任务一：两个骨干的 baseline 评估
GPU4 上 Selene-8B 与 Qwen3-14B 两条队列并行，每条队列依次评估完整192条的方法。使用 seven_methods_192 最新目标成功、Win@1、Win-Full 提示词，BF16，batch=2，显存预算分别22/32GiB。
输出：exp_code_ljj/artifacts/model_family_latest192_20261007/{meta_llama31,deepseek_r1_qwen14b}/。
旧生成逐字复用，核对固定原始索引、场景、空初始历史、golden和10个发言。发现角色串话会写入 *_audit.json，相关分数仅作旧输出诊断，不能称为修复后的正式成绩。已发现 Llama zero-shot 42/192 含换行 user/assistant 串话。未满192的方法不启动评估，LatentRATT缺失不能替代。

## 任务二：两个 Our 骨干训练
分别使用 Llama-3.1-8B、DeepSeek-R1-Distill-Qwen-14B，独立配置、模型锁、数据准备、LoRA和隐式模块，不复用Qwen2.5权重。
GPU1 两个控制器已启动。先进行模型/数据指纹冻结及数据准备。训练准入要求空闲显存至少31/44GiB，资源不足等待，不停止其他任务。当前已有约20GiB占用，因此不能保证双训练同时装入显存。
SFT保持59,868训练会话，1 epoch、7484步、累积8、学习率1e-4；离线RL保持1800场景、200开发场景、4个高斯候选、原correction奖励、无真实对话和Judge训练调用。原GPU4离线代码不改写，增加独立GPU1入口。
目录：eclp_runs_{meta_llama31,deepseek_r1_qwen14b}_20261007；job_status.json记录阶段，sft.log/offline_rl.log记录执行，失败明确停止。
训练控制器已串接SFT到RL。固定192的Our纯劝说生成、双Judge和表格回填仍需接入，尚未声称完成。评估含污染的旧生成不会自动覆盖正式表格。
