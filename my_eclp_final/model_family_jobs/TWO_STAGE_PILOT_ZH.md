# 两阶段推理试验（2026-10-09）

本试验由用户新增授权，区别于此前取消的公开分析版本。训练代码、数据和SFT/RL权重均不修改；使用同一完整Llama ECLP checkpoint。BF16，无量化。

每个生成回合包含两个独立生成阶段：

1. 完整ECLP上下文、当前隐式立场和策略模块生成简短私有JSON摘要：当前可观察顾虑concern、事实支持的理由reason、可行下一步next_step。提示总长度不超过60词，格式检查拒绝超过80词的摘要；无可靠顾虑时输出unknown。不给gold策略、心理标签、未来回复或golden。
2. 将摘要作为非事实性指导加入公开回复提示，同一个完整ECLP模型重新计算状态条件策略并生成实际说给对方听的话。公开回复需自然包含相关理由，但不能输出分析标签、JSON或第三人称旁白。

摘要单独保存在private_note及原始请求记录中，不加入history，不提供给模拟被劝说者或Judge。隐式状态仍仅用公开双方回复更新。第一个格式合格的摘要与回复被采用，Judge不参与候选选择。

同一固定192合同中的前20个场景用于本次对照；初始history为空。与正在运行的单阶段Our和PersuGPT共享模型权重（各自方法权重固定）、同一模拟被劝说者、输入目标约定、golden和Judge规则。两阶段每轮多一次私有摘要调用，推理成本不同，需要公开披露。

生成：GPU1运行Our，GPU4运行同一模拟被劝说者。评估：GPU1 Selene、GPU4 Qwen3，并行处理。现有批间调度暂时冻结，已运行的Judge自然完成；两阶段20条试验完成或生成/评估失败清理后，自动恢复原Our/PersuGPT的192条任务。不修改正式表为最佳变体，不根据测试分数选择checkpoint。

脚本：my_eclp_final/model_family_jobs/generate_llama_two_stage.py、run_two_stage_pilot.py。

结果：exp_code_ljj/results/llama_two_stage_pilot_comparison.md；详细合同、请求、状态：exp_code_ljj/artifacts/llama_two_stage_pilot_20261009/meta_llama31/。

20条分数只是诊断结果，不能保证完整192条提升。后续扩展需保留版本身份，不能选择性报告更高的Judge或指标。
