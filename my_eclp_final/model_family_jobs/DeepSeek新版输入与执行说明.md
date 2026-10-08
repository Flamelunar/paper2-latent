# DeepSeek新版输入与执行说明

用户指示日期：2026-10-08。Qwen和Llama保留既有设置；此版本仅用于后续DeepSeek实验。

- 固定192条：target_turn==1且原始history为空，五轮保留本次公开历史。
- 六种方法：Zero-shot、Dialogue-SFT、PersuGPT、MA2P、LatentRATT、Our（仅公开劝说输出）。
- 所有劝说方接收背景、精确目标、公开历史；模拟被劝说者也接收目标。“接收目标”是输入目标信息，不是强制同意目标。
- 五种baseline不接收数据集候选策略、gold已选策略、ISR推理标注或未来回复。MA2P保留自行推断和制定策略的三角色K=0流程。
- Our暂按场景`strategies`候选列表解释策略输入；不读取gold已选策略。原异步澄清尚未收到回复，如用户指定另一含义，Our生成前需更新独立合同。
- 训练与权重不因提示词更新而改变。Our和LatentRATT需对应DeepSeek骨干训练完成后才生成。
- 所有192条重新生成，不复用旧版本；Selene和Qwen3沿用现有评估提示词和指标。
- 仅物理GPU4，不停止其他项目任务。显存不足则排队，不假称任务已经开始生成。

远端代码：`my_eclp_final/model_family_jobs/generate_deepseek_goal_strategy.py`、`deepseek_goal_strategy_campaign.py`。

远端结果：`exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b/`。

表格：`exp_code_ljj/results/main_experiment_table_deepseek14b.md`。报告器已改为优先读取新合同目录；旧评估不进入新表。基线与Our的策略输入不对称必须在实验说明披露，不能将性能差全部归因于模型方法。

## 衔接修复

2026-10-08核实旧总调度进程处于暂停状态（T），因此新DeepSeek队列在四种已具备权重的方法完成生成与双Judge评估后，负责衔接对应Our SFT+RL及LatentRATT SFT+RL，再完成各自192条生成和评估。Our仍调用原训练入口和原配置；若已有同骨干训练监督进程，则等待该进程，不重复启动。Llama现有训练继续运行。

## 2026-10-08 GPU新增授权及加速安排

用户明确授权使用已空闲的GPU0和GPU1。生成GPU0；DeepSeek Our SFT+RL训练GPU1；Selene在GPU1按稳定20条前缀分批评估，Qwen3在GPU0完成生成后评估。Selene的进程显存上限22GiB，训练进程上限54GiB；评估准入至少26GiB空闲。正式表只导入完整192条结果，保留原始Judge记录和GPU1执行来源。

Llama Our既有GPU4训练继续；Llama LatentRATT在GPU4从checkpoint-3600续训，不再等待五种方法全部完成。其后接RL和固定192生成评估；模拟被劝说者不接收目标，以匹配Llama现有正式合同。Qwen设置和已完成分数保持。

生成、训练及判断的内容设置未因换卡改变；训练数据、epoch和loss权重保持。GPU2/3不使用，不停止其他项目任务。DeepSeek生成任务PID4006050在更换CPU监督器时原样保留，没有重复生成。
