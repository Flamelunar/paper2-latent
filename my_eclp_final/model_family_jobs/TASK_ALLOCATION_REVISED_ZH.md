# 修订任务分配与运行状态（2026-10-07）

0卡：本项目不使用，保留其他项目任务。1、4卡：不停止其他项目任务，按照实测空闲显存准入。
任务2此前按用户指令停止，文件保留。Llama SFT试跑曾因与其他训练共享显存而OOM，当前监督器只在预留显存满足门槛后恢复；Llama要求GPU1至少50GiB空闲，DeepSeek要求至少62GiB空闲。一个模型完成SFT+RL后才开始另一个模型，避免GPU1并发抢占。

任务1A：无需重新生成的baseline，核对实际最新生成代码、提示词、模型、固定192索引、空初始history及说话角色边界，通过检查才复用输出。优先检查SFT、PersuGPT、MA2P、LatentRATT；不能仅凭方法名认定无需重生成。GPU4双Judge队列；不足显存串行。
任务1B：接收任务3的新zero-shot和Our输出，再使用同一最新Judge协议评估。旧zero-shot/Our分数不得作为正式成绩或回填表格。当前已运行旧输出评估仅作为诊断，不新启动诊断任务；本次未停止已有任务1队列。
任务3（当前优先）：两套zero-shot已在生成。Llama在GPU4，DeepSeek在GPU1；最新可查进度分别70/192、48/192。每个模型生成完成后立即在原卡启动Selene与Qwen3评估，两个模型的评估对可在GPU1、GPU4并行。固定192条，5轮，初始history空，后续使用本次真实生成history。zero-shot严格仅背景和可观察历史，排除目标、策略、心理标签、golden。Our使用对应骨干的完整RL checkpoint，私下考虑策略、公开只输出劝说。保存原始输出和有限重试轨迹。
Our依赖：Llama/DeepSeek若尚无完成的各自Our checkpoint，则任务3先运行可用zero-shot；对应Our生成必须等待任务2，不能使用Qwen checkpoint替代。训练是否需要更新由修复是否涉及推理或训练决定；仅提示词/生成边界修复可复用同骨干的合格checkpoint。
任务2：排在两套zero-shot生成及双Judge评估之后自动恢复。每个骨干各自SFT 1epoch（59,868会话、7,484步）及全隐式RL（1,800训练场景、200开发场景），训练数据和奖励方案不变。上一轮3个未落checkpoint的试跑step保存在原目录；正式恢复使用新的`sft_full_retry_20261007`目录和每50步checkpoint。
新正式目录：`exp_code_ljj/artifacts/model_family_repaired192_20261007/`。两张主表已备份原版本并清除旧分数，显示新192任务的生成进度及待评估状态；各新Judge完成后自动回填。正式指标：五轮任一轮成功、Win@1、Win-Full、平均成功轮数（失败=5）、格式未解析。旧诊断分数不进入新表。
