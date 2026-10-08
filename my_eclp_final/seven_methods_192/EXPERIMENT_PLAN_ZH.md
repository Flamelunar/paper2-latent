# 七方法192场景重新生成与评估

固定192个无前置对话场景，原始索引不变，五轮任一轮明确接受精确目标计成功，失败轮数计5。旧分数全部归档，不复用旧Judge结论。

## 方法与来源

Zero-shot、MA2P、Our分析版、Our直接版重新生成。Dialogue-SFT、PersuGPT（ISR-SFT+SimDPO）和LatentRATT按用户最新决定复用历史1000条中的对应192条，仅重新评分，不重训。历史记录原样保存，历史输出边界审计另行披露。

Our使用offline_rl/checkpoint-001800完整模块，两版唯一预设差异是输出接口。分析版psychology/strategy只存档，utterance进入模拟对象、公开历史、隐式状态更新和Judge。直接版提示私下考虑心理和策略，只接受实际劝说话语，显式分析视为格式失败并有限重试。

Zero-shot仅背景和公开历史。MA2P保留Perception→World Model→Persuader、K=0。LatentRATT本轮复用历史生成，不启用新生成器；历史完整192条及原来源SHA已核验，4处输出边界疑点保留并披露。原生生成器仅作为备用入口保留。

新生成共同使用Qwen3.5-9B-persuadee/merged，BF16，无量化；temperature=.95、top_p=.7、top_k=50、max_new_tokens=512、seed=42。支持的方法batch2，LatentRATT单样本。不得输入golden、真实心理标签、真实策略、未来回复。不得依据胜率择优保留样本。

## 评分

Selene-1-Mini-Llama-3.1-8B和Qwen3-14B共用生成和golden，temperature=0、BF16、无量化；同一证据型精确目标Success和独立评价后比较的Win提示词。Success严格bool，证据来自当前回复；考虑、了解、接受前置行动不等于接受更广目标。Win@1比较首句，Win-Full比较全部实际生成对话和golden全部已有对话，不补造缺失轮次。AB/BA换序平均计分，win1/tie.5/loss0，报告冲突和平局；两次换序不是独立样本。

表格只在192条完整有效评分后填入最终数值，异常显示待完成，不虚构false或零分。记录格式重试、证据校验、模型和源码哈希，生成和Judge请求均支持断点。

## 排程与监督

GPU1与GPU4各最多两组本实验任务，按真实空闲显存预检调度：Our分析和Zero-shot在GPU1并行，Our直接和Qwen3评分在GPU4并行；MA2P在1或4卡生成槽位释放后立即进入。GPU0延续原Selene评分。其他任务不停止，原暂停训练保持暂停。supervise_parallel.py接管本实验已有子进程，避免重复启动；每15秒刷新全部GPU工作进程的真实状态、进度与表格。每任务最多三次进程级重试，失败后其他就绪任务继续。冻结的生成和Judge合同及提示词哈希不改写；Latent历史复用决定通过historical_latentratt_provenance.json补充记录。

首个batch检查角色、实际公开输出及五轮完整性后继续；不额外做20条择优实验。首批和后续输出均属于同一新正式run。

## 文件

代码：my_eclp_final/seven_methods_192/
结果：exp_code_ljj/artifacts/seven_methods_192_20261006/
表格：exp_code_ljj/results/main_experiment_table.md
旧表：新结果目录archive/main_experiment_table_before.md
状态：新结果目录live_status.json
