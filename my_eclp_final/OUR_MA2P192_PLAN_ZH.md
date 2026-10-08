# Our + MA2P：固定192场景实验

按用户要求保留原有MA2P (prompt adaptation, K=0) 的Perception → World Model → Persuader编排，将三个角色的模型后端统一替换成我们训练后的最终checkpoint。三个角色不是不同来源的模型；不使用额外的原始Qwen模型做规划。

冻结checkpoint：`eclp_runs_gpu4/offline_rl/checkpoint-001800`，checkpoint_id为 `377f19fdc349edeb91670fd85f546d535f89da037b9ef5450b371c22af882595`。不重训练、不根据192条测试集挑选checkpoint。

默认接口为完整ECLP，包括训练的语言模型LoRA、隐式策略与状态模块。每个MA2P角色调用相同checkpoint，使用该角色原来的system/user提示词和同一公开对话状态；Perception和World Model的生成也经过ECLP隐式策略，JSON语法仅受有限token约束。角色内部分析不被写入公开对话状态，只有实际双方发言才更新状态。可选lora_only接口仅用于用户明确要求的原始HF语言模型替换，不能与完整ECLP结果混排而不说明。

数据由 `exp_code_ljj/data/test_selection_192.json` 固定，共192个target_turn=1且原始history=[]的场景。保留原始index、scenario_id和golden对应关系。第一轮没有原始前置对话；后续用当前生成的公开历史。Perception及World Model只从公开背景、目标、角色名字、对话及本次生成的计划推断，不接触策略标签、reference、未来对话或Judge输出。

五轮固定Persuadee：`/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged`。MA2P角色沿用temperature=0.95、top_p=0.7、top_k=50、max_new_tokens=512；BF16、无量化、无在线更新。正常情况每场景15次Our模型调用和5次Persuadee调用；实际重试单独记录。

单场景验证发现Persuadee有时在第一句之后继续生成多轮`user/assistant`文本。正式实验仅将首句写入公开历史，同时保存完整原始输出及截断标记；否则伪造的后续对话会泄漏未来接受意愿并抬高成功率。对于Our Persuader清洗后的公开内容明显以第三人称写计划而非对话的输出，包括以Persuadee名字作第三人称叙述开头，使用同一MA2P提示词最多重采样两次；拒绝的原始输出单独记录。若三次都仍是计划叙述，保留最后一次完整ECLP输出并标记`retry_exhausted`，交给同一Judge评估，不合成替代话术。分析角色格式重试耗尽时同样保留模型原文，沿用MA2P原始解析和空值策略。此修正只作用于Our+MA2P新实验，因此旧MA2P行若保留未修复的生成路径，不可作为严格同条件的增益估计，需另行同协议重跑。

显卡按真实空闲容量分配，模型启动前检查BF16权重总需求及余量；不停止任何其他任务。当前计划在GPU4上同时承载三个角色共用的Our模型和固定Persuadee；Selene评估随后在同一GPU4上运行。

只运行Selene unified v2，沿用当前192条表格的Success、Win@1、Win-Full及严格JSON grammar。五轮内任一轮接受即成功，首次成功轮数失败计5；Win两次独立匿名换序，胜1、平0.5、负0后取平均。原始golden用于评估，不反馈生成。已知Selene语义误判风险仍然存在，本行使用同一协议保证可比性，不对新方法单独放宽或收紧。

生成目录：`eclp_runs_gpu4/our_ma2p192/generation/`。
评分目录：`eclp_runs_gpu4/evaluation_v2/selene8b/our_ma2p/`。
新入口：`my_eclp_final/generate_our_ma2p192.py`、`evaluate_our_ma2p192.py`。
主表Selene组新增 `Our + MA2P` 行；192条完成且逐条校验通过后填真实结果。

先验证一个场景的五轮及角色调用，再从同一固定合同接续剩余191条。所有运行合同、模型checkpoint身份、代码/数据/提示词哈希、生成轨迹、内部角色请求与Judge原始输出独立存档。若改变实现，需要新合同和新结果目录，不能混入原192条版本。
