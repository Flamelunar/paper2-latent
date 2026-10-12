# Llama 公开话语修复与实时评估计划

训练代码、SFT/RL权重及数据保持固定。Our使用原完整checkpoint，公开输出仅包含实际劝说话语，不输出心理/策略分析。PersuGPT保留原ISR-SFT+SimDPO权重。两者使用相同背景/目标/历史提示框架和公开输出检查；第一个格式合格回复进入对话，不依据Judge评分筛选。

固定192条，以相同稳定顺序按20条为一批推进（20、40、…、180、192）。Our actor在GPU1，PersuGPT actor在GPU4，二者共享GPU4的同一模拟被劝说者，模拟方仍不接收精确目标。原独立场景seed与解码分布保持固定。

每批生成后释放两位actor及共享模拟器，在GPU1运行Selene，在GPU4运行Qwen3；每个Judge分别评估Our和PersuGPT已生成的同一场景前缀。复用之前批次的原始判断，不能用分数挑选下一次输出。该安排避免额外常驻Judge导致显存不足，并让首批分数尽早可见。

修复Success对“考虑、研究、稍后决定”等未承诺行为的边界，并保留目标本来只要求尝试或探索的合法成功。输出仍是evidence、progress、布尔success。Win保持两次换序平均credit，新增明确的位置无关提醒，实时披露换序冲突率。两者的换序冲突包含胜/平变化，不等同全部完全翻转。

在Our/PersuGPT双Judge固定192评估完成后，zero-shot、Dialogue-SFT、MA2P以及届时可用的LatentRATT复用原公开对话，用同一新规则重新评估。LatentRATT没有完整192对话时继续标记pending。只在新结果完整且来源哈希校验通过后更新Llama正式表；旧表备份保留。新Judge规则下的Llama表不能默认与旧Qwen2.5表完全同口径。

实时文件：exp_code_ljj/results/llama_our_persugpt_live_v3.md。
完整合同、进度、分数、请求缓存：exp_code_ljj/artifacts/llama_public_repair_v3_20261009/meta_llama31/。
分数不足192时为partial；同一Judge下两者已评估数量不同时不作横向比较。

保护GPU0及GPU4上已有任务。调度器只终止自己本次创建的子进程。失败状态保存错误和日志，不接受旁白充当有效话语，不把解析失败当作负例写入最终表。
