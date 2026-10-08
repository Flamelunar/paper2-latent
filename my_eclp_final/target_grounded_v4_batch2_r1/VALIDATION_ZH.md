# 加速版验证记录（2026-10-06 17:21）

当前运行版本：target_grounded_v4_batch2_r1。远程位置：my_eclp_final/target_grounded_v4_batch2_r1/。结果位置：exp_code_ljj/artifacts/target_grounded_v4_batch2_r1/。

已通过静态编译、数据与完整checkpoint合同校验；30步独立随机流验证；不同提示词长度左padding验证；普通input_ids和连续inputs_embeds两个接口的实际HF批量generate验证。

初版实际模型检查发现开头角色标签会被误当成后续对话而提前停止；r1已修正，并增加了开头标签、后续标签、私有thinking块的检查。失败版本保留为独立目录，未混入正式新生成结果。

真实模型检查：MA2P第二个完整五轮场景落盘（92.5秒；这是单条补齐，不是双条吞吐基准）；Our两个场景同时完成五轮，整批约88.2秒，折合约44.1秒/场景，未OOM。两个场景均使用完整ECLP checkpoint，并各自保存公开历史、心理分析和策略；分析不进入模拟对象与Judge。该速度是首批实测，不能据此证明所有场景加速倍数或分数不变。

现已自动进入20条补齐阶段。当前已保存Zero-shot20、MA2P2、Our2；20条完成后进行新Judge判分，再扩展192。原先两项GPU0训练继续保持暂停，其他卡任务未停止。

采样实现：每行使用自己的torch.Generator，先应用HF Temperature/TopK/TopP warpers（0.95/50/0.7）进行随机抽样，再向生成引擎返回单一选中token。因此底层generate的do_sample=False和采样flags提示不表示取消温度采样；随机抽样由RowSampler完成。BF16、512 tokens上限、五轮、模型及Judge提示词保持原设置。不同batch的浮点舍入可能使实际文本不同。
