# Llama：Our / PersuGPT 实时进度

固定192条；不足192时为临时分数。训练与权重不变。Success为五轮任一轮明确接受目标；Win含平局credit，换序两次取平均。

| Judge | Method | 已生成 | 已评估 | Success (%) | Win@1 (%) | Win-Full (%) | 平均轮数 | Win@1换序冲突 (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| selene8b | zero_shot_repaired | 192/192 | 0/192 | - | - | - | - | - |
| selene8b | dialogue_sft | 192/192 | 0/192 | - | - | - | - | - |
| selene8b | persugpt | 192/192 | 0/192 | - | - | - | - | - |
| selene8b | ma2p_repaired | 192/192 | 0/192 | - | - | - | - | - |
| selene8b | latentratt | 0/192 | 0/192 | - | - | - | - | - |
| selene8b | our_direct | 0/192 | 0/192 | - | - | - | - | - |
| qwen3_14b | zero_shot_repaired | 192/192 | 0/192 | - | - | - | - | - |
| qwen3_14b | dialogue_sft | 192/192 | 0/192 | - | - | - | - | - |
| qwen3_14b | persugpt | 192/192 | 0/192 | - | - | - | - | - |
| qwen3_14b | ma2p_repaired | 192/192 | 0/192 | - | - | - | - | - |
| qwen3_14b | latentratt | 0/192 | 0/192 | - | - | - | - | - |
| qwen3_14b | our_direct | 0/192 | 0/192 | - | - | - | - | - |
