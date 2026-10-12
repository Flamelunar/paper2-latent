# deepseek14b: old Success + latest Win, fixed192

Success提示词精确复用seven_methods_192/protocol.py；Win@1及Win-Full使用当前位置中立规则。五轮任一轮成功；平均轮数失败计5；换序credit平均。仅重新评估既有公开对话，生成输入条件与方法接口不变。partial为临时结果。

| Judge | Method | N | Success (%) | Win@1 (%) | Win-Full (%) | Mean rounds (failure=5) | Success unresolved |
|---|---|---:|---:|---:|---:|---:|---:|
| selene8b | latentratt | 0/192 | - | - | - | - | - |
| selene8b | our_direct | 192/192 | 65.45 | 61.20 | 14.06 | 4.236 | 3 |
| qwen3_14b | latentratt | 0/192 | - | - | - | - | - |
| qwen3_14b | our_direct | 180/192 (partial) | 40.56 | 62.64 | 11.25 | 4.628 | 0 |
