# llama31: old Success + latest Win, fixed192

Success提示词精确复用seven_methods_192/protocol.py；Win@1及Win-Full使用当前位置中立规则。五轮任一轮成功；平均轮数失败计5；换序credit平均。仅重新评估既有公开对话，生成输入条件与方法接口不变。partial为临时结果。

| Judge | Method | N | Success (%) | Win@1 (%) | Win-Full (%) | Mean rounds (failure=5) | Success unresolved |
|---|---|---:|---:|---:|---:|---:|---:|
| selene8b | latentratt | 192/192 | 59.90 | 57.55 | 11.20 | 4.427 | 1 |
| selene8b | our_direct | 0/192 | - | - | - | - | - |
| qwen3_14b | latentratt | 192/192 | 34.90 | 56.12 | 12.50 | 4.714 | 0 |
| qwen3_14b | our_direct | 0/192 | - | - | - | - | - |
