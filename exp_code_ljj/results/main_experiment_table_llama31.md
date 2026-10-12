> 仅LatentRATT和Our重新评估，其他基线保留历史协议结果。新评估正在运行：旧Success提示词＋最新Win@1提示词；旧表暂留作历史记录。新结果见 [llama31_oldsuccess_newwin_192.md](llama31_oldsuccess_newwin_192.md)。GPT-luna新结果目录为 `gpt_luna_oldsuccess_newwin_192_20261010/`。

## Llama-3.1-8B — two-stage Our + public repair v3, fixed192

Our 与 PersuGPT使用原训练权重全新生成192条，Our先生成私有简短理由、再输出公开劝说话语；PersuGPT公开只输出劝说话语；其余已完成基线复用上一版本的原始对话。各方法共用不接收精确目标的模拟被劝说者与相同解码设置。zero-shot劝说者仍只接收背景与公开历史。

所有方法统一使用llama_public_repair_v3中的成功判定与Win提示词。Success为五轮任一轮明确接受准确目标；Win@1对比首句、Win-Full对比全部可用对话，分别与同场景golden对比。换序credit取平均（胜1、平0.5、负0）；换序冲突率公开披露。格式缺失且无已确认成功的场景排除Success分母。

结果与合同：`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/artifacts/llama_two_stage_pilot_20261009/meta_llama31`。本版本的Judge规则已修订，不能将旧Qwen2.5表直接作为相同评估版本的比较。

| Judge | Method | N | Success (%) | Win@1 (%) | Win-Full (%) | Mean rounds (failure=5) | Success unresolved | Win@1 order conflict (%) | Win-Full order conflict (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| selene8b | zero_shot_repaired | 192/192 | 45.31 | 46.22 | 5.47 | 4.40 | 1 | 67.19 | 12.50 |
| selene8b | dialogue_sft | 192/192 | 41.67 | 49.48 | 4.30 | 4.71 | 0 | 77.08 | 8.33 |
| selene8b | persugpt | 192/192 | 73.44 | 67.06 | 13.54 | 4.16 | 1 | 65.10 | 17.71 |
| selene8b | ma2p_repaired | 192/192 | 50.79 | 55.34 | 9.51 | 4.41 | 1 | 68.75 | 16.15 |
| selene8b | latentratt | 192/192 | 53.12 | 57.55 | 11.20 | 4.48 | 0 | 58.85 | 12.50 |
| selene8b | our_direct | 192/192 | 56.25 | 60.55 | 12.50 | 4.31 | 0 | 56.25 | 14.58 |
| qwen3_14b | zero_shot_repaired | 192/192 | 23.04 | 44.14 | 4.43 | 4.77 | 2 | 30.73 | 3.12 |
| qwen3_14b | dialogue_sft | 192/192 | 21.88 | 48.83 | 4.82 | 4.89 | 0 | 40.10 | 5.21 |
| qwen3_14b | persugpt | 192/192 | 46.35 | 87.76 | 27.73 | 4.49 | 0 | 21.88 | 26.04 |
| qwen3_14b | ma2p_repaired | 192/192 | 27.60 | 58.07 | 14.58 | 4.76 | 0 | 25.52 | 13.02 |
| qwen3_14b | latentratt | 192/192 | 34.38 | 56.12 | 12.50 | 4.71 | 0 | 49.48 | 20.31 |
| qwen3_14b | our_direct | 192/192 | 37.50 | 64.32 | 11.98 | 4.62 | 0 | 29.17 | 17.19 |

<!-- hybrid-latent-only-start -->
# llama31: old Success + latest Win, fixed192

Success提示词精确复用seven_methods_192/protocol.py；Win@1及Win-Full使用当前位置中立规则。五轮任一轮成功；平均轮数失败计5；换序credit平均。仅重新评估既有公开对话，生成输入条件与方法接口不变。partial为临时结果。

| Judge | Method | N | Success (%) | Win@1 (%) | Win-Full (%) | Mean rounds (failure=5) | Success unresolved |
|---|---|---:|---:|---:|---:|---:|---:|
| selene8b | latentratt | 192/192 | 59.90 | 57.55 | 11.20 | 4.427 | 1 |
| selene8b | our_direct | 0/192 | - | - | - | - | - |
| qwen3_14b | latentratt | 192/192 | 34.90 | 56.12 | 12.50 | 4.714 | 0 |
| qwen3_14b | our_direct | 0/192 | - | - | - | - | - |

<!-- hybrid-latent-only-end -->
