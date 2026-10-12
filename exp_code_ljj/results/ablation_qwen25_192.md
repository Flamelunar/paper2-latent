# Qwen2.5-7B 两项消融（固定192条）

历史主表Ours的精确generate_direct.py接口；模拟器按本次要求仅读背景与公共历史。完整Ours重新生成，原主表不覆盖。旧主表Success/Win@1提示词原样保留；Turn缺失判定且无已确认成功时计5，并单独披露缺失数量。

| Variant | Judge | N | Success (%) | Win@1 (%) | Turn (all N; failure/missing=5) | Success unresolved rounds |
|---|---|---:|---:|---:|---:|---:|
| full_ours | selene8b | 192/192 | 69.27 | 61.20 | 4.156 | 1 |
| full_ours | qwen3_14b | 192/192 | 52.08 | 67.06 | 4.443 | 0 |
| wo_g | selene8b | 192/192 | 71.35 | 61.98 | 4.104 | 1 |
| wo_g | qwen3_14b | 192/192 | 47.92 | 66.15 | 4.495 | 0 |
| wo_r_kl_only | selene8b | 192/192 | 71.35 | 61.98 | 4.104 | 1 |
| wo_r_kl_only | qwen3_14b | 192/192 | 47.92 | 66.15 | 4.495 | 0 |

w/o R（KL-only）经1800个context、226次实际更新验证，KL、梯度和参数变化均为0，复用w/o G记录；两行不是独立测量。

Success以全部已评场景为分母；Turn将无已确认成功的场景计为5。Win@1仅在两种次序均解析成功时计入有效分母。生成缺失回合与Judge解析失败分开报告：

| Variant | Judge | Success有效N / 全部N | Win@1有效N | 生成失败场景 | 缺失回合 | 无确认成功且判定缺失场景 |
|---|---|---:|---:|---:|---:|---:|
| full_ours | selene8b | 192/192 | 192 | 0 | 0 | 0 |
| full_ours | qwen3_14b | 192/192 | 192 | 0 | 0 | 0 |
| wo_g | selene8b | 191/192 | 192 | 1 | 2 | 1 |
| wo_g | qwen3_14b | 191/192 | 192 | 1 | 2 | 1 |
| wo_r_kl_only | selene8b | 191/192 | 192 | 1 | 2 | 1 |
| wo_r_kl_only | qwen3_14b | 191/192 | 192 | 1 | 2 | 1 |
