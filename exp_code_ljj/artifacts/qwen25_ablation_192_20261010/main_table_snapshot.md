> 仅LatentRATT和Our重新评估，其他基线保留历史协议结果。新评估正在运行：旧Success提示词＋最新Win@1提示词；旧表暂留作历史记录。新结果见 [qwen25_oldsuccess_newwin_192.md](qwen25_oldsuccess_newwin_192.md)。GPT-luna新结果目录为 `gpt_luna_oldsuccess_newwin_192_20261010/`。

# Qwen2.5-7B — fixed 192 scenarios, new evaluation

All prior scores are invalid for this run and archived. The fixed subset has target_turn = 1 and empty initial history. Maximum five rounds; success means exact goal acceptance in any round; failures count as five rounds.

Dialogue-SFT, PersuGPT and LatentRATT reuse the corresponding 192 historical dialogues verbatim; Zero-shot, MA2P and Our (persuasion only) generate new dialogues. Our exposes only spoken persuasion to the simulator and judges. Historical simulator paths identify Qwen3.5-9B-persuadee; exact old/new weight equivalence is not established by the path name.

Historical output boundary audit: 6 flagged turns; preserved and disclosed, not silently repaired. See historical_boundary_audit.json. LatentRATT history is audited separately in historical_latentratt_provenance.json; flagged turn count: 4.

Win@1 compares first replies; Win-Full compares all available conversations against golden. Each uses both candidate orders (win=1, tie=0.5, loss=0). These scores are preference credits including ties. Swapped calls are bias controls, not independent samples.

## Selene-1-Mini-Llama-3.1-8B

| Method | N | Goal acceptance within 5 rounds (%) | Win@1 (%) | Win-Full (%) | Mean rounds to success (failures = 5) | Success format unresolved (rounds) |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 192/192 | 54.69 (192/192) | 47.14 (192/192) | 9.77 (192/192) | 4.52 (192/192) | 0 |
| Dialogue-SFT | 192/192 | 57.59 (191/192) | 47.92 (192/192) | 17.84 (192/192) | 4.73 (191/192) | 1 |
| PersuGPT (ISR-SFT + SimDPO) | 192/192 | 59.90 (192/192) | 72.79 (192/192) | 36.07 (192/192) | 4.43 (192/192) | 0 |
| MA2P | 192/192 | 83.33 (192/192) | 71.61 (192/192) | 38.15 (192/192) | 3.823 (192/192) | 0 |
| LatentRATT | 192/192 | 54.69 (192/192) | 53.52 (192/192) | 17.06 (192/192) | 4.52 (192/192) | 0 |
| Our: persuasion only | 192/192 | 68.23 (192/192) | 61.20 (192/192) | 27.08 (192/192) | 4.27 (192/192) | 0 |

Order-conflict rates and tie rates are retained in each method summary. Partial runs do not populate final scores.

## Qwen3-14B

| Method | N | Goal acceptance within 5 rounds (%) | Win@1 (%) | Win-Full (%) | Mean rounds to success (failures = 5) | Success format unresolved (rounds) |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 192/192 | 31.25 (192/192) | 38.15 (192/192) | 10.16 (192/192) | 4.80 (192/192) | 0 |
| Dialogue-SFT | 192/192 | 35.94 (192/192) | 43.49 (192/192) | 18.23 (192/192) | 4.90 (192/192) | 0 |
| PersuGPT (ISR-SFT + SimDPO) | 192/192 | 35.94 (192/192) | 86.33 (192/192) | 39.58 (192/192) | 4.66 (192/192) | 0 |
| MA2P | 192/192 | 57.29 (192/192) | 87.89 (192/192) | 45.83 (192/192) | 4.344 (192/192) | 0 |
| LatentRATT | 192/192 | 39.06 (192/192) | 59.77 (192/192) | 21.61 (192/192) | 4.71 (192/192) | 0 |
| Our: persuasion only | 192/192 | 47.92 (192/192) | 67.06 (192/192) | 27.86 (192/192) | 4.63 (192/192) | 0 |

Order-conflict rates and tie rates are retained in each method summary. Partial runs do not populate final scores.
