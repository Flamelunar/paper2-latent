# Qwen2.5-7B — fixed 192 scenarios, new evaluation

All prior scores are invalid for this run and archived. The fixed subset has target_turn = 1 and empty initial history. Maximum five rounds; success means exact goal acceptance in any round; failures count as five rounds.

Dialogue-SFT, PersuGPT and LatentRATT reuse the corresponding 192 historical dialogues verbatim; Zero-shot, MA2P and the two Our variants generate new dialogues. Both Our variants expose only spoken persuasion to the simulator and judges. Historical simulator paths identify Qwen3.5-9B-persuadee; exact old/new weight equivalence is not established by the path name.

Historical output boundary audit: 6 flagged turns; preserved and disclosed, not silently repaired. See historical_boundary_audit.json. LatentRATT history is audited separately in historical_latentratt_provenance.json; flagged turn count: 4.

Win@1 compares first replies; Win-Full compares all available conversations against golden. Each uses both candidate orders (win=1, tie=0.5, loss=0). These scores are preference credits including ties. Swapped calls are bias controls, not independent samples.

## Selene-1-Mini-Llama-3.1-8B

| Method | N | Goal acceptance within 5 rounds (%) | Win@1 (%) | Win-Full (%) | Mean rounds to success (failures = 5) | Success format unresolved (rounds) |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 192/192 | 33.33 (192/192) | 47.14 (192/192) | 9.77 (192/192) | 4.52 (192/192) | 0 |
| Dialogue-SFT | 192/192 | 21.47 (191/192) | 47.92 (192/192) | 17.84 (192/192) | 4.73 (191/192) | 1 |
| PersuGPT (ISR-SFT + SimDPO) | 192/192 | 43.75 (192/192) | 72.79 (192/192) | 36.07 (192/192) | 4.43 (192/192) | 0 |
| MA2P | 0/192 | Pending | Pending | Pending | Pending | Pending |
| LatentRATT | 192/192 | 34.38 (192/192) | 53.52 (192/192) | 17.06 (192/192) | 4.52 (192/192) | 0 |
| Our: private analysis + persuasion | 160/192 | Pending | Pending | Pending | Pending | 0 |
| Our: persuasion only | 192/192 | 49.48 (192/192) | 61.20 (192/192) | 27.08 (192/192) | 4.27 (192/192) | 0 |

Order-conflict rates and tie rates are retained in each method summary. Partial runs do not populate final scores.

## Qwen3-14B

| Method | N | Goal acceptance within 5 rounds (%) | Win@1 (%) | Win-Full (%) | Mean rounds to success (failures = 5) | Success format unresolved (rounds) |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 120/192 | Pending | Pending | Pending | Pending | 0 |
| Dialogue-SFT | 192/192 | 8.85 (192/192) | 43.49 (192/192) | 18.23 (192/192) | 4.90 (192/192) | 0 |
| PersuGPT (ISR-SFT + SimDPO) | 192/192 | 27.08 (192/192) | 86.33 (192/192) | 39.58 (192/192) | 4.66 (192/192) | 0 |
| MA2P | 0/192 | Pending | Pending | Pending | Pending | Pending |
| LatentRATT | 192/192 | 23.96 (192/192) | 59.77 (192/192) | 21.61 (192/192) | 4.71 (192/192) | 0 |
| Our: private analysis + persuasion | 0/192 | Pending | Pending | Pending | Pending | Pending |
| Our: persuasion only | 0/192 | Pending | Pending | Pending | Pending | Pending |

Order-conflict rates and tie rates are retained in each method summary. Partial runs do not populate final scores.
