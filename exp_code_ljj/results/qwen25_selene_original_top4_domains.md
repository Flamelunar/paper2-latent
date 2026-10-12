# Qwen2.5 原始主表：Selene 高频领域对比

直接读取 main_experiment_table.md 对应的 seven_methods_192_20261006 历史生成和评估记录，不重新生成或评分。
领域频次按同一固定192条场景统计，每个场景对同一domain最多计一次；多领域场景可同时计入多个领域。
Success和Turn完全沿用原主表聚合：任一轮成功计成功，失败轮数为5；含未解析goal判定的场景从这两个指标的有效分母中排除。Win使用A/B和B/A平均信用，胜/平/负为1/0.5/0。

| Domain | N | Method | Success (%) | Win@1 (%) | Win-Full (%) | Mean rounds | Success有效N | 格式未解析回合 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Lifestyle | 36 | Zero-shot | 47.22 | 47.92 | 6.25 | 4.472 | 36 | 0 |
| Lifestyle | 36 | Dialogue-SFT | 69.44 | 45.14 | 18.75 | 4.611 | 36 | 0 |
| Lifestyle | 36 | Ours | 77.78 | 65.97 | 34.72 | 4.167 | 36 | 0 |
| Education | 29 | Zero-shot | 62.07 | 50.00 | 7.76 | 4.517 | 29 | 0 |
| Education | 29 | Dialogue-SFT | 55.17 | 43.10 | 15.52 | 4.966 | 29 | 0 |
| Education | 29 | Ours | 65.52 | 57.76 | 23.28 | 4.379 | 29 | 0 |
| Career | 19 | Zero-shot | 52.63 | 42.11 | 7.89 | 4.789 | 19 | 0 |
| Career | 19 | Dialogue-SFT | 57.89 | 43.42 | 18.42 | 4.789 | 19 | 0 |
| Career | 19 | Ours | 63.16 | 59.21 | 14.47 | 4.053 | 19 | 0 |
| Business | 18 | Zero-shot | 61.11 | 50.00 | 11.11 | 4.444 | 18 | 0 |
| Business | 18 | Dialogue-SFT | 66.67 | 47.22 | 13.89 | 4.833 | 18 | 0 |
| Business | 18 | Ours | 72.22 | 55.56 | 22.22 | 4.111 | 18 | 0 |

原始整体分数已复算并与主表对应summary逐项一致：
| Method | Success (%) | Win@1 (%) | Win-Full (%) | Mean rounds |
|---|---:|---:|---:|---:|
| Zero-shot | 54.69 | 47.14 | 9.77 | 4.516 |
| Dialogue-SFT | 57.59 | 47.92 | 17.84 | 4.728 |
| Ours | 68.23 | 61.20 | 27.08 | 4.271 |

来源：`/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/artifacts/seven_methods_192_20261006`。
逐场景ID、领域、原始判定、来源文件哈希及完整领域频次见 statistics.json；统计脚本为 summarize.py。
