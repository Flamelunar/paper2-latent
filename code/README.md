# 最新代码导航

分类入口是符号链接，运行时仍使用原始代码路径，避免改变 __file__ 和相对路径行为。不同模型的输入与评估版本按各自实验合同保留，不用时间戳猜测统一版本。

| 用途 | 推荐入口 | 说明 |
|---|---|---|
| ECLP核心 | my_eclp_final/eclp_elpo.py | 模型结构与联合监督训练 |
| ECLP离线优化 | my_eclp_final/offline_latent_rl.py | latent-policy optimization |
| Qwen原主表192生成/评估 | code/latest/qwen25_main192 | generate_direct.py / evaluate.py / protocol.py / win_rules.json |
| 多骨干生成与管理 | code/latest/model_family | 按模型、实验合同选相应脚本 |
| 后续旧Success+新Win评估 | code/latest/hybrid_judge | 与原主表协议分开，不能互换历史分数 |
| API Judge | code/latest/api_gpt_luna | evaluate_api.py |
| 消融 | code/latest/ablations | 独立输出和checkpoint |
| 普通SFT+文本GRPO | code/latest/standard_grpo | 20261011对照代码，已按用户要求停止 |
| 基线1–4原实现 | exp_code_ljj/baseline1_zero_shot.py 等 | 最新修复生成入口还需查看模型家族/192合同 |
| LatentRATT基线 | code/latest/latentratt | run_latent_ratt.py / latent_model.py |
| 公共推理工具 | exp_code_ljj/repro | 模拟器、模型加载等 |
| 固定测试集 | exp_code_ljj/data | test_selection_192.json；evaluation_points_from_start192.jsonl |
| 主表 | results_catalog | 三模型主表及消融表 |

历史代码存入archive/directory_cleanup_20261012；原路径有兼容链接。使用本导航里的正式入口，不使用.before、pending或Gemma旧工具作为最新实现。

停止状态保持：目录整理没有执行训练、生成、Judge评分或任何监督脚本。
