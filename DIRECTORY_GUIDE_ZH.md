# ljj-2-paper 目录说明（2026-10-12整理）

```text
ljj-2-paper/
├── code/latest/             最新代码分类入口（指向原实现）
├── my_eclp_final/           ECLP代码、各版本生成和评估
├── exp_code_ljj/
│   ├── data/                训练/开发/固定192测试数据
│   ├── configs/             原始及各骨干实验配置
│   ├── repro/               公共模型加载/模拟器工具
│   ├── results/             正式结果及逐样本评分
│   ├── artifacts/           各run_id实验记录
│   └── logs/organized/      从代码目录整理出的历史日志
├── baseline5_latentRATT/    LatentRATT实现与原论文
├── models_catalog/          按骨干/方法/阶段分类的checkpoint入口
├── results_catalog/         三模型主表和消融表入口
├── eclp_runs_*/             原始训练权重、prepared数据和生成记录
└── archive/directory_cleanup_20261012/  旧代码、历史压缩包及迁移清单
```

先看code/README.md找代码，models_catalog/README.md找权重。exp_code_ljj/results下的主表是结果入口；原始Qwen表与后续hybrid协议都有对应导航，不合并不同协议。

本次归档/整理58项，其中代码目录日志15个。每次迁移都记录在archive/directory_cleanup_20261012/manifest.json。未删除代码、checkpoint、训练数据或结果。兼容符号链接保留旧路径。

回滚：先检查manifest.json，再运行archive/directory_cleanup_20261012/rollback.py；它只撤销本次有记录的移动，不运行实验。

## 日志与缓存清理

2026-10-12按用户要求删除可再生Python缓存、空stdout和弃用Gemma运行日志；新旧prompt代码、配置及评分结果保留。删除清单：archive/directory_cleanup_20261012/deletion_manifest.json。本清理不可由旧rollback.py还原；被删除项目在清单中标识。

## GitHub 发布说明（2026-10-12）

本仓库是远程项目的公开代码/数据/结果快照。远程训练目录的全部中间 checkpoint 和运行日志不发布。微调权重改用 Release 附件，参见 weights/README.md；models_catalog 的原路径仅作为索引，不表示 clone 已带权重。code/latest 使用仓库内相对链接。完整远程回滚/日志归档结构仅适用于原服务器。
