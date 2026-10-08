# ECLP 论文交接包

本目录是2026-10-07整理的写作与实验设计交接，覆盖当前完整ECLP-Offline-LPO、公开输出仅劝说语句版本。数值实验尚未全部完成；TBD与计划实验不能写成论文结论。

建议阅读顺序：

1. `PAPER_AGENT_TASK_BRIEF_ZH.md`：可直接复制给另一个写作agent的任务与硬性事实。
2. `CCFA_PAPER_HANDOFF_ZH.md`：动机、逐章写法、方法公式、数据与baseline、消融及图表指南。
3. `PAPER_DRAFT_EN.md`：接近完整英文正文的草稿，结果部分明确待填。
4. `EXPERIMENT_CHECKLIST.csv`：实验优先级、控制变量、记录格式与完成标准。
5. `LITERATURE_AND_EVIDENCE.md`、`references.bib`：已核验引用、证据边界及仍需补查的文献。
6. `METHOD_AND_RESULTS_PROVENANCE.json`：源码与数据快照、技能版本和交接时间的机器记录。

本地目录：`/home/hezj/liujianjian/paper_writing/`。
远程镜像：`/app/hzj/liujianjian/ljj-2-paper/my_eclp_final/paper_writing/`。

源码快照和论文阅读材料保留在本地 `ccfa-workfiles/paper-handoff/eclp/evidence/`，用于核对，不把用户另一篇内部稿当作公开参考文献。安装的17个CCFA技能位于 `/home/hezj/.codex/skills/`，源commit为 `5969e6b20a3bbcef9118fa00796d1417d48fcbf3`。

当前运行的模型、训练与评估结果应从远程 `exp_code_ljj/artifacts/` 和 `exp_code_ljj/results/` 获取完整合同与逐条记录。本目录不是实时评分表。新增分析和消融在此只做设计，GPU4现有任务保持运行。
