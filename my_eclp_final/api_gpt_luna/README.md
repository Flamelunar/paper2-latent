# GPT-6-luna 固定192条API评估

请求端点为 `https://api.chatanywhere.tech/v1/chat/completions`，请求模型为 `gpt-6-luna`。接口验证返回模型 `gpt-6-luna-2026-09-22`；这是服务商返回的标识，非独立的模型身份验证。

共16组：Qwen2.5六种方法，Llama3.1六种方法，DeepSeek四种已完成基线。每组192条，所有组共3072场景、21504个原始请求。当前不做Win-Full。所有组共用远程最新 `llama_public_repair_v3/protocol.py` 成功判定及 `win_rules.json` Win@1规则。准确源文件、哈希、固定索引、golden哈希见 `run_contract.json`。

每场景分别评估五轮当前被劝说者的目标接受状态；任一轮明确接受算成功；平均轮数取首次成功轮，完整失败计5。若无已确认成功且存在未解析轮，排除该场景的Success及平均轮数分母。Win@1比较公开首句和同场景golden首句，AB/BA换序各请求一次，胜1、平0.5、负0，平均为场景credit，两次都有效才进入分母。

原样复用共享system prompt及业务提示词，API使用JSON object输出模式；不加入方法名称、私有分析、心理标签、候选策略或Judge旧分数。严格校验JSON字段、布尔类型、progress与success一致性及引用确实属于当前回复。格式重试追加仅约束引用格式的提醒，保持原判断规则；仅对格式/网络失败重试，不按分数挑选判断。每个请求的prompt哈希、服务商返回模型名、原始响应、token用量及重试保存到 `requests.sqlite3`。

输入为既有完整生成，不重新生成。Qwen当前两阶段Our尚不完整，因此使用完整旧公开回复版本。Llama对应最新统一v3表的生成来源。DeepSeek仅四个完成的基线。生成方已有的目标可见性、历史模拟器和方法推理预算差异不会因换Judge而消失；只能称统一评估协议，不能称完全统一生成条件。

48路线程并发；缓存持久化后可断点继续。API未指定temperature，使用服务商默认；这与本地Judge的贪心解码不同，不能声称两者推理配置完全相同。`max_completion_tokens=1024`含可能的推理token。API密钥读取本地私有文件 `/home/hezj/.config/paper2-luna/api_key`，权限600，不随代码、结果或日志上传。

运行 `python3 evaluate_api.py`。`summary.json` 和 `main_experiment_table_gpt_luna.md` 每30秒刷新，完整192条前为临时结果。同步程序 `sync_results.py` 每分钟上传结果至远程 `exp_code_ljj/results/gpt_luna_192_20261009/`。
