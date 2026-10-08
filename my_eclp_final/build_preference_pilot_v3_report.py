"""Recompute paired pilot scores and write a standalone comparison report."""
import json
import statistics
import sys
from pathlib import Path

p = Path(sys.argv[1])
c = json.loads((p / 'run.json').read_text())
s = json.loads((p / 'summary.json').read_text())
assert s['complete'] and len(s['results']) == len(c['methods'])
labels = {'baseline1_zero_shot': 'Zero-shot', 'baseline2_dialogue_sft': 'Dialogue-SFT',
          'baseline3_isr_simdpo': 'ISR-SFT + SimDPO', 'baseline4_ma2p': 'MA2P',
          'baseline5_latentratt': 'LatentRATT', 'our_eclp_offline_lpo': 'Our', 'our_ma2p': 'Our + MA2P'}
lines = ['# Selene 新旧偏好提示词小样本对比', '',
         '固定 192 条无前置对话测试集中的共同 20 条；按 SHA256(matched-v2-pilot-42:scenario_id) 排序取前 20 条，选样不依赖模型输出。复用已生成对话，训练和生成均未重跑。', '',
         'Judge：Selene-1-Mini-Llama-3.1-8B，BF16；旧版 unified v2 与新版独立判断提示词使用相同模型、系统提示词、temperature=0、严格 JSON 解码和计分方式。新版要求先分别判断再比较，未采用“默认平局、极少判胜”的规则。输出仅 winner，无法从输出验证内部判断步骤是否被执行。', '',
         '每方法 Win@1 和 Win-Full 各做 20×2 个匿名换序请求。胜=1、平=0.5、负=0，先在场景内平均，再在 20 条场景间平均。成功指标没有改变：五轮任一轮接受精确目标记成功，失败轮数计 5。', '',
         '| 方法 | N | 成功率 (%) | 旧 Win@1 (%) | 新 Win@1 (%) | 旧 Win-Full (%) | 新 Win-Full (%) |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for result in s['results']:
    method = result['method']
    data = json.loads((p / 'methods' / (method + '.json')).read_text())
    assert data['contract_hash'] == s['contract_hash']
    rows = data['records']
    assert len(rows) == 20 and [r['index'] for r in rows] == c['expected_indexes']
    assert all(len(r['round_success']) == 5 and all(type(x) is bool for x in r['round_success']) for r in rows)
    assert result['success_percent'] == 100 * sum(any(r['round_success']) for r in rows) / len(rows)
    for version in ['old', 'new']:
        for metric in ['win1', 'win_full']:
            values = [r[version][metric] for r in rows]
            for v in values:
                assert all(v[o] in ['A', 'B', 'Tie'] for o in ['ab', 'ba'])
                credits = [0.5 if v[o] == 'Tie' else float(v[o] == ('A' if o == 'ab' else 'B')) for o in ['ab', 'ba']]
                assert v['credit'] == statistics.fmean(credits)
            assert result[version][metric]['score_percent'] == 100 * statistics.fmean(v['credit'] for v in values)
    lines.append(f"| {labels[method]} | 20 | {result['success_percent']:.2f} | "
                 f"{result['old']['win1']['score_percent']:.2f} | {result['new']['win1']['score_percent']:.2f} | "
                 f"{result['old']['win_full']['score_percent']:.2f} | {result['new']['win_full']['score_percent']:.2f} |")
lines += ['', '## 判分稳定性', '',
          '冲突率：换序后对同一候选给出的 credit 不一致的场景比例；显式平局率：40 次请求中直接输出 Tie 的比例。换序冲突与显式平局分开统计。', '',
          '| 方法 | Win@1 冲突率旧→新 (%) | Win-Full 冲突率旧→新 (%) | Win@1 平局率旧→新 (%) | Win-Full 平局率旧→新 (%) |',
          '| --- | ---: | ---: | ---: | ---: |']
for r in s['results']:
    cells = []
    for field in ['order_conflict_percent', 'explicit_tie_percent']:
        for metric in ['win1', 'win_full']:
            cells.append(f"{r['old'][metric][field]:.2f} → {r['new'][metric][field]:.2f}")
    lines.append('| ' + labels[r['method']] + ' | ' + ' | '.join(cells) + ' |')
lines += ['', '## 范围与来源', '',
          '所有 Win 均与对应 golden 对比，并非方法之间直接对比。Win@1 比较第一条完整劝说回复；Win-Full 比较保存的五轮对话与原始 golden 全部可用对话，长度可能不同。公开背景和精确目标提供给 Judge，策略和隐藏心理状态未额外提供。', '',
          '旧分数来自同一 20 条保存的 v2 判断；ISR-SFT + SimDPO 无该旧记录，使用原 v2 提示词补评了这 20 条的旧 Win 和五轮 Success。该方法保存的 Persuadee 路径为 /data1/liujianjian/Persuadee/Qwen3.5-9B-persuadee/merged，与其余来源的 /app 路径不同，未在此小测验证权重等价。因此本报告用于检查提示词在同一保存输出上的变化，不能证明各方法生成条件完全相同。', '',
          'Our + MA2P 的模拟对象续写只截取第一条回复；旧 MA2P 未实行相同截取，两者不能视为严格同条件的模型替换实验。', '',
          '20 条只用于提示词诊断，分数不能替换 192 条主表或支持稳定的性能排名。主表未修改。', '',
          '原始索引：' + ', '.join(map(str, c['expected_indexes'])), '',
          '逐条判断保存在 methods/；原始 Judge 输出和 token 数保存在 request_batches/；来源与哈希见 run.json。']
(p / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'verified': True, 'methods': len(s['results']), 'n_each': 20, 'report': str(p / 'comparison.md')}, ensure_ascii=False))
