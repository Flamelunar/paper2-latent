"""Audit and summarize the completed, checkpoint-matched comparison."""
import collections
import csv
import json
from pathlib import Path
import sys

from supervise import ROOT, RUN, write
from evaluate import summarize

sys.path.insert(0, str(ROOT / 'my_eclp_final'))
from experiment_io import digest, read_json, read_jsonl, object_hash, atomic_json


def domains(value):
    if isinstance(value, str):
        return {value.strip()} if value.strip() else set()
    if isinstance(value, list):
        return set().union(*(domains(v) for v in value)) if value else set()
    if isinstance(value, dict):
        return domains(value.get('domain', []))
    raise TypeError(value)


def main():
    selection = read_json(ROOT / 'exp_code_ljj/data/test_selection_192.json')
    selected = set(selection['original_indexes'])
    assert len(selected) == 192
    points = read_jsonl(ROOT / 'exp_code_ljj/data/evaluation_points.jsonl')
    assert digest(ROOT / 'exp_code_ljj/data/evaluation_points.jsonl') == selection['source_sha256']
    frequency = collections.Counter(d for i in selected for d in domains(points[i]['scenario']['domain']))
    top4 = sorted(frequency, key=lambda d: (-frequency[d], d))[:4]
    assert [(d, frequency[d]) for d in top4] == [('Lifestyle', 36), ('Education', 29), ('Career', 19), ('Business', 18)]
    definitions = [('SFT+GRPO', RUN, 'ordinary_sft_grpo'), ('ECLP', RUN / 'eclp_comparison', 'our_direct')]
    contracts = [read_json(p / 'generation_run.json') for _, p, _ in definitions]
    for key in ['expected_indexes', 'decode', 'seed', 'compute_dtype', 'quantization', 'models', 'protocol_sha256', 'batching_sha256', 'selection_sha256']:
        assert contracts[0][key] == contracts[1][key], key
    rows, evidence = [], {}
    judge_contracts = []
    for method, directory, label in definitions:
        gen_path = directory / 'generation' / f'{label}.jsonl'
        gen = {r['index']: r for r in read_jsonl(gen_path)}
        assert len(gen) == 192 and set(gen) == selected
        judge_path = directory / 'judges/selene8b/evaluation' / label / 'judgments.jsonl'
        judgments = read_jsonl(judge_path)
        assert len(judgments) == 192 and {r['index'] for r in judgments} == selected
        judge_contract = read_json(directory / 'judges/selene8b/evaluation_run.json')
        judge_contracts.append(judge_contract)
        for r in judgments:
            i = r['index']
            assert r['scenario_id'] == gen[i]['scenario_id'] == points[i]['scenario_id']
            assert r['source_row_hash'] == object_hash(gen[i])
            assert r['contract_hash'] == object_hash(judge_contract)
            assert gen[i]['initial_history'] == [] and len(gen[i]['trajectory']) == 10
        expected = read_json(judge_path.parent / 'summary.json')
        actual = summarize(judgments, 192)
        for key in ['success_percent', 'win1_percent', 'win_full_percent', 'mean_rounds_fail5']:
            assert actual[key] == expected[key], key
        evidence[method] = {'generation_path': str(gen_path), 'generation_sha256': digest(gen_path),
            'judgments_path': str(judge_path), 'judgments_sha256': digest(judge_path), 'overall': actual}
        for domain in ['Overall'] + top4:
            records = judgments if domain == 'Overall' else [r for r in judgments if domain in domains(points[r['index']]['scenario']['domain'])]
            result = summarize(records, len(records))
            rows.append({'method': method, 'domain': domain, 'N': len(records),
                'Success': result['success_percent'], 'Win@1': result['win1_percent'],
                'Win-Full': result['win_full_percent'], 'Turn': result['mean_rounds_fail5'],
                'success_effective_n': result['success_evaluable_n'], 'win1_effective_n': result['win1_evaluable_n'],
                'win_full_effective_n': result['win_full_evaluable_n'], 'success_unresolved_rounds': result['format_unresolved'],
                'win1_unresolved': result['win1_unresolved'], 'win_full_unresolved': result['win_full_unresolved']})
    for key in ['judge_model', 'model_files', 'golden_sha256', 'protocol_sha256', 'win_rules_sha256', 'original_win_protocol_sha256', 'goal_schema', 'success_rule', 'win_credit', 'compute_dtype', 'quantization', 'temperature']:
        assert judge_contracts[0][key] == judge_contracts[1][key], key
    sft_summary = read_json(RUN / 'training/sft/summary.json')
    grpo_summary = read_json(RUN / 'training/grpo/summary.json')
    assert sft_summary['complete'] and sft_summary['sessions'] == 59868
    assert grpo_summary['complete'] and grpo_summary['contexts'] == 1800 and grpo_summary['reference_unchanged']
    ordinary_log = read_jsonl(RUN / 'training/sft/train.jsonl')
    eclp_log_path = ROOT / 'eclp_runs_gpu4/sft_full/train.jsonl'
    eclp_log = read_jsonl(eclp_log_path)
    assert len(ordinary_log) == len(eclp_log) == 7484
    assert ordinary_log[-1]['position'] == eclp_log[-1]['next_position'] == 59868
    assert [r['step'] for r in eclp_log] == list(range(1, 7485))
    segments = []
    previous = None
    for record in eclp_log:
        if previous and record['seconds_since_start'] < previous['seconds_since_start']:
            segments.append(previous['seconds_since_start'])
        previous = record
    segments.append(eclp_log[-1]['seconds_since_start'])
    eclp_sft_seconds = sum(segments)
    eclp_rl = read_json(RUN / 'eclp_release/offline_rl/summary.json')
    migration = read_json(RUN / 'eclp_release/migration.json')
    assert migration['weight_file_hashes_identical'] and migration['prepared_data_file_hashes_identical']
    release_lock = read_json(RUN / 'eclp_release/protocol.lock.json')
    sft_manifest = read_json(RUN / 'training/sft/manifest.json')
    assert sft_manifest['prompt_source_sha256'] == release_lock['sources']['exp_code_ljj/repro/persuadee_simulation.py']
    assert eclp_rl['complete'] and eclp_rl['contexts'] == 1800 and eclp_rl['continued_from_run_hash'] is None
    assert contracts[1]['checkpoint_id'] == eclp_rl['checkpoint_id']
    assert contracts[0]['checkpoint_id'] == digest(Path(grpo_summary['checkpoint']) / 'adapter/adapter_model.safetensors')
    role_migration = read_json(RUN / 'training/grpo/role_name_migration.json')
    assert role_migration['new_source_sha256'] == digest(RUN / 'source/train_text_grpo.py')
    discarded_seconds = role_migration['discarded_uncommitted_sampling_seconds']
    assert discarded_seconds >= 0
    timing = [
        {'method': 'SFT+GRPO', 'sft_minutes': sft_summary['training_seconds'] / 60,
         'rl_minutes': grpo_summary['training_seconds'] / 60},
        {'method': 'ECLP', 'sft_minutes': eclp_sft_seconds / 60,
         'rl_minutes': eclp_rl['seconds_this_invocation'] / 60}]
    for r in timing:
        r['total_minutes'] = r['sft_minutes'] + r['rl_minutes']
        r['discarded_retry_minutes'] = discarded_seconds / 60 if r['method'] == 'SFT+GRPO' else 0.0
        r['total_with_recorded_retries_minutes'] = r['total_minutes'] + r['discarded_retry_minutes']
    result = {'complete_generation_and_evaluation': True, 'needs_owner_completion_audit': True,
        'rows': rows, 'timing': timing, 'evidence': evidence, 'top4': top4,
        'domain_frequencies': dict(frequency), 'test_selection_sha256': digest(ROOT / 'exp_code_ljj/data/test_selection_192.json'),
        'eclp_sft_log_sha256': digest(eclp_log_path), 'eclp_sft_invocation_segment_seconds': segments,
        'dependency_migration': migration,
        'grpo_role_name_migration': role_migration,
        'timing_note': 'Ordinary SFT: summary timer includes data processing and intermediate/final saves, excludes initial model loading. ECLP SFT: historical optimizer-loop segments include intermediate saves, exclude loading and final save; not a fresh shared-environment benchmark. RL: complete fresh 1800-context epochs including rollouts/rewards/updates/checkpointing, excluding initial loading. Discarded incomplete GRPO sampling is reported separately; totals with retries include this recorded failure overhead, not downtime or failed startup loading.'}
    atomic_json(RUN / 'comparison_details.json', result)
    for filename, records in [('comparison_scores.csv', rows), ('comparison_training_time.csv', timing)]:
        with (RUN / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    lines = ['# Qwen2.5 普通 SFT+GRPO 与 ECLP：Selene 比较', '',
        '固定192条无前置历史场景，五轮对话；生成、模拟器、评分提示词和解码设置已核对一致。四领域按场景计数，多标签分组有重叠。', '',
        '| Domain | N | Method | Success (%) | Win@1 (%) | Win-Full (%) | Turn | Success有效N | 未解析回合 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for domain in ['Overall'] + top4:
        for r in [r for r in rows if r['domain'] == domain]:
            fmt = lambda value, places=2: '-' if value is None else f'{value:.{places}f}'
            lines.append(f"| {domain} | {r['N']} | {r['method']} | {fmt(r['Success'])} | {fmt(r['Win@1'])} | {fmt(r['Win-Full'])} | {fmt(r['Turn'], 3)} | {r['success_effective_n']} | {r['success_unresolved_rounds']} |")
    lines += ['', '| Method | SFT 1 epoch (min) | RL 1 epoch (min) | Sum (min) |', '|---|---:|---:|---:|']
    for r in timing:
        lines.append(f"| {r['method']} | {r['sft_minutes']:.2f} | {r['rl_minutes']:.2f} | {r['total_minutes']:.2f} |")
    lines += ['', '普通SFT采用最终summary计时，含数据处理、中间及最终保存，不含初始模型加载。ECLP SFT为历史优化循环两段计时，含中间保存，不含加载和最终保存；计时边界及运行环境存在差异，不能作为严格同环境速度基准。RL两组均为本次物理GPU4上的完整1800条一轮，计入采样、奖励、优化及保存，不含初始加载。', '',
        f'普通GRPO因角色字段格式修复而丢弃的未提交采样为{discarded_seconds / 60:.2f}分钟，另行记录，不计入上表有效训练时间。含这次已记录重跑开销的SFT+GRPO合计为{timing[0]["total_with_recorded_retries_minutes"]:.2f}分钟，不含停机等待及失败启动的加载时间。', '',
        'ECLP本次从原联合SFT重跑统一配置的LPO，评分对应这次实际checkpoint，不与旧主表分数混用。普通GRPO更新文本LoRA，ECLP更新连续latent policy；两者奖励、KL估计方式和架构存在方法固有差异。', '',
        '原训练锁中基础依赖的旧源码未在已查备份中找到。本次建立显式新依赖锁并记录代码哈希迁移；原联合SFT全部权重和准备数据逐文件哈希保持一致，原锁、原checkpoint和当前项目代码均未覆盖。', '',
        '逐场景记录、配置、训练进度、执行命令、权重身份和文件哈希保留在本run目录；原主表和原checkpoint未覆盖。']
    (RUN / 'comparison.md').write_text('\n'.join(lines) + '\n')
    result_dir = ROOT / 'exp_code_ljj/results'
    for name in ['comparison.md', 'comparison_scores.csv', 'comparison_training_time.csv']:
        (result_dir / ('qwen25_standard_grpo_' + name)).write_bytes((RUN / name).read_bytes())
    print(json.dumps({'completed_artifacts': str(RUN), 'timing': timing, 'overall': [r for r in rows if r['domain'] == 'Overall']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
