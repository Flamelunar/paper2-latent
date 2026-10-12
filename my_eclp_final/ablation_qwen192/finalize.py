"""Finalize finished ablations without modifying experimental observations."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from supervise import A, D, R, report
from audit import inspect


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    status = json.loads((A / 'status.json').read_text())
    if status['stage'] != 'complete':
        raise RuntimeError('Supervisor has not finished; do not finalize partial results')
    report()
    audit = inspect()
    if not audit['complete']:
        raise RuntimeError('Completion audit incomplete')
    manifest = json.loads((A / 'experiment_manifest.json').read_text())
    # Keep recorded execution-source hashes intact; reporting changes are separate.
    manifest['final_delivered_code_sha256'] = {
        p.name: sha(p) for p in sorted(D.iterdir())
        if p.is_file() and p.suffix in {'.py', '.json'}
    }
    manifest['reporting_revision'] = 'explicit effective denominators and KL-equivalence disclosure'
    (A / 'experiment_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    diffs = []
    for name in ['generate.py', 'evaluate.py', 'simulator_prompt.py', 'kl_only.py',
                 'supervise.py', 'early_evaluate.py', 'audit.py', 'finalize.py']:
        original = R / 'my_eclp_final/seven_methods_192' / (
            'generate_direct.py' if name == 'generate.py' else name)
        left = str(original) if original.exists() else '/dev/null'
        result = subprocess.run(['git', 'diff', '--no-index', '--', left, str(D / name)],
                                capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        diffs.append(result.stdout)
    (A / 'code_changes.patch').write_text(''.join(diffs))
    requirements = {
        'historical_actor_and_judge_protocol': ['experiment_manifest.json', 'main_table_snapshot.md', 'completion_audit.json'],
        'joint_sft_and_final_lpo_checkpoint_identity': ['checkpoint_provenance.json', 'wo_g/generation_run.json', 'full_ours/generation_run.json'],
        'fixed_192_ids_and_empty_initial_history': ['test_ids.json', 'completion_audit.json'],
        'simulator_background_public_history_only_and_matched_control': ['experiment_manifest.json', 'wo_g/generation_run.json', 'full_ours/generation_run.json'],
        'bf16_no_quantization_seed_and_decode': ['wo_g/generation_run.json', 'full_ours/generation_run.json'],
        'state_strategy_and_mean_action_mechanism': ['wo_g/generation.log', 'full_ours/generation.log'],
        'task_rewards_off_policy_only_exact_kl_and_no_drift': ['wo_r_kl_only/diagnostic.json', 'wo_r_kl_only/updates.jsonl'],
        'kl_only_reuse_verified_not_independent': ['wo_r_kl_only/reuse_manifest.json', 'completion_audit.json'],
        'actual_commands': ['commands.jsonl'],
        'both_judges_complete_actual_records_and_metrics': ['completion_audit.json', 'scores.json'],
        'training_sources_and_train_test_disjointness': ['experiment_manifest.json'],
        'real_generation_failures_and_effective_denominators': ['scores.json', '实验说明.md'],
        'delivered_code_changes_and_documentation': ['code_changes.patch', '实验说明.md'],
    }
    for evidence in requirements.values():
        for name in evidence:
            if not (A / name).is_file():
                raise RuntimeError('Missing required evidence: ' + name)
    (A / 'requirement_evidence.json').write_text(
        json.dumps(requirements, ensure_ascii=False, indent=2) + '\n')
    sources = sorted(p for p in A.rglob('*') if p.is_file() and not p.is_symlink()
                     and p.name != 'delivery_sha256.json' and '.lock' not in p.name)
    checksums = {str(p.relative_to(A)): sha(p) for p in sources}
    (A / 'delivery_sha256.json').write_text(json.dumps(checksums, indent=2) + '\n')
    print(json.dumps({'complete': True, 'verified_files': len(checksums),
                      'results': str(R / 'exp_code_ljj/results/ablation_qwen25_192.csv')}, indent=2))


if __name__ == '__main__':
    if '--watch' in sys.argv:
        status_path = A / 'status.json'
        current = json.loads(status_path.read_text())
        pid = current['pid']
        while True:
            current = json.loads(status_path.read_text())
            if current['stage'] in {'complete', 'failed'}:
                break
            cmdline = Path('/proc') / str(pid) / 'cmdline'
            if not cmdline.exists() or b'ablation_qwen192/supervise.py' not in cmdline.read_bytes():
                raise RuntimeError('Supervisor missing without terminal status')
            time.sleep(20)
    main()
