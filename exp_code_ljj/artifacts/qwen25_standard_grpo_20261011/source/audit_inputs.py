"""Read-only provenance and timing audit for the Qwen2.5 GRPO comparison."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/app/hzj/liujianjian/ljj-2-paper')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    out = {'root': str(ROOT), 'qwen_adapters': [], 'timing': {}}
    for root in [ROOT, Path('/app/hzj/liujianjian/persuader_atapter_save')]:
        for p in root.rglob('adapter_config.json'):
            if any(s.startswith('checkpoint-') for s in p.parts):
                continue
            cfg = json.loads(p.read_text())
            if 'qwen2.5' in str(cfg.get('base_model_name_or_path', '')).lower():
                out['qwen_adapters'].append({'path': str(p), 'sha256': sha(p), 'config': cfg})
    p = ROOT / 'eclp_runs_gpu4/sft_full/train.jsonl'
    if p.exists():
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        boundaries = []
        previous = None
        for row in rows:
            t = row['seconds_since_start']
            if previous is not None and t < previous['seconds_since_start']:
                boundaries.append({'last_step': previous['step'], 'seconds': previous['seconds_since_start'],
                                   'next_step': row['step']})
            previous = row
        out['timing']['eclp_sft'] = {'path': str(p), 'sha256': sha(p), 'rows': len(rows),
            'first': rows[0], 'last': rows[-1], 'invocation_resets': boundaries,
            'sum_observed_invocation_seconds': sum(x['seconds'] for x in boundaries) + rows[-1]['seconds_since_start'],
            'note': 'Observed log endpoints; checkpoint save tails and unlogged interrupted work require separate audit.'}
    for name in ['summary.json', 'continuation_manifest.json']:
        p = ROOT / 'eclp_runs_gpu4/offline_rl' / name
        if p.exists():
            out['timing'][name] = json.loads(p.read_text())
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
