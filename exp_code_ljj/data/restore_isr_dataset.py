"""Restore the original ISR dataset from its lossless compressed distribution."""
import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
metadata = json.loads((root / 'isr_sft.archive.json').read_text())
target = root / 'isr_sft.jsonl'
if target.exists():
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != metadata['sha256']:
        raise RuntimeError('Existing ISR dataset differs; it has not been overwritten.')
    print('ISR dataset already restored and verified.')
else:
    temporary = root / 'isr_sft.jsonl.restoring'
    digest = hashlib.sha256()
    total = 0
    with gzip.open(root / 'isr_sft.jsonl.gz', 'rb') as source, temporary.open('wb') as output:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            output.write(chunk)
    if digest.hexdigest() != metadata['sha256'] or total != metadata['bytes']:
        temporary.unlink()
        raise RuntimeError('Dataset checksum mismatch.')
    temporary.replace(target)
    print('ISR dataset restored; original SHA-256 and size verified.')
