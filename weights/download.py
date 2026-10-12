#!/usr/bin/env python3
"""Download published unmerged adapters/modules and verify all SHA-256 values."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true')
    group.add_argument('--name', action='append')
    args = parser.parse_args()
    rows = json.loads((ROOT / 'weights/manifest.json').read_text())
    if args.name:
        known = {row['name'] for row in rows}
        missing = set(args.name) - known
        if missing:
            parser.error('Unknown packages: ' + ', '.join(sorted(missing)))
        rows = [row for row in rows if row['name'] in args.name]
    cache = ROOT / '.weight-downloads'
    cache.mkdir(exist_ok=True)
    for row in rows:
        path = cache / row['asset']
        if not path.exists() or digest(path) != row['asset_sha256']:
            temp = path.with_suffix(path.suffix + '.partial')
            urllib.request.urlretrieve(row['download_url'], temp)
            if digest(temp) != row['asset_sha256']:
                raise ValueError('Archive checksum mismatch: ' + row['name'])
            temp.replace(path)
        allowed = {item['path'] for item in row['files']}
        with tarfile.open(path, 'r:gz') as archive:
            members = archive.getmembers()
            for member in members:
                target = (ROOT / member.name).resolve()
                if (member.name not in allowed or not member.isfile()
                        or ROOT not in target.parents):
                    raise ValueError('Unexpected archive member: ' + member.name)
            archive.extractall(ROOT, members=members)
        for item in row['files']:
            if digest(ROOT / item['path']) != item['sha256']:
                raise ValueError('File checksum mismatch: ' + item['path'])
        print('Verified:', row['name'], flush=True)

if __name__ == '__main__':
    main()
