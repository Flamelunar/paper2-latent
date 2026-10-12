"""Undo only recorded organization moves, without starting experiments."""
import json
from pathlib import Path

ROOT=Path('/app/hzj/liujianjian/ljj-2-paper')
manifest=Path(__file__).with_name('manifest.json')
data=json.loads(manifest.read_text())
for item in reversed(data['actions']):
    source=ROOT/item['source'];target=ROOT/item['target']
    if not target.exists():
        if source.exists():continue
        raise FileNotFoundError(target)
    if source.is_symlink():
        assert source.resolve()==target.resolve()
        source.unlink()
    assert not source.exists(),source
    target.rename(source)
print('Recorded moves restored. Classification links and guide files retained.')
