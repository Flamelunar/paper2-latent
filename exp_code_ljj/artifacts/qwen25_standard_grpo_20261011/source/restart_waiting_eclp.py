"""Replace only this task's idle CPU continuation after the explicit migration."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from supervise import ROOT, RUN, PYTHON

assert (RUN / 'eclp_release/migration.json').is_file()
state = json.loads((RUN / 'supervisor_state.json').read_text())
assert state['stage'] == 'waiting_for_sft'
old_state = json.loads((RUN / 'eclp_supervisor_state.json').read_text())
assert old_state['stage'] == 'waiting_for_standard_comparison'
launch = json.loads((RUN / 'eclp_supervisor_launch.json').read_text())
pid = launch['pid']
proc = Path('/proc') / str(pid)
command = proc.joinpath('cmdline').read_bytes().split(b'\0')
assert str(RUN / 'source/continue_eclp.py').encode() in command and b'--worker' in command
assert not proc.joinpath(f'task/{pid}/children').read_text().strip()
archive = RUN / 'eclp_dependency_migration_supervisor'
archive.mkdir()
os.kill(pid, signal.SIGTERM)
for _ in range(50):
    if not proc.exists() or '\nState:\tZ' in proc.joinpath('status').read_text():
        break
    time.sleep(0.1)
else:
    raise RuntimeError('Idle continuation did not terminate; do not duplicate it')
for name in ['eclp_supervisor_launch.json', 'eclp_supervisor_state.json']:
    (RUN / name).rename(archive / name)
(archive / 'restart_reason.json').write_text(json.dumps({'stopped_pid': pid, 'unix': time.time(),
    'reason': 'Update idle CPU continuation to the explicit dependency-migrated release; SFT/GPU jobs untouched'}))
subprocess.run([PYTHON, str(RUN / 'source/continue_eclp.py')], cwd=ROOT, check=True)
