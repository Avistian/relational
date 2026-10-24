"""Record a bounded read-only command and its raw output for REL-02."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
name, *command = sys.argv[1:]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
t0 = time.monotonic()
try:
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=180)
    code, stdout, stderr = result.returncode, result.stdout, result.stderr
except subprocess.TimeoutExpired as exc:
    code, stdout, stderr = 124, exc.stdout or b'', exc.stderr or b''
(OUT / (name + '.stdout.txt')).write_bytes(stdout)
(OUT / (name + '.stderr.txt')).write_bytes(stderr)
status = dict(command=command, cwd=str(ROOT), started_utc=started,
              exit_code=code, duration_s=round(time.monotonic()-t0, 3),
              source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              source_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
              definition_sha256=hashlib.sha256((ROOT/'labs/_solution_map_content.py').read_bytes()).hexdigest(),
              test_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT/'tests/visual_contracts').glob('*.py'))})
(OUT / (name + '.status.json')).write_text(json.dumps(status, indent=2)+'\n')
print(json.dumps(status))
print(stdout.decode(errors='replace')[-2500:])
print(stderr.decode(errors='replace')[-5000:])
sys.exit(code)
