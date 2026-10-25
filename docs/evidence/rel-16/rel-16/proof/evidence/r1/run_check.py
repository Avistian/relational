"""Capture builder commands without rewriting retained evidence."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

out = Path(__file__).resolve().parent
label, *argv = sys.argv[1:]
root = Path('/tmp/lead-wt/night-rel16-r1')
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
result = subprocess.run(argv, cwd=root, capture_output=True, text=True)
(out / (label + '.stdout.txt')).write_text(result.stdout)
(out / (label + '.stderr.txt')).write_text(result.stderr)
status = dict(argv=argv, cwd=str(root), exit_code=result.returncode,
              source=dict(head=head, dirty=bool(dirty), porcelain=dirty),
              created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out / (label + '.json')).write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status, indent=2))
print(result.stdout)
print(result.stderr)
sys.exit(result.returncode)
