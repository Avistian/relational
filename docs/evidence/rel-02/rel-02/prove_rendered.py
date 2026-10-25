"""Compare actual rendered bytes with the inventory captured before REL-02."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
before = json.loads((OUT/'rendered-before.json').read_text())
tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
paths = [p for p in tracked if p and (p.endswith('.html') or p.startswith('assets/'))
         and (ROOT/p).is_file()]
after = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
(OUT/'rendered-after.json').write_text(json.dumps(after, indent=2, sort_keys=True)+'\n')
result = dict(before_files=len(before), after_files=len(after),
              added=sorted(after.keys()-before.keys()),
              removed=sorted(before.keys()-after.keys()),
              changed=sorted(p for p in before.keys() & after.keys() if before[p]!=after[p]),
              source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              actual_byte_hashes_equal=before==after)
(OUT/'rendered-comparison.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
assert before == after, 'Rendered bytes changed'
