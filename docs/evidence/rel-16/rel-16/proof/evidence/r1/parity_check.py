"""Builder file-set proof using Bash/GNU cp, independent of the parser.

Adapted from the retained builder oracle. This is a publication inventory,
not a canonical TDD replay or an independent verifier review.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import yaml


ROOT = Path('/tmp/lead-wt/night-rel16-r1')
OUT = Path(__file__).resolve().parent
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True))
workflow_path = ROOT / '.github/workflows/pages.yml'
workflow = yaml.safe_load(workflow_path.read_text())
plans = [step['run'] for step in workflow['jobs']['build']['steps']
         if step.get('name') == 'Build site']
assert len(plans) == 1
lines = plans[0].splitlines()
skipped = [line for line in lines if line.strip().startswith(('VER=', 'sed '))]
assert len(skipped) == 2
# Only the two existing HTML byte mutations are omitted. Copy operands,
# conditionals, the numbered loop and the literal removal run in Bash.
assert skipped[0].strip().startswith('VER=$(python3 -c ')
assert skipped[1].strip().startswith('sed -i ')
with tempfile.TemporaryDirectory(prefix='rel16-r1-publication-') as temporary:
    public = Path(temporary) / 'public'
    script = ('set -eu\ncp() { command cp --reflink=auto "$@"; }\n' +
              '\n'.join(line for line in lines if line not in skipped)
              .replace('public/', str(public) + '/'))
    (OUT / 'workflow-copy.sh.txt').write_text(script)
    result = subprocess.run(['bash', '-c', script], cwd=ROOT, capture_output=True, text=True)
    (OUT / 'workflow-copy.stdout.txt').write_text(result.stdout)
    (OUT / 'workflow-copy.stderr.txt').write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    actual = []
    for current, directories, names in os.walk(public):
        for name in names + [name for name in directories if (Path(current) / name).is_symlink()]:
            actual.append((Path(current) / name).relative_to(public).as_posix())
    (OUT / 'actual-files.json').write_text(json.dumps(sorted(actual), indent=2) + '\n')
compared = subprocess.run([str(ROOT / '.venv/bin/python'), '-B',
                          'scripts/pages_publish_list.py', '--compare',
                          '.github/workflows/pages.yml'], cwd=ROOT, capture_output=True, text=True)
(OUT / 'compare.stdout.json').write_text(compared.stdout)
(OUT / 'compare.stderr.txt').write_text(compared.stderr)
assert compared.returncode == 0, compared.stderr
report = json.loads(compared.stdout)
expected, observed = set(report['files']), set(actual)
retained = json.loads((ROOT / 'docs/evidence/rel-16/rel-16/evidence/builder/outside-files.json').read_text())
receipt = dict(source=dict(head=head, dirty=dirty), workflow_sha256=hashlib.sha256(workflow_path.read_bytes()).hexdigest(),
               generated_count=len(expected), actual_count=len(observed), equal=expected == observed,
               retained_inventory_equal=sorted(actual) == retained,
               only_generated=sorted(expected - observed), only_shell=sorted(observed - expected),
               shell_exit_code=result.returncode, compare_exit_code=compared.returncode,
               comparison=report['comparison'],
               diagnostics={key: report[key] for key in
                            ('missing_sources', 'missing_references', 'omitted_references')})
(OUT / 'parity.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
assert receipt['equal'] and receipt['retained_inventory_equal']
assert receipt['generated_count'] == receipt['actual_count'] == 18497
assert all(not values for values in receipt['diagnostics'].values())
