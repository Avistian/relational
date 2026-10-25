"""Independent oracle: Bash expansion and GNU cp into disposable copy tree.

No generator imports and no parser reuse. Skip only the two content-version
commands. Every copy, mkdir, condition, loop and removal runs as in the workflow.
Use reflinks where supported, otherwise GNU cp copies ordinary bytes.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[6]
PROOF = Path(__file__).resolve().parent
workflow = (ROOT / '.github/workflows/pages.yml').read_text()
block = workflow.split('      - name: Build site\n        run: |\n', 1)[1].split('\n      - name:', 1)[0]
lines = [line[10:] for line in block.splitlines() if line.startswith('          ')]
lines = [line for line in lines if not line.startswith(('VER=', 'sed '))]
with tempfile.TemporaryDirectory(prefix='rel16-outside-') as directory:
    public = Path(directory) / 'public'
    script = ('set -eu\ncp() { command cp --reflink=auto "$@"; }\n' +
              '\n'.join(lines).replace('public/', str(public) + '/'))
    (PROOF / 'outside-command.sh.txt').write_text(script)
    result = subprocess.run(['bash', '-c', script], cwd=ROOT, capture_output=True, text=True)
    (PROOF / 'outside.stdout.txt').write_text(result.stdout)
    (PROOF / 'outside.stderr.txt').write_text(result.stderr)
    if result.returncode:
        raise SystemExit('outside shell copy failed: ' + str(result.returncode))
    actual = []
    for current, dirs, names in os.walk(public):
        for name in names + [name for name in dirs if (Path(current) / name).is_symlink()]:
            actual.append((Path(current) / name).relative_to(public).as_posix())
    (PROOF / 'outside-files.json').write_text(json.dumps(sorted(actual), indent=2) + '\n')
    generated = subprocess.run([str(ROOT / '.venv/bin/python'), '-B',
                               'scripts/pages_publish_list.py', '--compare', '.github/workflows/pages.yml'],
                              cwd=ROOT, capture_output=True, text=True)
    (PROOF / 'compare.stdout.json').write_text(generated.stdout)
    (PROOF / 'compare.stderr.txt').write_text(generated.stderr)
    if generated.returncode:
        raise SystemExit('generator compare failed: ' + str(generated.returncode))
    report = json.loads(generated.stdout)
    expected, observed = set(report['files']), set(actual)
    receipt = dict(equal=expected == observed, generated_count=len(expected), actual_count=len(observed),
                   only_generated=sorted(expected-observed), only_shell=sorted(observed-expected),
                   missing_sources=report['missing_sources'], missing_references=report['missing_references'],
                   omitted_references=report['omitted_references'], shell_exit_code=result.returncode,
                   compare_exit_code=generated.returncode,
                   source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    (PROOF / 'outside.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    if not receipt['equal']:
        raise SystemExit(1)
