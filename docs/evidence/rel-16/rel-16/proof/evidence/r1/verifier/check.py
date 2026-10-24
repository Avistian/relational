"""Read-only verifier: focused execution, external receipts and static history audit."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
ROOT = Path('/tmp/lead-wt/night-vrel16-r1')
OUT = Path(__file__).resolve().parent
R1 = OUT.parent
BASE = 'fa4521280ab256beb622bfa3e0c8ec66e2ef68b7'
SHA = '86fe6e7284421f128686da00240aeddb5595cbab'
def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)
def digest(data):
    return hashlib.sha256(data).hexdigest()
def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')
source = dict(head=git('rev-parse', 'HEAD').decode().strip(), dirty=bool(git('status', '--porcelain')))
assert source == dict(head=SHA, dirty=False)
argv = [str(ROOT / '.venv/bin/python'), '-B', '-m', 'unittest', 'discover', '-s', 'tests/pages', '-v']
r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
(OUT / 'focused.stdout.txt').write_text(r.stdout)
(OUT / 'focused.stderr.txt').write_text(r.stderr)
save('focused.json', dict(source=source, argv=argv, exit_code=r.returncode, verdict='PASS' if r.returncode == 0 else 'FAIL'))
assert r.returncode == 0, r.stderr
builder = json.loads((R1 / 'READY.json').read_text())
checks = {}
for path, expected in builder['artifacts_sha256'].items():
    checks[path] = digest(Path(path).read_bytes()) == expected
for path, expected in builder['source_sha256'].items():
    checks[path] = digest((ROOT / path).read_bytes()) == expected
assert all(checks.values()), checks
integrity = json.loads((R1 / 'integrity.json').read_text())
historical = {path: digest((ROOT / path).read_bytes()) == expected and git('show', BASE+':'+path) == (ROOT/path).read_bytes()
              for path, expected in integrity['historical_evidence_unchanged'].items()}
assert all(historical.values())
frozen = json.loads((ROOT/'docs/evidence/rel-16/rel-16/evidence/builder/frozen-tests.json').read_text())
frozen_check = {path: all(digest(git('show', commit+':'+path)) == expected for commit in ['9e09a39b', '1ba8e647', BASE, SHA]) for path, expected in frozen.items()}
assert all(frozen_check.values())
cycles = []
for cycle in builder['tdd']['cycles']:
    red, green = cycle['red_sha'], cycle['green_sha']
    tests_equal = git('show', red+':tests/pages/test_publish_list_shell_regressions.py') == git('show', green+':tests/pages/test_publish_list_shell_regressions.py')
    red_paths = git('diff-tree', '--no-commit-id', '--name-only', '-r', red).decode().splitlines()
    green_paths = git('diff-tree', '--no-commit-id', '--name-only', '-r', green).decode().splitlines()
    status = json.loads((R1/(cycle['red_log']+'.json')).read_text())
    log = (R1/(cycle['red_log']+'.stderr.txt')).read_text()
    summary = re.search(r'FAILED \(failures=(\d+)\)', log)
    assert summary and int(summary[1]) == cycle['assertion_failures']
    assert not re.search(r'^ERROR:', log, re.M)
    assert status['source'] == dict(head=red, dirty=False, porcelain='') and status['exit_code'] == 1
    assert tests_equal and red_paths == ['tests/pages/test_publish_list_shell_regressions.py'] and green_paths == ['scripts/pages_publish_list.py']
    assert git('show', red+'^:scripts/pages_publish_list.py') == git('show', red+':scripts/pages_publish_list.py')
    cycles.append(dict(**cycle, independently_checked=True, red_paths=red_paths, green_paths=green_paths,
                       green_status=json.loads((R1 / {'REL16-V1':'v1-green.json','REL16-V2 wildcard':'v2-glob-green.json','REL16-V2 root corrected':'v2-root-green.json'}[cycle['behavior']]).read_text())))
correction = git('show', 'c1a98c2e', '--', 'tests/pages/test_publish_list_shell_regressions.py').decode()
(OUT/'fixture-correction.diff.txt').write_text(correction)
invalid_log=(R1/'v2-root-red-committed.stderr.txt').read_text()
assert 'public/.' in invalid_log and 'refusing to remove' in invalid_log
assert re.search(r'FAILED \(failures=10\)', invalid_log)
assert git('show', '3a628006:scripts/pages_publish_list.py') == git('show', 'c1a98c2e:scripts/pages_publish_list.py')
old_ready = R1.parent/'verifier/original-builder-READY.json'
old = json.loads(old_ready.read_text())
old_hashes = {path: digest(Path(path).read_bytes()) == expected for path,expected in old.get('artifacts_sha256',{}).items()}
# Preserve old receipt's actual schema and provenance; do not invent a committed artifact.
packaging = {}
for commit in [BASE, SHA]:
    tree = git('ls-tree','-r','--name-only',commit,'docs/evidence/rel-16/rel-16').decode().splitlines()
    packaging[commit] = {name: 'docs/evidence/rel-16/rel-16/'+name in tree for name in ['README.md','READY.json','report.md']}
assert all(p == dict(**{'README.md':True,'READY.json':False,'report.md':False}) for p in packaging.values())
changed=git('diff','--name-only',BASE,SHA).decode().splitlines()
assert changed == builder['changed_files']
assert git('show',BASE+':.github/workflows/pages.yml') == (ROOT/'.github/workflows/pages.yml').read_bytes()
diffcheck = subprocess.run(['git','diff','--check',BASE,SHA],cwd=ROOT,capture_output=True,text=True)
assert diffcheck.returncode == 0
for name in ['scripts/pages_publish_list.py','tests/pages/test_publish_list.py','tests/pages/test_publish_list_shell_regressions.py']:
    ast.parse((ROOT/name).read_text(),filename=name)
config_path=Path('/home/avist/agent-stack/lead/config/verify-projects.json')
config=json.loads(config_path.read_text())['relational']
save('verify-project-not-run.json',dict(verdict='NOT_RUN',exit_code=None,source=source,configured=config,config_sha256=digest(config_path.read_bytes()),reason='Configured unittest discovery includes tests/relmini/test_relmini.py:69-78 -> labs/relmini/baseline.py:72-78 -> LogisticRegression model.fit. User prohibits model fitting. No lead-verify execution or config changes. Historical 74 PASS does not cover R1.'))
save('static-history.json',dict(source=source,changed_files=changed,artifact_hash_checks=checks,historical_evidence_unchanged=historical,frozen_tests=frozen_check,cycles=cycles,invalid_fixture=dict(sha='3a628006af7cf984832ef3f187bf79355d54a257',failures=10,genuine_parser_failures=9,invalid_shell_expectations=1,corrected_before_production_fix=True),old_ready=dict(path=str(old_ready),sha256=digest(old_ready.read_bytes()),artifact_hash_checks=old_hashes),package_tree=packaging,workflow_unchanged=True,ast_parse='PASS',diff_check='PASS'))
print('OK focused Pages tests 10; artifact hashes; original frozen tests; three R1 cycles; correction history; source/package checks')
