"""Static audit of original cycle, canonical receipt, and all retained local logs."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
ROOT=Path('/tmp/lead-wt/night-vrel16-r1')
OUT=Path(__file__).resolve().parent
R1=OUT.parent
E=ROOT/'docs/evidence/rel-16/rel-16/evidence/builder'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT).decode().strip()
receipt=json.loads((E/'builder-tdd-receipt.json').read_text())
assert sha(E/'builder-tdd.json')==receipt['report_sha256']
assert receipt['head']=='1ba8e647d4ef15eb573b8caa13033cdad5e3b9ff'
assert receipt['red']=='9e09a39b' and len(receipt['test_ids'])==7
receipt_checks={p:sha(Path(p))==h for p,h in receipt['artifacts'].items()}
assert all(receipt_checks.values())
assert sha(E/'canonical-red.log') in receipt['artifacts'].values()
assert sha(E/'canonical-green.log') in receipt['artifacts'].values()
assert 'FAILED (failures=6)' in (E/'canonical-red.log').read_text()
assert '"failed": 6, "errors": 0' in (E/'canonical-red.log').read_text()
assert '"failed": 0, "errors": 0' in (E/'canonical-green.log').read_text()
assert git('diff-tree','--no-commit-id','--name-only','-r','9e09a39b').splitlines()==['tests/pages/__init__.py','tests/pages/test_publish_list.py']
assert git('diff-tree','--no-commit-id','--name-only','-r','1ba8e647').splitlines()==['scripts/pages_publish_list.py']
logs={}
for status_path in sorted(R1.glob('*.json')):
    if not status_path.with_suffix('.stderr.txt').exists(): continue
    status=json.loads(status_path.read_text())
    log=status_path.with_suffix('.stderr.txt').read_text()
    logs[status_path.name]=dict(exit_code=status.get('exit_code'),source=status.get('source'),summary=re.findall(r'(?:Ran \d+ tests?[^\n]*|FAILED \([^\n]+\)|^OK$)',log,re.M),sha256=sha(status_path.with_suffix('.stderr.txt')))
    if 'green' in status_path.name or status_path.name=='final-focused.json':
        assert status['exit_code']==0 and re.search(r'^OK$',log,re.M)
old_ready=R1.parent/'verifier/original-builder-READY.json'
old=json.loads(old_ready.read_text())
old_checks={p:sha(Path(p))==h for p,h in old['artifacts_sha256'].items()}
assert all(old_checks.values())
original=Path('/tmp/lead-wt/night-rel-16/docs/evidence/rel-16/rel-16/evidence/builder/READY.json')
assert original.read_bytes()==old_ready.read_bytes()
retained_status=json.loads((R1.parent/'verifier/retained-exact-head-verify-status.json').read_text())
assert retained_status['source']['head']=='fa4521280ab256beb622bfa3e0c8ec66e2ef68b7'
result=dict(original_cycle=dict(red=git('rev-parse','9e09a39b'),green=receipt['head'],tests=7,red_failures=6,runner_errors=0,unchanged_frozen_tests=True,receipt_hash_checks=receipt_checks,canonical_replay='HISTORICAL_PASS_ONLY; no new replay',methodology_limitation='Original cycle committed all seven tests before one implementation commit; does not meet installed TDD one-test-at-a-time cadence. Genuine public-CLI RED/GREEN and frozen hashes retained; no rewrite or R1 acceptance credit.'),r1_retained_logs=logs,old_ready=dict(preserved_path=str(old_ready),original_untracked_path=str(original),byte_identical=True,sha256=sha(old_ready),artifact_hash_checks=old_checks),historical_74_pass=dict(source=retained_status['source'],verdict=retained_status['verdict'],exit_code=retained_status['exit_code'],r1_coverage=False),provenance_sha256={str(p):sha(p) for p in [R1/'READY.json',R1/'report.md',R1.parent/'verifier/READY.json',R1.parent/'verifier/report.md',old_ready]})
(OUT/'retained-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('OK original frozen cycle/receipt; retained local GREEN logs; original external READY and hashes; historical verification correctly bound')
