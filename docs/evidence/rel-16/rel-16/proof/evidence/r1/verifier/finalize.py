"""Validate completed verifier evidence, then write READY.json last."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path('/tmp/lead-wt/night-vrel16-r1')
OUT=Path(__file__).resolve().parent
SHA='86fe6e7284421f128686da00240aeddb5595cbab'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name): return json.loads((OUT/name).read_text())
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
porcelain=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
assert head==SHA and porcelain==''
focused=read('focused.json'); parity=read('parity.json'); static=read('static-history.json'); retained=read('retained-audit.json')
assert focused['exit_code']==0 and focused['source']==dict(head=SHA,dirty=False)
assert 'Ran 10 tests' in (OUT/'focused.stderr.txt').read_text()
assert parity['source']==dict(head=SHA,dirty=False) and parity['equal'] and parity['actual_count']==parity['generated_count']==18497
assert all(static['old_ready']['artifact_hash_checks'].values())
for path,expected in retained['provenance_sha256'].items():
    assert sha(Path(path))==expected,path
assert not (OUT/'READY.json').exists(), 'Preserve existing READY; never overwrite'
ready=dict(task='rel-16',revision='r1',role='verifier',actor='separate read-only verifier /tmp/lead-wt/night-vrel16-r1',status='REVIEW_COMPLETE_SOURCE_PASS',done=False,sha=SHA,source=dict(head=SHA,dirty=False),RISK_FLAG='YES',source_code_verdict='PASS',findings=[],findings_resolved=['REL16-V1','REL16-V2'],focused_fixtures=dict(verdict='PASS',tests=10,exit_code=0,source=focused['source'],evidence=str(OUT/'focused.json')),current_file_set=dict(verdict='PASS',generated_count=18497,actual_count=18497,equal=True,retained_inventory_equal=True,evidence=str(OUT/'parity.json')),blast_radius=dict(verdict='PASS_BOUNDED_REPAIR',safety_fact_level=4,reviewer='same separate verifier, no additional role spawned',evidence=str(OUT/'blast-probes.json')),tdd=dict(r1_cycles='RETAINED_LOCAL_RED_GREEN_VERIFIED_STATICALLY',original_frozen_tests='UNCHANGED',historical_artifacts='HASHES_MATCH',invalid_fixture_history='PRESERVED: 10 failures included 9 parser assertions plus 1 invalid GNU rm public/. expectation; corrected separately before root fix, then 12 genuine failures',intermediate_green_binding='preceding RED HEAD with dirty production fix; not clean GREEN SHA',original_methodology_limitation=retained['original_cycle']['methodology_limitation'],canonical_replay='NOT_RUN for R1; historical PASS at 1ba8e647 retained only',evidence=[str(OUT/'static-history.json'),str(OUT/'retained-audit.json')]),verify_project=dict(verdict='NOT_RUN',exit_code=None,source=dict(head=SHA,dirty=False),reason='Configured discovery includes model.fit, prohibited by user. No lead-verify execution/config change; focused tests do not replace configured verification.',historical_74_pass_applies_to_r1=False,evidence=str(OUT/'verify-project-not-run.json')),packaging=dict(verdict='PROVENANCE_CORRECTED_EXTERNAL_HANDOFF',committed_readme=True,committed_original_ready=False,committed_original_report_md=False,original_ready=retained['old_ready'],r1_evidence='EXTERNAL_NOT_COMMITTED',verifier_evidence='EXTERNAL_NOT_COMMITTED',remaining_blocker='Any committed READY/report package requirement remains unfulfilled and parent-owned; no artifact invented.'),acceptance='NOT_DONE',blockers=dict(configured_verification='NOT_RUN: model fitting prohibited',packaging='Historical committed READY/report absent; external receipts only; parent must disposition any committed-package requirement',acceptance='Parent-owned; source PASS is not software-verified or accepted'),hosted_ci='NOT_RUN',push='NOT_RUN',merge='NOT_RUN',canonical_replay='NOT_RUN',preserved_verifier_harness_failure=dict(evidence=str(OUT/'parity-attempt-1.json'),stage='before Bash launch',resolution='Same script supplied through stdin once; no source/profile/gate change'),skills=['/home/avist/agent-stack/lead/.agents/skills/verify-project/SKILL.md','/home/avist/agent-stack/lead/.agents/skills/blast-radius/SKILL.md'],tdd_skill_read_for_audit_only='/home/avist/agent-stack/lead/.agents/skills/tdd/SKILL.md',report=str(OUT/'report.md'),created_utc=datetime.now(timezone.utc).isoformat(),source_sha256={p:sha(ROOT/p) for p in static['changed_files']},preserved_prior_receipts_sha256=retained['provenance_sha256'],artifacts_sha256={str(p):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='READY.json'})
# This must be the last write: all logs, scripts, reports and static checks exist.
(OUT/'READY.json').write_text(json.dumps(ready,indent=2)+'\n')
print('LEAD_RESULT=REVIEW_COMPLETE_SOURCE_PASS')
print('OK exact SHA '+SHA+' dirty:false; Pages tests 10; publication files 18497')
print('NOT_RUN configured verification; NOT_DONE acceptance')
print('READY='+str(OUT/'READY.json'))
