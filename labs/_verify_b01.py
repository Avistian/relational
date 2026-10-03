"""Contract adversaries and frozen-source integrity; original L200 is read-only."""
import json,shutil,tempfile
from pathlib import Path
from _test_b01 import checks
from _audit_b01 import replay
from relkit.comparison_b01 import compare_contracts,paired_effect,claim_gate

def verify(root):
    root=Path(root);assert checks(compare_contracts,paired_effect,claim_gate)=='PASS'
    wrong=[(lambda *a:dict(status='MATCHED',mismatches=[],unknown=[]),paired_effect,claim_gate),
        (compare_contracts,lambda *a:dict(draws=list(range(10)),mean=.1,positive=10),claim_gate),
        (compare_contracts,paired_effect,lambda *a:'SUPPORTED_DESCRIPTIVE_REPLAY')]
    for funcs in wrong:
        try:checks(*funcs)
        except (AssertionError,ValueError):pass
        else:raise AssertionError('Wrong learner function passed')
    report=replay(root);assert report==json.loads((root/'report.json').read_text())
    for name in ['l200-reproducer.zip','inherited-l200-protocol.md']:
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'evidence';shutil.copytree(root,q)
            p=q/name;p.write_bytes(p.read_bytes()+b'altered')
            try:replay(q)
            except ValueError:pass
            else:raise AssertionError('Changed frozen input accepted')
    return dict(status='PASS',report_parity='EXACT',wrong_learner_functions_rejected=3,frozen_inputs_corruptions_rejected=2,
        inherited_raw_evidence_corruptions_rejected=report['independent_verification']['corruptions_rejected'],
        independent_rank_auc_runs=report['independent_verification']['independent_rank_auc_runs'],
        independent_auc_maximum_error=report['independent_verification']['maximum_error'])

if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=verify(p/'evidence/b01')
    (p/'_verify_b01_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
