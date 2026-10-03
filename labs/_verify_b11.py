"""Verify artifact consistency, archive rejection and approved scope boundaries."""
import hashlib,json,shutil,tempfile
from pathlib import Path
import nbformat
from _audit_b11 import audit
import _audit_b11 as audit_module
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b11'
def verify():
    m=json.loads((E/'mechanism.json').read_text());a=json.loads((E/'replay.json').read_text())
    assert len(m['cases'])==9 and m['seeds']==[0,1,2]
    assert max(r['output_error'] for r in m['cases'])<1e-9 and max(r['gradient_error'] for r in m['cases'])<1e-9
    assert a['verified_predictions']==13849 and a['checkpoint_selections']==11
    assert a['fresh_B11_benchmark_training']==a['matched_RelGNN_RelGT_benchmark']==a['whole_paper']=='NOT_RUN'
    n=nbformat.read(P/'solutions/b11-supervised-relational-baselines.ipynb',4)
    execution=json.loads((P/'_execution_b11_results.json').read_text());delivery=json.loads((P/'_delivery_b11_results.json').read_text())
    assert execution['status']==delivery['status']=='PASS' and execution['exact_author_report_parity']
    assert hashlib.sha256('\n'.join(c.source for c in n.cells if c.cell_type=='code').encode()).hexdigest()==execution['code_sha256']
    assert delivery['browser_states']==104
    # A single changed source byte or missing seed must fail before any scores are accepted.
    original=(audit_module.S,audit_module.A,audit_module.E);rejected=0
    with tempfile.TemporaryDirectory(prefix='b11-corruption-') as td:
        tmp=Path(td);shutil.copytree(P/'sources/b11',tmp/'sources');audit_module.S=tmp/'sources';audit_module.A=tmp/'sources/archive';audit_module.E=tmp/'out'
        target=audit_module.S/'relgt_original.py';data=target.read_bytes();target.write_bytes(data+b'# corruption\n')
        try:audit()
        except AssertionError:rejected+=1
        target.write_bytes(data)
        target=audit_module.A/'evidence/l143/seed-4/predictions.npz';target.unlink()
        try:audit()
        except FileNotFoundError:rejected+=1
    audit_module.S,audit_module.A,audit_module.E=original
    assert rejected==2
    result=dict(status='PASS',source_cases=9,verified_predictions=13849,notebook_report_parity=True,corrupt_archives_rejected=rejected,browser_states=104,paid_usd=0,whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
    (P/'_verify_b11_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':verify()
