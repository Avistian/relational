"""Complete replay plus corruption and learner-function mutation checks."""
import json,shutil,tempfile
from pathlib import Path
from _audit_b05 import audit_course
from _source_b05 import source_gate
from _test_b05 import run_tests,check_feature_view,check_neighbors,check_overlap

def verify(root):
    root=Path(root);e=root/'evidence/b05';source=source_gate(root/'sources/b05');course=audit_course(e)
    assert source==json.loads((e/'source-gate.json').read_text())
    assert course==json.loads((e/'course-audit.json').read_text())
    run_tests();mutations=0
    for kind in ['missing','duplicate','key','neighbor','label','prediction','nan','target']:
      with tempfile.TemporaryDirectory() as td:
        dest=Path(td);shutil.copytree(e,dest,dirs_exist_ok=True);p=dest/'episodes.json';r=json.loads(p.read_text())
        if kind=='missing':r.pop()
        if kind=='duplicate':r.append(r[0])
        if kind=='key':r[0]['query_ids'][0]=r[0]['support_ids'][0]
        if kind=='neighbor':r[0]['neighbor_ids'][0][0]=999
        if kind=='label':r[0]['query_y'][0]+=1
        if kind=='prediction':r[0]['prediction'][0]+=1
        if kind=='nan':r[0]['prediction'][0]=float('nan')
        if kind=='target':r[0]['target_intervention_ids'].reverse()
        p.write_text(json.dumps(r))
        try:audit_course(dest)
        except (AssertionError,ValueError,IndexError):mutations+=1
        else:raise AssertionError('Accepted corrupt '+kind)
    wrong=[(check_feature_view,lambda t,c:(t.copy(),t[:,c].copy())),(check_neighbors,lambda s,q,ids,k:__import__('numpy').tile(ids[:k],(len(q),1))),(check_overlap,lambda a,b:'INDEPENDENT')]
    for check,fn in wrong:
        try:check(fn)
        except (AssertionError,ValueError):pass
        else:raise AssertionError('Wrong learner function passed')
    return dict(source=source,course=course,checks=dict(status='PASS',evidence_corruptions_rejected=mutations,learner_mutations_rejected=3))
if __name__=='__main__':
    root=Path(__file__).resolve().parent;r=verify(root);(root/'_verify_b05_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r['checks'])
