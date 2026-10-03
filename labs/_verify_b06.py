"""Independent coverage/scoring, real saved-checkpoint inference, corruption rejection."""
import copy,hashlib,json,shutil,tempfile
from pathlib import Path
import numpy as np
import torch
from relkit.prior_b06 import CellLearner,predict_tasks,state_hash
from _audit_b06 import audit_course
from _source_b06 import source_gate
from _test_b06 import check_prior,check_normalize,check_pair
from _model_test_b06 import check_model
from relkit.prior_b06 import choose_prior,support_normalize,paired_effect
P=Path(__file__).resolve().parent;E=P/'evidence/b06'

def verify():
    torch.set_num_threads(1);c=json.loads((E/'course-protocol.json').read_text())
    check_prior(choose_prior);check_normalize(support_normalize);check_pair(paired_effect);check_model(c)
    r=audit_course(E);assert r==json.loads((E/'course-audit.json').read_text())
    gate=source_gate(P/'sources/b06');assert gate==json.loads((E/'source-gate.json').read_text())
    tasks=json.loads((E/'runs/tasks.json').read_text());maximum=0
    for seed in c['seeds']:
        for arm in c['arms']:
            run=json.loads((E/f'runs/run-{arm}-{seed}.json').read_text())
            assert run['identity']['source_sha256']==hashlib.sha256((P/'relkit/prior_b06.py').read_bytes()).hexdigest()
            model=CellLearner(c);model.load_state_dict(torch.load(E/f'runs/weights-{arm}-{seed}.pt',weights_only=True,map_location='cpu'))
            assert state_hash(model)==run['training']['final_sha256']
            pred=predict_tasks(model,tasks,c)
            for a,b in zip(pred,run['records']):
                assert {k:a[k] for k in a if k!='p'}=={k:b[k] for k in b if k!='p'}
                maximum=max(maximum,max(abs(x-y) for x,y in zip(a['p'],b['p'])))
    assert maximum<=1e-7
    rejected=[]
    with tempfile.TemporaryDirectory(prefix='b06-corrupt-') as td:
        root=Path(td)/'evidence';shutil.copytree(E,root,ignore=shutil.ignore_patterns('reproducer.zip','artifact-manifest.json'))
        file=root/'runs/run-scm-0.json';original=file.read_bytes()
        mutations={
          'missing_row':lambda x:x['records'].pop(),
          'duplicate_identity':lambda x:x['records'].__setitem__(0,copy.deepcopy(x['records'][1])),
          'wrong_label':lambda x:x['records'][0].__setitem__('y',1-x['records'][0]['y']),
          'invalid_probability':lambda x:x['records'][0].__setitem__('p',[.7,.7]),
          'initialization':lambda x:x['training'].__setitem__('initial_sha256','broken'),
          'update_budget':lambda x:x['training']['loss'].pop(),
          'prior_budget':lambda x:x['training']['prior_counts'].__setitem__('scm',1),
          'input_hash':lambda x:x['identity'].__setitem__('task_sha256','broken')}
        for name,fn in mutations.items():
            x=json.loads(original);fn(x);file.write_text(json.dumps(x))
            try:audit_course(root)
            except AssertionError:rejected.append(name)
            else:raise AssertionError('Corruption accepted: '+name)
            file.write_bytes(original)
    wrong=[('prior',check_prior,lambda u,p:'scm'),('normalization',check_normalize,lambda s,q:(np.zeros_like(s),q)),('pairing',check_pair,lambda a,b:np.array(list(a.values()))-np.array(list(b.values())))]
    for name,check,fn in wrong:
        try:check(fn)
        except AssertionError:rejected.append('wrong_'+name)
        else:raise AssertionError('Wrong implementation passed: '+name)
    return dict(status='PASS',saved_checkpoint_predictions=4320,max_probability_error=maximum,corruptions_rejected=rejected,source_files=gate['source_files'],course_fits=r['fits'],paper=gate['status'],learner='PENDING_WRITTEN_DEFENSE')
if __name__=='__main__':
    r=verify();(P/'_verify_b06_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
