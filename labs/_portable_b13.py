"""Check extracted source operators and reject plausible incorrect learner functions."""
import hashlib,json,subprocess,tempfile,zipfile
from pathlib import Path
import numpy as np
import _test_b13 as tests
P=Path(__file__).resolve().parent;R=P.parent
wrong={'canonical_schema':lambda edges,n=4:tuple(sorted(edges)),
       'parent_mean':lambda values,keys:np.full(len(keys[0]),np.mean(np.concatenate(values))),
       'fit_ridge':lambda x,y,alpha=1.:dict(mean=np.zeros(x.shape[1]),scale=np.ones(x.shape[1]),coef=np.ones(x.shape[1]),intercept=0.)}
rejected=0
for name,check in [('canonical_schema',tests.test_schema),('parent_mean',tests.test_parent_mean),('fit_ridge',tests.test_ridge)]:
    original=getattr(tests,name);setattr(tests,name,wrong[name])
    try:check()
    except AssertionError:rejected+=1
    else:raise AssertionError('Incorrect learner operation accepted: '+name)
    finally:setattr(tests,name,original)
with tempfile.TemporaryDirectory(prefix='b13-portable-') as td:
    root=Path(td)
    with zipfile.ZipFile(P/'evidence/b13/portable-sources.zip') as z:z.extractall(root)
    manifest=json.loads((root/'source-manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    result=subprocess.run([str(R/'.venv/bin/python'),'labs/_reproduce_b13.py','--audit'],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    target=json.loads((root/'labs/evidence/b13/plurel-full-target.json').read_text())
    assert len(target['slots'])==108 and all(x['status']=='NOT_RUN' for x in target['slots'])
    for operator in ['_fetch_l166.py','_run_l166.py']:
        result=subprocess.run([str(R/'.venv/bin/python'),'labs/'+operator,'--help'],cwd=root,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
    assert (root/'labs/sources/b13/plurel-paper/rustler/Cargo.toml').is_file()
    assert (root/'labs/sources/l166/upstream/model_pretrain/src/models.py').is_file()
out=dict(status='PASS',authenticated_source_files=len(manifest),extracted_full_target_slots=108,wrong_learner_functions_rejected=rejected,inherited_fresh_operator_cli='PASS',fresh_model_inference='NOT_RUN')
(P/'_portable_b13_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
