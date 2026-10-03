"""Independent source gate and diagnostic evidence audit; standard library only."""
import hashlib,itertools,json,math,zipfile
from pathlib import Path

ARCHIVES={'TabPFN-main.zip':'936f4a9433bb89ec660dd0343ed5dca61c842ae9c60e271dc82c87a3a99a2db5','TabPFN-original-stale.zip':'0a995999b235d47eba66488c42cd565d7646008589ba1693af94f167dd4cd80a'}

def audit_sources(root):
    root=Path(root);names={}
    for name,digest in ARCHIVES.items():
        b=(root/name).read_bytes()
        if hashlib.sha256(b).hexdigest()!=digest:raise ValueError('Archive identity changed: '+name)
        with zipfile.ZipFile(root/name) as z:
            names[name]=z.namelist()
            if name=='TabPFN-original-stale.zip':
                prefix='TabPFN-Release-main/'
                source=z.read(prefix+'tabpfn/scripts/tabular_evaluation.py').decode()
                assert 'from tabpfn.datasets import' in source
                assert not any(n.startswith(prefix+'tabpfn/datasets') for n in z.namelist())
                assert 'N_SPLITS = 5' in source
                helper=z.read(prefix+'tabpfn/scripts/tabular_evaluation_notebook_utils.py').decode()
                assert 'range(0, 10)' in helper
    return dict(status='INCOMPLETE_SOURCE_PROTOCOL',experiment='B03-TABPFNV2-BLOOD-OFFICIAL-SPLITS',archives=ARCHIVES,missing_loader='tabpfn.datasets',benchmark_runs=0,
        gaps=['Paper-to-OpenML task and ten ordered split identities not authenticated','Archived evaluation imports an absent dataset loader','Historical experiment configuration and checkpoint linkage not authenticated','Per-split reference predictions or exact numeric target not recovered'],
        observations=['Archive helper defaults disagree: five versus ten splits','Multiple OpenML tasks use dataset 1464; same dataset is not same split'],
        fresh_benchmark='NOT_RUN',pretraining='NOT_RUN',full_benchmark='NOT_RUN',cloud_usd=0)

def audit_predictions(record):
    expected={'data':'d88c5d9a92842376c4d766aca0aad26d5990c781a168200443aecbcee2cb4228','labels':'76bbcf80f5bd04bba239fb42dd5b607981b67533a20aac705d2b945eee06df3b','train_ids':'b076716bdb9b38e672f31844b9f2d688fbfe08951077c33240a67f6aadb0f167','test_ids':'fbb83398fe16cbc895e927025afed751b2ddb95be8add1ca0d028456a9488287'}
    for key,digest in expected.items():
        assert hashlib.sha256(json.dumps(record[key],separators=(',',':')).encode()).hexdigest()==digest,key
    assert record['checkpoint_sha256']=='f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f'
    assert record['n_estimators']==4 and record['split_seed']==record['model_seed']==0
    assert record['precision']=='CPU float32' and record['tolerance']==dict(atol=1e-6,rtol=0)
    assert record['versions']['tabpfn']=='2.0.9' and record['versions']['sklearn']=='1.6.1'
    rows=record['records'];assert [tuple(x['permutation']) for x in rows]==list(itertools.permutations(range(3)))
    base=rows[0]['probabilities'];n=len(record['test_ids']);targets=[record['labels'][i] for i in record['test_ids']]
    output=[]
    for row in rows:
        p=row['probabilities'];pi=row['permutation'];assert len(p)==n
        for v in p:
            assert len(v)==3 and all(math.isfinite(a) and 0<=a<=1 for a in v)
            assert abs(math.fsum(v)-1)<1e-6
        aligned=[[v[pi[c]] for c in range(3)] for v in p]
        delta=max(abs(aligned[i][c]-base[i][c]) for i in range(n) for c in range(3))
        nll=-math.fsum(math.log(max(aligned[i][targets[i]],1e-15)) for i in range(n))/n
        accuracy=sum(max(range(3),key=lambda c:aligned[i][c])==targets[i] for i in range(n))/n
        output.append(dict(permutation=pi,max_abs_delta=delta,within_tolerance=delta<=1e-6,accuracy=accuracy,log_loss=nll))
    repeat=rows[0]['repeat_probabilities'];assert len(repeat)==n and all(len(v)==3 for v in repeat)
    assert all(math.isfinite(a) for v in repeat for a in v)
    repeat_delta=max(abs(repeat[i][c]-base[i][c]) for i in range(n) for c in range(3))
    assert repeat_delta<=1e-6
    return dict(status='PASS',evidence='COMPLETE_COURSE_DIAGNOSTIC',permutations=6,prediction_rows=6*n,rows=output,max_abs_delta=max(r['max_abs_delta'] for r in output),repeat_max_abs_delta=repeat_delta,
        conclusion='Observed permutation sensitivity' if any(not r['within_tolerance'] for r in output) else 'Within tolerance on this fixture only',
        benchmark='NOT_RUN',equivariant_model_training='NOT_RUN',query_feature_independence='NOT_ESTABLISHED')
