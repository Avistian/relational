"""Frozen L176 nested-support grid; predictor operations inherited unchanged from L169."""
def run176(root,source,out,phase):
    import os
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1')
    import hashlib,json,random,sys,time
    from pathlib import Path
    import numpy as np
    import torch
    from importlib.metadata import version
    root=Path(root);source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'input-manifest.json').read_text())
    assert manifest['experiment']=='L176 Nested-Support ICL Evaluation'
    assert manifest['contexts']==[64,128,256,512,1024] and manifest['fresh_contexts']==manifest['contexts']
    assert manifest['seeds']==list(range(10))
    for name,item in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Input hash mismatch: '+name)
    for name,digest in manifest['source_files'].items():
        path=source/name.split('upstream/',1)[1]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Source hash mismatch: '+name)
    sys.path.insert(0,str(source/'model_pretrain'))
    from src.models import ModelConfig,build_model,load_checkpoint,build_classifier
    from src.eval_utils import fill_nans,predict_proba_in_chunks
    from sklearn.metrics import roc_auc_score
    from tabicl import TabICLClassifier
    if phase not in ['pilot','remaining','all']:raise ValueError('Unknown phase')
    assert version('tabicl')=='0.1.3'
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');records=[]
    for spec in manifest['experiments']:
        db=spec['database'];data=np.load(root/(db+'.npz'))
        for arm in manifest['arms']:
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
            start=time.perf_counter()
            if arm!='TabICLv1.1':
                cfg=ModelConfig(num_layers=6);model=build_model(cfg)
                load_checkpoint(model,root/(arm+'.pt'),str(device));model.to(device).eval()
            load_seconds=time.perf_counter()-start
            for k in manifest['fresh_contexts']:
                for seed in manifest['seeds']:
                    is_pilot=k==1024 and seed==0
                    if (phase=='pilot' and not is_pilot) or (phase=='remaining' and is_pilot):continue
                    name=f'{db}-{arm}-{k}-{seed}';path=out/(name+'.npz')
                    assert not path.exists(),'Immutable run already exists'
                    start=time.perf_counter();idx=data['support_'+str(k)][seed]
                    X,Q=fill_nans(data['X_train'][idx],data['X_test']);y=data['y_train'][idx]
                    clf=(build_classifier(model,device,cfg) if arm!='TabICLv1.1' else TabICLClassifier(model_path=str(root/'tabicl-classifier-v1.1-0506.ckpt'),allow_auto_download=False,device=str(device)))
                    assert arm!='TabICLv1.1' or clf.n_estimators==32
                    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
                    clf.fit(X,y);prob=predict_proba_in_chunks(clf,Q,2000)
                    assert prob.shape==(spec['test_rows'],2) and np.isfinite(prob).all()
                    np.testing.assert_allclose(prob.sum(1),1,atol=1e-5)
                    np.savez_compressed(path,keys=data['test_keys'],label=data['y_test'],probability=prob[:,1],support_keys=data['train_keys'][idx])
                    r=dict(database=db,arm=arm,context=k,seed=seed,rows=len(Q),auc=float(roc_auc_score(data['y_test'],prob[:,1])),seconds=time.perf_counter()-start,
                           load_seconds=load_seconds,peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),filename=path.name)
                    records.append(r);(out/(name+'.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
            if arm!='TabICLv1.1':del model
            del clf
            if torch.cuda.is_available():torch.cuda.empty_cache()
    expected=6 if phase=='pilot' else 294 if phase=='remaining' else 300
    assert len(records)==expected
    receipt=dict(records=records,phase=phase,device=str(device),packages={n:version(n) for n in ['torch','numpy','pandas','scikit-learn','pydantic','tabicl']},input_manifest_sha256=hashlib.sha256((root/'input-manifest.json').read_bytes()).hexdigest())
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--phase',choices=['pilot','remaining','all'],required=True);a=p.parse_args()
    run176(a.input,a.source,a.out,a.phase)
