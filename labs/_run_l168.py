"""Complete selected released-checkpoint evaluation; no checkpoint search or fitting."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1')
import hashlib,json,sys,time,random
from pathlib import Path
import numpy as np
import torch

def run168(root,source,out,seeds):
    from importlib.metadata import version
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'input-manifest.json').read_text())
    for name,entry in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==entry['sha256'],name
    sys.path.insert(0,str(Path(source)/'model_pretrain'))
    from src.models import ModelConfig,build_model,load_checkpoint,build_classifier
    from src.eval_utils import fill_nans,predict_proba_in_chunks
    from sklearn.metrics import roc_auc_score
    from tabicl import TabICLClassifier
    assert version('tabicl')=='0.1.3'
    assert manifest["dataset"]=="rel-trial-dfs-2" and manifest["test_rows"]==825
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=np.load(root/'prepared.npz');records=[]
    for arm in ['RDBPFN','RDBPFN_single','TabICLv1.1']:
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
        start=time.perf_counter()
        if arm!='TabICLv1.1':
            cfg=ModelConfig(num_layers=6);model=build_model(cfg)
            load_checkpoint(model,root/(arm+'.pt'),str(device));model.to(device).eval()
        load_seconds=time.perf_counter()-start
        for seed in seeds:
            path=out/f'{arm}-{seed}.npz'
            assert not path.exists(),'Immutable attempt already exists'
            start=time.perf_counter();idx=data['support'][seed]
            X,Q=fill_nans(data['X_train'][idx],data['X_test']);y=data['y_train'][idx]
            clf=(build_classifier(model,device,cfg) if arm!='TabICLv1.1' else
                 TabICLClassifier(model_path=str(root/'tabicl-classifier-v1.1-0506.ckpt'),allow_auto_download=False,device=str(device)))
            assert arm!='TabICLv1.1' or clf.n_estimators==32
            if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
            clf.fit(X,y);prob=predict_proba_in_chunks(clf,Q,2000)
            assert prob.shape==(manifest["test_rows"],2) and np.isfinite(prob).all()
            np.testing.assert_allclose(prob.sum(1),1,atol=1e-5)
            np.savez_compressed(path,keys=data['test_keys'],label=data['y_test'],probability=prob[:,1],support_keys=data['train_keys'][idx])
            record=dict(arm=arm,seed=int(seed),rows=len(Q),support=len(idx),auc=float(roc_auc_score(data['y_test'],prob[:,1])),seconds=time.perf_counter()-start,
                        load_seconds=load_seconds,peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            records.append(record);(out/f'{arm}-{seed}.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
        if arm!='TabICLv1.1':del model
        del clf
        if torch.cuda.is_available():torch.cuda.empty_cache()
    receipt=dict(records=records,device=str(device),packages={n:version(n) for n in ['torch','numpy','pandas','scikit-learn','pydantic','tabicl']},input_manifest_sha256=hashlib.sha256((root/'input-manifest.json').read_bytes()).hexdigest())
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='/tmp/l168-input');parser.add_argument('--source',default=str(Path(__file__).parent/'sources/l166/upstream'));parser.add_argument('--out',required=True);parser.add_argument('--seeds',default='0,1,2,3,4,5,6,7,8,9');a=parser.parse_args()
    run168(a.input,a.source,a.out,[int(v) for v in a.seeds.split(',')])
