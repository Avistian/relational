"""Execute ONE frozen course configuration; checkpoints are external, no downloads."""
import hashlib,json,os,resource,sys,time,platform
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/b04'
sys.path.insert(0,str(P/'sources/b04/upstream/src'))
import numpy as np
import torch,sklearn
from tabicl import TabICLClassifier
from relkit.scalable_b04 import fit_missingness,transform_missingness

def predict_without_query_labels(xs,ys,xq,checkpoint,settings,batch):
    model=TabICLClassifier(model_path=checkpoint,**settings)
    t=time.perf_counter();model.fit(xs,ys);fit_seconds=time.perf_counter()-t
    t=time.perf_counter();prob=np.concatenate([model.predict_proba(xq[i:i+batch]) for i in range(0,len(xq),batch)])
    return prob,model.classes_.tolist(),fit_seconds,time.perf_counter()-t,model.model_config_

if __name__=='__main__':
    name,checkpoint=sys.argv[1:];protocol=json.loads((E/'diagnostic-protocol.json').read_text())
    config=next(x for x in protocol['configs'] if x['name']==name);dest=E/(name+'.json')
    if dest.exists():raise SystemExit('Immutable result already exists')
    assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()==protocol['checkpoint']['sha256']
    b=(E/'inputs.json').read_bytes();assert hashlib.sha256(b).hexdigest()==protocol['inputs_sha256']
    # Verify every actual imported upstream Python byte against the archived source.
    import tarfile
    with tarfile.open(P/'sources/b04/upstream.tar.gz') as t:
        for m in t.getmembers():
            parts=Path(m.name).parts[1:]
            if m.isfile() and str(Path(*parts)).startswith('src/'):
                assert (P/'sources/b04/upstream'/Path(*parts)).read_bytes()==t.extractfile(m).read()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0)
    raw=json.loads(b);X=np.array(raw['X']);y=np.array(raw['y']);ids=raw['support_order'][:config['support_n']];queries=raw['query_ids']
    if config['missing']:X[np.array(raw['missing_mask'])]=np.nan
    xs,xq=X[ids],X[queries];means=fit_missingness(xs)
    xs=transform_missingness(xs,means,config['indicators']);xq=transform_missingness(xq,means,config['indicators'])
    prob,classes,fit,predict,model_config=predict_without_query_labels(xs,y[ids],xq,checkpoint,protocol['settings'],config['query_batch'])
    out=dict(config=config,protocol_sha256=hashlib.sha256((E/'diagnostic-protocol.json').read_bytes()).hexdigest(),inputs_sha256=protocol['inputs_sha256'],checkpoint_sha256=protocol['checkpoint']['sha256'],support_ids=ids,query_ids=queries,classes=classes,probabilities=prob.tolist(),means=means.tolist(),transformed_features=xs.shape[1],fit_seconds=fit,predict_seconds=predict,peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_config=model_config,environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__,platform=platform.platform(),threads=1,cuda=False),status='COMPLETE_COURSE_DIAGNOSTIC')
    dest.write_text(json.dumps(out,indent=2)+'\n');print(name,fit,predict,out['peak_process_rss_mib'],flush=True)
