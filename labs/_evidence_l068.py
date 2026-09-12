"""Reconstruct temporal evidence from original data/source, independently of lesson code."""
import argparse,ast,contextlib,hashlib,io,json,os,sys,types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from _sources_l068_v2 import ensure_source
ROOT=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def original_environment():
    source=ensure_source()
    p=types.ModuleType('tabpfn');p.__path__=[str(source/'tabpfn')];sys.modules['tabpfn']=p
    from tabpfn.scripts.model_builder import load_model
    module=types.ModuleType('tabpfn.datasets');module.__path__=[]
    class Dataset:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
    module.DistributionShiftDataset=Dataset;sys.modules['tabpfn.datasets']=module
    ns=dict(__name__='tabpfn.datasets.dist_shift_datasets',__package__='tabpfn.datasets',np=np,pd=pd,torch=torch,os=os,MODULE_DIR=str(ROOT/'data/cache/l068-release/tabpfn/datasets'),TASK_TYPE_MULTICLASS='multiclass')
    names=['dataframe_to_distribution_shift_ds','get_electricity_data','get_parking_birmingham_data','get_chess_data','get_intersecting_blobs']
    path=source/'tabpfn/datasets/dist_shift_datasets.py'
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return load_model,ns

def split_rows(c,count,seed,cap):
    rng=np.random.default_rng(seed);parts={'train':[],'id':[],'ood':[]}
    for index,domain in enumerate(np.unique(c)):
        ids=rng.permutation(np.flatnonzero(c==domain))
        if cap is not None:ids=ids[:cap]
        if index<count:
            n=max(1,int(len(ids)*.1));parts['id'].extend(ids[:n]);parts['train'].extend(ids[n:])
        else:parts['ood'].extend(ids)
    return {k:np.array(v,dtype=int) for k,v in parts.items()}

def metric(y,p):
    y=np.asarray(y);p=np.asarray(p);k=p.shape[1]
    loss=float(-np.log(np.clip(p[np.arange(len(y)),y],np.finfo(np.float32).eps,1-np.finfo(np.float32).eps)).mean())
    auc=None
    if len(np.unique(y))==k:
        values=[]
        for cl in ([1] if k==2 else range(k)):
            positive=y==cl;n=int(positive.sum());m=len(y)-n
            ranks=rankdata(p[:,cl]);values.append(float((ranks[positive].sum()-n*(n+1)/2)/(n*m)))
        auc=float(np.mean(values))
    return dict(accuracy=float(np.mean(p.argmax(1)==y)),log_loss=loss,auc=auc)

def check(path,output):
    torch.set_num_threads(1);r=json.loads(Path(path).read_text());cfg=r['config']
    assert r['protocol']['source_fractions']==[.4,.55,.7] and r['protocol']['temperature']==1 and r['protocol']['views']==1
    expected={(d,s,a,k) for d in cfg['datasets'] for s in cfg['seeds'] for a in ['base_no_time','base_time','drift','drift_zero','noT2V'] for k in ['id','ood']}
    actual=[(v['dataset'],v['seed'],v['arm'],v['split']) for v in r['records']]
    assert len(actual)==len(set(actual)) and set(actual)==expected
    load_model,ns=original_environment();datasets={};models={};checks=[]
    for name,fn in [('electricity','get_electricity_data'),('parking','get_parking_birmingham_data'),('chess','get_chess_data'),('blobs','get_intersecting_blobs')]:
        if name not in cfg['datasets']:continue
        data=ns[fn]();x=data.x.float().numpy();y=data.y.numpy().astype(int);c=data.dist_shift_domain.numpy().astype(int)
        datasets[name]=(x,y,c)
        for key,array in [('x',x),('y',y),('c',c)]:assert hashlib.sha256(array.tobytes()).hexdigest()==r['datasets'][name][key+'_sha256'],(name,key,array.dtype)
    for row in r['records']:
        x,y,c=datasets[row['dataset']];j=cfg['seeds'].index(row['seed']);count=max(2,int(len(np.unique(c))*[.4,.55,.7][j%3]))
        split=split_rows(c,count,row['seed'],cfg['cap_per_domain']);context=split['train'];query=split[row['split']]
        assert row['context_ids']==context.tolist() and row['query_ids']==query.tolist() and row['source_count']==count
        assert row['targets']==y[query].tolist() and row['query_domains']==c[query].tolist()
        assert row['feature_seed']==17+row['seed'];arm=row['arm']
        family='base' if arm.startswith('base') else 'dist_ablation_no_t2v' if arm=='noT2V' else 'dist'
        ck=1 if arm=='noT2V' else cfg['checkpoints'][j%len(cfg['checkpoints'])];assert row['checkpoint']==ck
        key=(family,ck)
        if key not in models:
            checkpoint=ROOT/f'data/cache/l068-release/tabpfn/model_cache/tabpfn_{family}_model_{ck}.cpkt'
            with contextlib.redirect_stdout(io.StringIO()):loaded,_=load_model(str(checkpoint),'cpu',verbose=False)
            models[key]=loaded[2].eval()
            label='noT2V' if family=='dist_ablation_no_t2v' else family
            assert digest(checkpoint)==r['models'][str((label,ck))]['sha256']
        model=models[key];model.generator_device=torch.device('cpu');model.generator.manual_seed(row['feature_seed'])
        ids=np.r_[context,query];raw=x[ids].copy();time=c[ids].astype(np.float32)
        if arm=='base_time':raw=np.column_stack([raw,time])
        if arm=='drift_zero':time=np.zeros_like(time)
        inputs={'main':torch.from_numpy(raw)[:,None]}
        if family!='base':inputs['dist_shift_domain']=torch.from_numpy(time)[:,None,None]
        with torch.no_grad():
            logits=model((inputs,torch.tensor(y[context],dtype=torch.float32)[:,None]),single_eval_pos=len(context))[:,0,:len(np.unique(y[context]))]
            p=logits.softmax(-1).numpy()
        saved=np.array(row['probabilities']);delta=float(np.max(np.abs(p-saved)))
        assert delta<2e-4,(row['dataset'],arm,row['split'],delta)
        values=metric(y[query],saved)
        for k,v in values.items():
            if v is None:assert row[k] is None
            else:assert abs(v-row[k])<2e-6,(k,v,row[k])
        for domain_row in row['domain_metrics']:
            domain=domain_row['domain'];mask=c[query]==domain;assert domain_row['rows']==sum(mask)
            for k,v in metric(y[query][mask],saved[mask]).items():
                if v is None:assert domain_row[k] is None
                else:assert abs(v-domain_row[k])<2e-6
        checks.append(dict(dataset=row['dataset'],seed=row['seed'],arm=arm,split=row['split'],predictions=len(query),max_probability_error=delta))
        print(checks[-1],flush=True)
    # Recompute each reported within-dataset summary from individual saved records.
    for item in r['summary']:
        rows=[v for v in r['records'] if all(v[k]==item[k] for k in ['dataset','arm','split'])]
        assert item['n']==len(rows)
        for key in ['accuracy','auc','log_loss']:
            values=[v[key] for v in rows if v[key] is not None];assert item[key+'_n']==len(values)
            mean=np.mean(values) if values else None;sd=np.std(values,ddof=1) if len(values)>1 else 0.
            if mean is None:assert item[key+'_mean'] is None
            else:assert abs(mean-item[key+'_mean'])<1e-12
            assert abs(sd-item[key+'_sd'])<1e-12
    report=dict(status='PASS',evidence_sha256=digest(path),checker_sha256=digest(__file__),scope='Original released dataset functions, every raw row/label/domain hash, deterministic temporal partitions, complete declared record roster, original checkpoint forwards, saved probability metrics and per-domain/within-dataset summaries. Separate SCM reconstruction is not original prior distribution.',records=checks,predictions=sum(v['predictions'] for v in checks),max_probability_error=max(v['max_probability_error'] for v in checks),torch=torch.__version__)
    Path(output).write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',default=str(ROOT/'_verify_l068_v2_results.json'));p.add_argument('--output',default=str(ROOT.parent/'reviews/lesson-quality-audit-047-070/068-evidence.json'));a=p.parse_args();check(a.evidence,a.output)
