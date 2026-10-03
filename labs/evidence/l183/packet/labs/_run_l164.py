"""Immutable release replay and bounded resource probe; no reduced paper preset."""
import os,sys,time,json,hashlib,copy
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
TASK='rel-f1-driver-dnf'

def setup(root):
    root=Path(root).resolve();os.chdir(root)
    sys.path.insert(0,str(P/'sources/l164/upstream'));torch.set_num_threads(1)
    torch.manual_seed(42);np.random.seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed(42)
    return root

def to_device(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,(list,tuple)):return type(value)(to_device(v,device) for v in value)
    return value

def pilot(root,out):
    root=setup(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    from hdataset import Graph,Task
    from hloaderwrapper import LoaderWrapperTask,buildindice_downsample_absolute,buildindice
    from hFloatEmb import SimpleRepeater
    from hmodel import GriffinMod
    from safetensors.torch import load_file
    from sklearn.metrics import roc_auc_score
    device='cuda' if torch.cuda.is_available() else 'cpu'
    graph=Graph(str(root/'data'));task=Task(str(root/'data'))
    model=GriffinMod(hiddim=512,num_mp=4,use_rev=True,use_gate=False).to(device)
    model.load_state_dict(load_file(str(root/'checkpoint/model.safetensors')),strict=True)
    optim=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=2e-4)
    floatenc=SimpleRepeater(512)
    def dataset(split):return LoaderWrapperTask(graph,256,split=='train',dict(floatemb=floatenc,fanout=20,hop=2),task,[TASK],split,3)
    train=dataset('train');valid=dataset('valid')
    train.ind=buildindice_downsample_absolute(True,train.lens,256,512,42)
    valid.ind=buildindice(False,valid.lens,256)
    events=[]
    # Exactly two full256 training batches plus ALL566 validation queries.
    # Zero loader workers isolates sample/compute timing and avoids16worker-prefetch
    # memory in the resource probe. Full release replay retains original workers.
    for phase,ds in [('train',train),('valid',valid)]:
        model.train(phase=='train')
        for i in range(len(ds)):
            start=time.perf_counter();batch=ds[i];sample=time.perf_counter()-start
            shapes=[list(v.shape) for _,v in batch[0]]
            b=to_device(batch,device);del batch
            if device=='cuda':torch.cuda.synchronize()
            begin=time.perf_counter()
            with torch.set_grad_enabled(phase=='train'):
                logits=model(*b[:-3])[b[-1]]@b[-2].T
                loss=torch.nn.functional.cross_entropy(logits,b[-3])
                if phase=='train':
                    optim.zero_grad();loss.backward()
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'Nonfinite gradients'
                    optim.step()
            if device=='cuda':torch.cuda.synchronize()
            ev=dict(phase=phase,batch=i,rows=len(b[-3]),sample_seconds=sample,compute_seconds=time.perf_counter()-begin,shapes=shapes,loss=float(loss.detach()),peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
            events.append(ev);(out/'pilot.json').write_text(json.dumps(dict(status='RUNNING',events=events),indent=2)+'\n');print(ev,flush=True)
            del b,logits,loss
    r=dict(status='COMPLETE_TIMING_ONLY',device=device,events=events,training_queries=512,validation_queries=566,test='NOT_RUN',paper_result='NOT_RUN',loader_workers=0,release_loader_workers=16)
    (out/'pilot.json').write_text(json.dumps(r,indent=2)+'\n');return r

def run_source(root,out,arm,size,seed):
    root=setup(root);out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    assert arm in ['no-pretrain','others-2'];assert size in [512,4096];assert seed in range(42,47)
    import argparse
    import hmaintask_downsample_absolute_eval_sample as release
    from sklearn.metrics import roc_auc_score
    # Source references model.device although GriffinMod has no such property.
    # This compatibility fix changes no tensor computation or training settings.
    release.GriffinMod.device=property(lambda self:next(self.parameters()).device)
    original=release.compute_metric;evaluations=[];active={}
    original_eval=release.eval_task
    def scored(outputs,labels,metric):
        score=original(outputs,labels,metric)
        probability=outputs.softmax(-1)[:,1].numpy();truth=labels.numpy()
        oracle=float(roc_auc_score(truth,probability));assert abs(oracle-score)<1e-6
        ds=active['dataset'];split=ds.split
        task=ds.task;offset={'train':0,'valid':11411,'test':11977}[split]
        source=task.tasks[TASK][slice(offset,offset+len(truth))]
        assert np.array_equal(source['label'].numpy(),truth)
        rows=[dict(row_index=offset+i,nodeidx=int(source['nodeidx'][i]),cutoff=int(source['timestamp'][i]),label=int(y),probability=float(p)) for i,(y,p) in enumerate(zip(truth,probability))]
        filename=f'{len(evaluations):04d}-{split}-predictions.json'
        (out/filename).write_text(json.dumps(rows)+'\n')
        evaluations.append(dict(split=split,source_auroc=score,independent_auroc=oracle,file=filename,sha256=hashlib.sha256((out/filename).read_bytes()).hexdigest()))
        (out/'evaluations.json').write_text(json.dumps(evaluations,indent=2)+'\n')
        return score
    def evaluated(model,dec,dataset,*args,**kwargs):
        active['dataset']=dataset;return original_eval(model,dec,dataset,*args,**kwargs)
    release.compute_metric=scored;release.eval_task=evaluated
    args=argparse.Namespace(dataset=str(root/'data'),logdir=str(out/'logs'),logname='griffin',tasks=[TASK],savepath=str(out/'checkpoints'),loadpath=str(root/'checkpoint') if arm=='others-2' else None,mode='train',seed=42,batchsize=256,eval_batchsize=256,lr=3e-4,wd=2e-4,maxepoch=200,patience=10,eval_per_epoch=2,downsample_num=size,downsample_seed=seed,eval_sample_ratio=1.,eval_sample_seed=seed,num_mp=4,hiddim=512,fanout=20,fewshotfanout=3,hop=2,use_rev=True,use_gate=False)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    release.main(args)
    assert evaluations[-1]['split']=='test'
    result=dict(status='COMPLETE_RELEASE_REPLAY',arm=arm,size=size,split_seed=seed,model_seed=42,final=evaluations[-1],historical_identity='NOT_ESTABLISHED')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--root',default='/tmp/l164-release');a.add_argument('--out',default='/tmp/l164-run');a.add_argument('--phase',choices=['pilot','fit'],default='pilot');a.add_argument('--arm',choices=['no-pretrain','others-2'],default='others-2');a.add_argument('--size',type=int,default=512);a.add_argument('--seed',type=int,default=42)
    v=a.parse_args();print(pilot(v.root,v.out) if v.phase=='pilot' else run_source(v.root,v.out,v.arm,v.size,v.seed))
