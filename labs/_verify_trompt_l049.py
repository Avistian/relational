"""Separate numeric Trompt execution probe; no comparison or paper-result claim."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,log_loss,accuracy_score
from relkit.claim_data import paper_data
from relkit.trompt_l049 import Trompt,train_trompt,predict_trompt
from relkit.saint_experiment import environment


def probe(name='pima',seeds=(0,1,2),epochs=20,d=16,prompts=8,layers=2,device="cpu",model_cls=Trompt,train_fn=train_trompt,predict_fn=predict_trompt):
    """Exercise the supplied visible model; fixed validation-only checkpoint rule."""
    from _live_identity_l051 import code_fingerprint
    # Record the supplied executable path, including live notebook primitives.
    dependencies=model_cls.__init__.__globals__
    cell_cls,head_cls=dependencies['TromptCell'],dependencies['TromptHead']
    functions={}
    for label,cls in [('model',model_cls),('cell',cell_cls),('head',head_cls)]:
        for method in ['__init__','forward']:functions[label+'.'+method]=getattr(cls,method)
    functions.update(train=train_fn,predict=predict_fn,
        loss=train_fn.__globals__['trompt_loss'],
        prompt_weights=cell_cls.forward.__globals__['prompt_weights'],
        prompt_reduce=cell_cls.forward.__globals__['prompt_reduce'])
    # If the notebook instruments calls, include the wrapped original operations too.
    for name_,fn in list(functions.items()):
        original=fn.__globals__.get('original_'+name_)
        if callable(original):functions[name_+'.wrapped']=original
    executable_identity={key:code_fingerprint(fn) for key,fn in functions.items()}
    torch.set_num_threads(1);start=time.time();data=paper_data(name);rows=[]
    for seed in seeds:
        torch.manual_seed(seed)
        model=model_cls(data['x'].shape[1],d=d,prompts=prompts,layers=layers)
        model,hist=train_fn(model,data['x'],data['y'],data['train'],data['valid'],seed=seed,epochs=epochs,device=device)
        prob=predict_fn(model,data['x'][data['test']]);y=data['y'][data['test']]
        rows.append(dict(seed=seed,auroc=float(roc_auc_score(y,prob[:,1])),accuracy=float(accuracy_score(y,prob.argmax(1))),
            log_loss=float(log_loss(y,prob)),history=hist,test_probabilities=prob.tolist()))
    return dict(scope='Trompt numeric implementation execution on an ExcelFormer author split; not a Trompt benchmark task/protocol',
        verdict='INCOMPARABLE',executable_identity={'schema':'semantic-code-v1','functions':executable_identity},config=dict(dataset=name,seeds=list(seeds),epochs=epochs,d=d,prompts=prompts,layers=layers,batch=64,lr=.001,patience=8,device=device),
        protocol=data['meta'],runs=rows,summary={m:dict(mean=float(np.mean([r[m] for r in rows])),sample_std=float(np.std([r[m] for r in rows],ddof=1)) if len(rows)>1 else None) for m in ['auroc','accuracy','log_loss']},
        elapsed_seconds=time.time()-start,environment=environment())

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capacity',action='store_true',help='Paper d=P=128,L=6 capacity on Pima, still INCOMPARABLE');parser.add_argument('--device',default='cpu');parser.add_argument('--out',default=str(Path(__file__).with_name('_verify_trompt_l049_results.json')));a=parser.parse_args()
    r=probe(d=128,prompts=128,layers=6,epochs=100,device=a.device) if a.capacity else probe(device=a.device)
    r['canonical_reference_sha256']=hashlib.sha256((Path(__file__).parent/'relkit/trompt_l049.py').read_bytes()).hexdigest()
    Path(a.out).write_text(json.dumps(r,indent=2));print(r['summary'])
