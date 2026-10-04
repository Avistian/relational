"""Run the entire preregistered12-fit course grid. Never choose a winning seed."""
import json,time,platform
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,log_loss
from relkit.curriculum_b20 import task_pool,fit_course,evaluate,array_hash,paired_contract
P=Path(__file__).resolve().parent;E=P/'evidence/b20';E.mkdir(exist_ok=True,parents=True)
def metrics(y,p):
    return dict(auc=float(np.mean([roc_auc_score(a,b) for a,b in zip(y,p)])),logloss=float(log_loss(y.ravel(),p.ravel(),labels=[0,1])))
def main():
    torch.use_deterministic_algorithms(True)
    testparts=[task_pool(90001,'single',12),task_pool(90002,'relational',12)]
    test={k:np.concatenate([p[k] for p in testparts]) for k in testparts[0]}
    assert all(len(set(row[16:]))==2 for row in test['y'])
    np.savez_compressed(E/'evaluation.npz',**test)
    evalhash=array_hash(test['x'],test['y']);records=[]
    for seed in [0,1,2]:
        for family in ['single','relational']:
            pool=task_pool(1000+seed,family);np.savez_compressed(E/f'pool-{family}-{seed}.npz',**pool)
            pair=[]
            for mode in ['staged','shuffled']:
                started=time.monotonic();model,trace=fit_course(pool,mode,seed);pred=evaluate(model,test);elapsed=time.monotonic()-started
                name=f'{family}-{mode}-{seed}';np.savez_compressed(E/f'predictions-{name}.npz',p=pred,y=test['y'][:,16:])
                torch.save(model.state_dict(),E/f'weights-{name}.pt')
                row=dict(name=name,family=family,mode=mode,seed=seed,pool=array_hash(pool['x'],pool['y']),evaluation=evalhash,updates=192,batch_size=1,optimizer=dict(name='AdamW',lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8),selection='final',exposures=np.bincount(trace['order'],minlength=96).tolist(),generated_cells=int(pool['cells'].sum()),model_feature_cell_exposures=192*32*6,seconds=elapsed,final=array_hash(*[v.detach().numpy() for v in model.state_dict().values()]),parameters=sum(p.numel() for p in model.parameters()),**trace,**metrics(test['y'][:,16:],pred))
                records.append(row);pair.append(row);print(name,round(row['auc'],4),round(row['logloss'],4),round(elapsed,2),flush=True)
                (E/'partial.json').write_text(json.dumps(records,indent=2)+'\n')
            paired_contract(*pair)
    out=dict(status='COMPLETE_COURSE_EXPERIMENT',protocol='B20-ORDER-2x2-3',versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),fits=12,predictions=4608,rows=records)
    (E/'diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');(E/'partial.json').unlink()
if __name__=='__main__':main()
