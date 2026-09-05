"""L047: fixed-split, three-dataset/three-model/three-seed ablation.
CPU: OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l047.py
"""
import json
import os
from pathlib import Path
import time
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from relkit.saint import SAINT, train_saint, predict_saint
from relkit.saint_experiment import prepare, environment, summarize


def run(datasets=('credit_g','diabetes','blood_transfusion'), seeds=(0,1,2), epochs=20,
        model_cls=SAINT, train_fn=train_saint, predict_fn=predict_saint):
    torch.set_num_threads(1)
    t0=time.time(); per={}; protocols={}; context={}
    for name in datasets:
        fr=prepare(name); protocols[name]=fr['meta']; per[name]=[]
        xn,xc,y,tr,va,te=[fr[k] for k in ('xn','xc','y','train','valid','test')]
        for seed in seeds:
            row={}
            for variant in ('col','colrow'):
                torch.manual_seed(seed)
                model=model_cls(xn.shape[1],fr['cards'],d=8,depth=1,heads=4,ff_dropout=.1,variant=variant)
                model,hist=train_fn(model,xn,xc,y,tr,va,seed=seed,epochs=epochs,batch_size=64)
                p=predict_fn(model,xn[te],xc[te],64)
                row[variant]=float(roc_auc_score(y[te],p))
                # The same trained weights, same test rows and labels; only companions change.
                singleton=predict_fn(model,xn[te],xc[te],1)
                context[f'{name}/{seed}/{variant}']={
                    'max_abs_probability_change':float(np.max(np.abs(p-singleton))),
                    'auc_batch64':row[variant], 'auc_batch1':float(roc_auc_score(y[te],singleton)),
                    'best_epoch':max(hist,key=lambda r:r['valid_score'])['epoch']}
            cbx=np.column_stack([np.nan_to_num(xn),xc]).astype(object)
            cat_idx=list(range(xn.shape[1],xn.shape[1]+xc.shape[1]))
            for j in cat_idx: cbx[:,j]=cbx[:,j].astype(str)
            cb=CatBoostClassifier(iterations=300,depth=6,learning_rate=.05,verbose=False,
                                  random_seed=seed,thread_count=1,allow_writing_files=False,eval_metric='AUC')
            cb.fit(cbx[tr],y[tr],cat_features=cat_idx,eval_set=(cbx[va],y[va]),use_best_model=True)
            row['catboost']=float(roc_auc_score(y[te],cb.predict_proba(cbx[te])[:,1]))
            per[name].append(row); print(name,seed,row,flush=True)
    return {'lesson':47,'environment':environment(),'seeds':list(seeds),
            'config':{'epochs':epochs,'d':8,'depth':1,'heads':4,'ff_dropout':.1,
                      'batch_size':64,'learning_rate':.001,'selection':'validation AUC each epoch',
                      'catboost':{'iterations':300,'depth':6,'learning_rate':.05}},
            'protocols':protocols,'results':summarize(per,('col','colrow','catboost')),
            'context_probe':context,'wall_s':time.time()-t0,
            'scope':'Downscaled supervised ablation on substitutes. col is depth-matched; not paper SAINT-s (six layers). No pretraining; not a paper benchmark reproduction.'}


if __name__ == '__main__':
    out=run()
    Path(__file__).with_name('_verify_l047_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out['results'],indent=2))
