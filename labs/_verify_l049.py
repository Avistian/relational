"""L049 three author-split datasets, three seeds; separate timestamp transfer probe."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score,log_loss
from relkit.claim_models import ExcelFormer,train_excel,predict_excel
from relkit.claim_data import paper_data,temporal_data
from relkit.saint_experiment import summarize,environment

ARMS=('excel_no_da','excel_feat_mix','catboost')


def experiment(data,seeds=(0,1,2),epochs=30,d=32,layers=2,heads=4,lr=.001,batch=64,
               patience=12,dropout=.1,trees=300,device='cpu',model_cls=ExcelFormer,
               train_fn=train_excel,predict_fn=predict_excel,artifact_dir=None):
    torch.set_num_threads(1)
    x,y,tr,va,te=[data[k] for k in ('x','y','train','valid','test')]
    rows,losses,diagnostics=[],[],{}
    if artifact_dir is not None:
        artifact_dir=Path(artifact_dir);artifact_dir.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        auc,logs={},{}
        for arm in ARMS:
            if arm=='catboost':
                model=CatBoostClassifier(iterations=trees,depth=6,learning_rate=.05,random_seed=seed,
                    thread_count=1,verbose=False,allow_writing_files=False,eval_metric='Logloss')
                model.fit(x[tr],y[tr],eval_set=(x[va],y[va]),use_best_model=True)
                p=model.predict_proba(x[te])[:,1]
                diagnostics[f'{seed}/{arm}']={'selected_trees':int(model.tree_count_)}
                if artifact_dir is not None: model.save_model(str(artifact_dir/f'{seed}-{arm}.cbm'))
            else:
                torch.manual_seed(seed)
                model=model_cls(x.shape[1],d=d,layers=layers,heads=heads,dropout=dropout)
                model,hist=train_fn(model,x,y,tr,va,seed=seed,epochs=epochs,lr=lr,batch=batch,
                    patience=patience,augmentation=arm=='excel_feat_mix',importance=data['importance'],device=device)
                p=predict_fn(model,x[te])
                diagnostics[f'{seed}/{arm}']={'history':hist,'parameters':sum(p.numel() for p in model.parameters())}
                if artifact_dir is not None: torch.save(model.state_dict(),artifact_dir/f'{seed}-{arm}.pt')
            auc[arm]=float(roc_auc_score(y[te],p));logs[arm]=float(log_loss(y[te],p,labels=[0,1]))
            diagnostics[f'{seed}/{arm}']['test_probabilities']=p.tolist()
        rows.append(auc);losses.append(logs)
    return {'auroc':rows,'log_loss':losses,'diagnostics':diagnostics,'protocol':data['meta']}


def run(**kwargs):
    start=time.time();paper={};transfer={}
    for name in ('pima','breast','banknote'):
        paper[name]=experiment(paper_data(name),**kwargs)
        print('paper split finished:',name,flush=True)
    for kind in ('random','temporal'):
        transfer[kind]=experiment(temporal_data(kind),**kwargs)
        print('transfer finished:',kind,flush=True)
    return {'paper_split_runs':paper,'transfer_runs':transfer,
            'summary':summarize({k:v['auroc'] for k,v in paper.items()},ARMS),
            'transfer_summary':summarize({k:v['auroc'] for k,v in transfer.items()},ARMS),
            'environment':environment(),'elapsed_seconds':time.time()-start,
            'model_sha256':hashlib.sha256((Path(__file__).parent/'relkit/claim_models.py').read_bytes()).hexdigest(),
            'configuration':dict(epochs=30,d=32,layers=2,heads=4,lr=.001,batch=64,patience=12,dropout=.1,trees=300,seeds=[0,1,2]),
            'verdict':'INCOMPARABLE to published tuned means: width, depth, training/search budgets and seed protocol differ. Transfer is a separate task.'}


if __name__=='__main__':
    result=run()
    Path(__file__).with_name('_verify_l049_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['summary'],indent=2))
