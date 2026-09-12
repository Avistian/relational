"""Independently refit L070 candidates and reconstruct selection/interventions.

Shares only the immutable, previously source-checked corrected TabM architecture;
does not import the L070 evaluator, preprocessing, training loop or student tasks.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from relkit.tabm_v2 import TabM
from _archive_l070 import ROOT,task,loss,digest

def check(evidence,output):
    start=time.perf_counter();torch.set_num_threads(1);r=json.loads(Path(evidence).read_text());cfg=r['config'];checks=[]
    assert digest(ROOT/'relkit/tabm_v2.py')=='0fbd840771d45973f4519bac409630ad000946bdc0f01323da08275608e5694e'
    expected={(d,a,s)for d in cfg['datasets']for a in ['XGBoost-fresh-control','TabM-mini-v2']for s in cfg['seeds']}
    observed=[(v['dataset'],v['arm'],v['seed'])for v in r['records']];assert len(set(observed))==len(observed) and set(observed)==expected
    for dataset in cfg['datasets']:
        x,y,parts,sha,labels=task(dataset.split('/')[0]);meta=r['datasets'][dataset];assert meta['data_sha256']==sha
        for name,ids in parts.items():assert ids.tolist()==meta['ids'][name]
        # Match the imputer's column-major float32 realization, independently of its code.
        tr=np.asfortranarray(x[parts['train']]);va=np.asfortranarray(x[parts['val']]);te=np.asfortranarray(x[parts['test']]);yt=y[parts['train']];yv=y[parts['val']];ytest=y[parts['test']]
        cx=tr-tr.mean(0);cy=yt-yt.mean();den=np.sqrt(np.sum(cx**2,axis=0)*np.sum(cy**2));corr=np.divide(cx.T@cy,den,out=np.zeros(tr.shape[1]),where=den>0);column=int(np.argmax(np.abs(corr)));altered=te.astype(float);replacement=float(np.median(tr[:,column]));altered[:,column]=replacement
        for row in [v for v in r['records'] if v['dataset']==dataset]:
            assert row['validation_ids']==parts['val'].tolist() and row['test_ids']==parts['test'].tolist()
            assert row['validation_targets']==yv.tolist() and row['targets']==ytest.tolist()
            assert row['intervention']['column']==column and row['intervention']['replacement']==replacement
            # Float32 reductions differ slightly between C- and Fortran-layout inputs.
            correlation_delta=abs(row['intervention']['train_correlation']-corr[column]);assert correlation_delta<2e-6
            assert len(row['candidates'])==2;errors=[];predictions=[]
            for i,record in enumerate(row['candidates']):
                assert record['candidate']==i
                if row['arm']=='XGBoost-fresh-control':
                    model=XGBClassifier(n_estimators=cfg['trees'],max_depth=[3,6][i],learning_rate=.05,n_jobs=1,random_state=row['seed'],subsample=.8,colsample_bytree=.8,tree_method='hist');model.fit(tr,yt)
                    vp=model.predict_proba(va)[:,1];history=[loss(yv,vp)];epoch=None
                    def predict(z):return model.predict_proba(z)[:,1]
                else:
                    torch.manual_seed(row['seed']);prep=StandardScaler().fit(tr);xx=torch.tensor(prep.transform(tr),dtype=torch.float32);vv=torch.tensor(prep.transform(va),dtype=torch.float32);yy=torch.tensor(yt,dtype=torch.long)
                    model=TabM(din=tr.shape[1],width=64,depth=3,k=8,arch='mini',seed=row['seed'],dropout=.1)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=[.001,.003][i],weight_decay=1e-4);history=[];best=float('inf');state=None;epoch=None
                    for step in range(cfg['epochs']):
                        model.train()
                        for batch in torch.randperm(len(xx)).split(128):
                            objective=model.member_losses(xx[batch],yy[batch]);optimizer.zero_grad();objective.backward();optimizer.step()
                        model.eval()
                        with torch.no_grad():vp=model.predict(vv)[:,1].numpy()
                        score=loss(yv,vp);history.append(score)
                        if score<=best:best=score;state=copy.deepcopy(model.state_dict());epoch=step+1
                    model.load_state_dict(state);model.eval()
                    def predict(z):
                        with torch.no_grad():return model.predict(torch.tensor(prep.transform(z),dtype=torch.float32))[:,1].numpy()
                    vp=predict(va);np.testing.assert_allclose(record['preprocessing']['mean'],prep.mean_,atol=1e-12);np.testing.assert_allclose(record['preprocessing']['scale'],prep.scale_,atol=1e-12)
                assert record['selected_epoch']==epoch
                np.testing.assert_allclose(record['epoch_losses'],history,atol=1e-10,rtol=1e-10)
                delta=float(np.max(abs(vp-np.asarray(record['validation_predictions']))));assert delta<2e-7,(dataset,row['arm'],row['seed'],i,delta)
                error=loss(yv,vp);assert abs(error-record['validation_error'])<1e-10;errors.append(error)
                # These independent audit predictions cannot influence archived selection.
                predictions.append((predict(te).copy(),predict(altered).copy()))
            selected=int(np.argmin(errors));assert selected==row['selected'];p,changed=predictions[selected]
            delta=float(np.max(abs(p-np.asarray(row['predictions']))));change_delta=float(np.max(abs(changed-np.asarray(row['intervention']['predictions']))));assert max(delta,change_delta)<2e-7,(dataset,row['arm'],row['seed'],delta,change_delta)
            assert abs(loss(ytest,row['predictions'])-row['error'])<1e-12
            assert abs(loss(ytest,p)-row['error'])<1e-6
            assert abs(loss(ytest,row['intervention']['predictions'])-row['intervention']['error'])<1e-12
            assert abs(loss(ytest,changed)-row['intervention']['error'])<1e-6
            assert abs(row['intervention']['error']-row['error']-row['intervention']['delta_loss'])<1e-12
            assert abs(row['lifecycle_100_batches_seconds']-(row['fit_selection_seconds']+100*row['predict_seconds']))<1e-10
            assert abs(row['seconds']-row['fit_selection_seconds']-row['predict_seconds'])<1e-10
            checks.append(dict(dataset=dataset,arm=row['arm'],seed=row['seed'],candidates=2,validation_rows=len(yv),test_rows=len(ytest),prediction_delta=delta,intervention_delta=change_delta,correlation_delta=correlation_delta,selected=selected));print('REFIT',dataset,row['arm'],row['seed'],delta,flush=True)
    report=dict(status='PASS',evidence_sha256=digest(evidence),checker_sha256=digest(__file__),raw_data_checker_sha256=digest(ROOT/'_archive_l070.py'),model_sha256=digest(ROOT/'relkit/tabm_v2.py'),records=len(checks),candidate_fits=2*len(checks),validation_predictions=sum(2*v['validation_rows'] for v in checks),selected_and_intervention_predictions=sum(2*v['test_rows'] for v in checks),seconds=time.perf_counter()-start,checks=checks,scope='Independent raw data/partitions and fresh fitting of every candidate, complete epoch-loss traces, validation vectors, candidate/epoch selection, train-only intervention, all selected test/intervention probabilities and metric/cost arithmetic. Shares the immutable source-checked TabM architecture, not the L070 evaluator. Wall times are not independently reproduced.')
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='checks'}));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evidence');p.add_argument('output');a=p.parse_args();check(a.evidence,a.output)
