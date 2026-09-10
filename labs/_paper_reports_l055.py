"""Extract/reanalyze pinned author reports. No model training is performed.

Usage: python labs/_paper_reports_l055.py --checkout /path/to/tabred
Without --checkout, fetch a detached pinned checkout in an ignored data cache.
A missing report is recorded, never filled with another split, seed or ensemble.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
REVISION='b5ef15b3749f30da7a1eb8fba21a5b54d706bf32'
PAPER_ARMS={'mlp':'MLP','mlp-plr':'MLP-PLR','xgboost_':'XGBoost','tabr':'TabR-S'}
PAPER_TASKS=('homesite-insurance','ecom-offers','homecredit-default','sberbank-housing','cooking-time','delivery-eta','maps-routing','weather')


def aggregate_reports(records, matched=False):
    """Equal window weight after seed means; optional common-seed sensitivity.

    Seed SD is conditional within a window. Window range is descriptive because
    sliding histories overlap. No t interval treating 45 runs as independent.
    """
    result=[]
    for task in PAPER_TASKS:
        common={}
        for window in range(3):
            common[window]=set.intersection(*[{r['seed'] for r in records if r['task']==task and r['window']==window and r['model']==model and r['protocol']==protocol}
                for model in PAPER_ARMS.values() for protocol in ('random','temporal')])
        for model in PAPER_ARMS.values():
            for protocol in ('random','temporal'):
                windows=[]
                for window in range(3):
                    selected=[r for r in records if r['task']==task and r['window']==window and r['model']==model and r['protocol']==protocol and (not matched or r['seed'] in common[window])]
                    scores=np.array([r['value'] for r in selected])
                    if not len(scores): raise ValueError('Empty model/window cell')
                    windows.append(dict(window=window,n=len(scores),mean=float(scores.mean()),seed_sd=float(scores.std(ddof=1))))
                means=[w['mean'] for w in windows]
                result.append(dict(task=task,model=model,protocol=protocol,metric=selected[0]['metric'],
                    mean=float(np.mean(means)),window_min=min(means),window_max=max(means),windows=windows))
    return result


def extract(checkout):
    revision=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    if revision != REVISION: raise ValueError('Checkout must match pinned revision')
    tree=subprocess.check_output(['git','-C',str(checkout),'ls-tree','-r',REVISION,'--','paper/exp/temporal-shift-analysis'],text=True)
    blobs={line.split('\t',1)[1]:line.split()[2] for line in tree.splitlines()}
    records=[];missing=[]
    for arm,model in PAPER_ARMS.items():
        for task in PAPER_TASKS:
            for protocol,prefix in [('random','random'),('temporal','sliding-window')]:
                for window in range(3):
                    for seed in range(15):
                        path=Path('paper/exp/temporal-shift-analysis')/arm/f'{task}-{prefix}-{window}'/'evaluation'/str(seed)/'report.json'
                        if str(path) not in blobs:
                            if (checkout/path).exists(): raise ValueError('Untracked report injected: '+str(path))
                            missing.append(str(path));continue
                        raw=(checkout/path).read_bytes()
                        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
                        if blob != blobs[str(path)]: raise ValueError('Report differs from committed blob: '+str(path))
                        d=json.loads(raw)
                        config=d['config'];assert config['seed']==seed
                        assert config['data']['split']==f'{prefix}-{window}'
                        assert config['data']['path']==f':data/{task}'
                        metric='rmse' if 'rmse' in d['metrics']['test'] else 'roc-auc'
                        val=float(d['metrics']['test'][metric]);score=d['metrics']['test']['score']
                        assert np.isclose(score, -val if metric=='rmse' else val)
                        records.append(dict(task=task,model=model,protocol=protocol,window=window,seed=seed,metric=metric,value=val,
                            validation=float(d['metrics']['val'][metric]),git_blob=blob,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                            config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()))
    return dict(status='AUTHOR_REPORT_REANALYSIS',revision=revision,expected=2880,found=len(records),missing=missing,
                source='https://github.com/yandex-research/tabred/tree/'+revision+'/paper/exp/temporal-shift-analysis',
                retraining='NOT_RUN',records=records,summary=aggregate_reports(records),matched_seed_summary=aggregate_reports(records,matched=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkout',type=Path);a=p.parse_args()
    checkout=a.checkout or ROOT/'data/cache/l055/paper-source'
    if not checkout.exists():
        subprocess.run(['git','clone','--quiet','https://github.com/yandex-research/tabred.git',str(checkout)],check=True)
        subprocess.run(['git','-C',str(checkout),'checkout','--quiet',REVISION],check=True)
    result=extract(checkout)
    out=ROOT/'_paper_l055_results.json';out.write_text(json.dumps(result,indent=1)+'\n')
    print(json.dumps({k:result[k] for k in ('status','expected','found','missing')}))
    for task in PAPER_TASKS:
        rows={(r['model'],r['protocol']):r['mean'] for r in result['summary'] if r['task']==task}
        metric=next(r['metric'] for r in result['summary'] if r['task']==task)
        margins=[(rows['XGBoost',s]-rows['MLP-PLR',s])*(1 if metric=='roc-auc' else -1) for s in ('random','temporal')]
        print(task,'XGBoost advantage over MLP-PLR:',*[round(v,6) for v in margins])
