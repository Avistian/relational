"""DCNv2 Table 6 MovieLens attempt. Presets are explicitly documented below.

Same model as the notebook; injectable model/train/predict keep student work live.
Completed seeds resume, incomplete seeds restart. Never silently mix run configs.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from relkit.dcnv2 import DCNv2, train_dcn, predict_dcn
from relkit.dcnv2_data import load_movielens
from relkit.saint_experiment import environment
from relkit.paper_repro import PaperTarget, LabFinding, ScaleUpRun, classify_number, format_ledger

PRESETS={
    'smoke':dict(cap=2000,seeds=[0],epochs=1,embedding_dim=4,hidden=[16,16],batch_size=128,lr=.001,ema_decay=None),
    'closer':dict(cap=None,seeds=[0,1,2],epochs=5,embedding_dim=16,hidden=[128,128],batch_size=128,lr=.001,ema_decay=None),
    'paper':dict(cap=None,seeds=[0,1,2,3,4],epochs=20,embedding_dim=30,hidden=[128,128],batch_size=128,lr=.001,ema_decay=.9999)}
DEVIATIONS=[
    'Original split indices/seeds unavailable; new fixed random 80/10/10 split, seed 48.',
    'Six fields interpreted as user, movie, gender, age, occupation, ZIP; train-only vocabularies.',
    'MovieLens-specific winning depth/MLP/LR/step count not disclosed in Table 6; chosen depth=2, parallel layout and fixed settings, no full hyperparameter search.',
    'PyTorch implementation and initialization details differ from the TensorFlow v1 experiments.',
    'Criteo and private production data are outside this attempt.']


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def run(preset='closer',output_dir='data/cache/l048-closer',archive_path=None,device='cpu',
        model_cls=DCNv2,train_fn=train_dcn,predict_fn=predict_dcn,implementation_id=None):
    torch.set_num_threads(1)
    if implementation_id is None and (model_cls is not DCNv2 or train_fn is not train_dcn or predict_fn is not predict_dcn):
        raise ValueError('Injected notebook implementations must supply their source digest as implementation_id.')
    start=time.time(); cfg=PRESETS[preset].copy()
    dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
    fr=load_movielens(archive_path,cap=cfg['cap'])
    meta=fr['meta']; protocol_hash=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    source=Path(__file__).parent/'relkit/dcnv2.py'
    fingerprint={'preset':preset,'config':cfg,'protocol_sha256':protocol_hash,
                 'model_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                 'implementation_id':implementation_id or 'canonical',
                 'data_loader_sha256':hashlib.sha256((source.parent/'dcnv2_data.py').read_bytes()).hexdigest(),
                 'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'versions':environment(),'device':device}
    manifest=dest/'run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=fingerprint:
        raise ValueError('Existing directory belongs to different code/data/config/device. Choose a new output directory.')
    atomic_json(manifest,fingerprint)
    atomic_json(dest/'protocol.json',meta)
    xn,xc,y,tr,va,te=[fr[k] for k in ('xn','xc','y','train','valid','test')]
    rows=[]
    for seed in cfg['seeds']:
        saved=dest/f'seed-{seed}.json'
        if saved.exists() and (dest/f'seed-{seed}.pt').exists() and (dest/f'seed-{seed}-predictions.npz').exists():
            rows.append(json.loads(saved.read_text()));print('Reusing completed seed',seed,flush=True);continue
        tick=time.time();torch.manual_seed(seed)
        model=model_cls(0,fr['cards'],embedding_dim=cfg['embedding_dim'],hidden=cfg['hidden'],
                        depth=2,kind='dense',layout='parallel')
        model,history=train_fn(model,xn,xc,y,tr,va,seed=seed,epochs=cfg['epochs'],
                               batch_size=cfg['batch_size'],lr=cfg['lr'],device=device,ema_decay=cfg['ema_decay'])
        p=predict_fn(model,xn[te],xc[te])
        row={'seed':seed,'logloss':float(log_loss(y[te],p,labels=[0,1])),
             'auroc':float(roc_auc_score(y[te],p)), 'wall_s':time.time()-tick,
             'parameters':sum(v.numel() for v in model.parameters()),'history':history}
        torch.save(model.state_dict(),dest/f'seed-{seed}.pt')
        np.savez_compressed(dest/f'seed-{seed}-predictions.npz',row_ids=np.asarray(meta['row_ids'])[te],target=y[te],probability=p)
        atomic_json(saved,row);rows.append(row)
        print('Completed',seed,'logloss',row['logloss'],'AUROC',row['auroc'],'seconds',row['wall_s'],flush=True)
    values=np.array([r['logloss'] for r in rows]);aucs=np.array([r['auroc'] for r in rows])
    deviations=DEVIATIONS+[f'Preset {preset}: {cfg}. Preset name does not certify protocol equality.']
    target=PaperTarget(paper='Wang et al. WWW 2021',arxiv='2008.13535v2',table='Table 6 DCN-V2',
                        dataset='MovieLens-1M',metric='logloss',paper_value=.3170,paper_std=.00036,
                        higher_is_better=False,abs_tol=.001,paper_split='random 80/10/10')
    attempt=ScaleUpRun(method='dense parallel DCNv2',dataset='MovieLens-1M',metric='logloss',
                      value=float(values.mean()),std=float(values.std(ddof=1)) if len(values)>1 else None,
                      n_seeds=len(rows),hardware=device+' / '+platform.machine(),wall_s=sum(r['wall_s'] for r in rows),
                      protocol_deviations=deviations,protocol_match=False)
    verdict=classify_number(target,attempt)
    ledger=format_ledger(title='L048 — MovieLens binary task',
                         lab=[LabFinding('Local comparison','see _verify_l048_results.json','Three substitute tables, fixed small budgets')],
                         paper=[(target,attempt,verdict)],extra_lines=['Also reported AUROC: '+str(float(aucs.mean()))])
    result={'preset':preset,'config':cfg,'rows':len(y),'seeds':rows,'mean_logloss':attempt.value,
            'sample_std_logloss':attempt.std,'mean_auroc':float(aucs.mean()),
            'sample_std_auroc':float(aucs.std(ddof=1)) if len(aucs)>1 else None,
            'verdict':verdict,'protocol_deviations':deviations,'ledger':ledger,
            'wall_s':time.time()-start,'training_wall_s':attempt.wall_s,
            'protocol_sha256':protocol_hash,'environment':environment()}
    atomic_json(dest/'summary.json',result);(dest/'ledger.txt').write_text(ledger+'\n')
    print(ledger,flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--preset',choices=PRESETS,default='closer')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--archive-path')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args(argv)
    return run(**vars(args))


if __name__=='__main__':
    main()
