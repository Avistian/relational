"""Authenticate raw fresh-run receipts and every prediction before summarizing."""
import hashlib,json
from pathlib import Path
import numpy as np
from relkit.exit_l200 import keyed_auc,complete_grid,exit_gate

def audit(root,score=keyed_auc,grid=complete_grid,gate=exit_gate):
    root=Path(root);manifest=json.loads((root/'packet-manifest.json').read_text())
    for name,h in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('Changed input '+name)
    data=np.load(root/'packet/prepared.npz');records=[];runs=[];vectors={a:[] for a in ['RDBPFN','RDBPFN_single','TabICLv1.1']}
    for phase in ['pilot-1','full-1']:
        receipt=json.loads((root/phase/'receipt.json').read_text())
        if receipt['input_manifest_sha256']!=manifest['input_manifest_sha256']:raise ValueError('Wrong input receipt')
        for row in receipt['records']:
            records.append(row);path=root/phase/f"{row['arm']}-{row['seed']}.npz"
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Changed predictions')
            x=np.load(path);seed=row['seed']
            np.testing.assert_array_equal(x['keys'],data['test_keys']);np.testing.assert_array_equal(x['label'],data['y_test'])
            np.testing.assert_array_equal(x['support_keys'],data['train_keys'][data['support'][seed]])
            if len(set(map(tuple,x['support_keys'])))!=512 or set(map(tuple,x['keys'])) & set(map(tuple,x['support_keys'])):raise ValueError('Invalid support identities')
            auc=score(data['test_keys'],data['y_test'],x['keys'],x['probability'])
            if abs(auc-row['auc'])>1e-12:raise ValueError('Receipt score mismatch')
            runs.append(dict(arm=row['arm'],seed=seed,auc=auc))
    counts=grid(records)
    for row in sorted(runs,key=lambda r:(r['arm'],r['seed'])):vectors[row['arm']].append(row['auc'])
    targets=dict(RDBPFN=.7219,RDBPFN_single=.6640,**{'TabICLv1.1':.7176})
    models={a:dict(mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)),per_seed=v,paper_target=targets[a],status='CLOSE' if abs(np.mean(v)-targets[a])<=.02 else 'OUTSIDE_TOLERANCE') for a,v in vectors.items()}
    deltas=np.array(vectors['RDBPFN'])-vectors['TabICLv1.1']
    return dict(experiment='L200 RDB-PFN v5 Table9 F1 driver-dnf 512-support fresh checkpoint evaluation',execution='COMPLETE_SELECTED_REPRODUCTION',**counts,models=models,paired_rdbpfn_minus_tabicl=dict(mean=float(deltas.mean()),sample_sd=float(deltas.std(ddof=1)),positive=int((deltas>0).sum()),per_seed=deltas.tolist()),comparison='CLOSE' if all(m['status']=='CLOSE' for m in models.values()) else 'OUTSIDE_TOLERANCE',tolerance=.02,exam=gate('PASS' if all(m['status']=='CLOSE' for m in models.values()) else 'FAIL','PENDING','PENDING'),learner='PENDING_WRITTEN_DEFENSE',rdb_learn='INCOMPLETE_SOURCE_PREPROCESSING_GATE',fresh_pretraining='NOT_RUN',whole_paper='NOT_RUN',historical_identity='NOT_ESTABLISHED',full_feature_availability='NOT_ESTABLISHED',proposal_source='Frozen L198 and L199 author reports; learner submission required')

if __name__=='__main__':
    e=Path(__file__).resolve().parent/'evidence/l200';r=audit(e);(e/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
