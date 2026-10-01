"""Independent raw-score oracle, randomized pairing and mutation rejection."""
import copy,hashlib,json,math,shutil,tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from _audit_l169 import audit169
from _replay_l170 import replay170
from _check_l170 import check170
from relkit.transfer_l167 import keyed_auc
from relkit.scaling_l169 import sample_context,scaling_curve,scaling_claim
from relkit.design_l170 import paired_design,adaptation_mode,claim_gate
P=Path(__file__).resolve().parent;E=P/'evidence/l170';manifest=json.loads((E/'input-manifest.json').read_text())
assert check170(paired_design,adaptation_mode,claim_gate)=='PASS'
for funcs in [(lambda rows:{'by_task_context':{}},adaptation_mode,claim_gate),(paired_design,lambda *a:'NO_TARGET_LABELS_OR_UPDATES',claim_gate),(paired_design,adaptation_mode,lambda *a:dict(status='READY_FOR_REVIEW',missing=[]))]:
    try:check170(*funcs)
    except (AssertionError,KeyError,ValueError):pass
    else:raise AssertionError('Wrong learner implementation accepted')
report=replay170(P,manifest,audit169,keyed_auc,sample_context,scaling_curve,scaling_claim,paired_design,adaptation_mode,claim_gate)
assert report==json.loads((E/'report.json').read_text())
original=json.loads((P/'evidence/l169/input-manifest.json').read_text());raw_rows=[];count=0;max_error=0
# Independently load only canonical experiment receipts, not the assembled report.
for spec in original['experiments']:
    db=spec['database']
    folders=[(P/spec['original_folder']/phase,512) for phase in spec['original_phases']]+[(P/'evidence/l169'/phase,None) for phase in ['pilot-1','remaining-complete']]
    for folder,fixed in folders:
        for record in json.loads((folder/'receipt.json').read_text())['records']:
            if fixed is None and record['database']!=db:continue
            k=fixed or record['context'];arm=record['arm'];seed=record['seed']
            path=folder/(record.get('filename') or f'{arm}-{seed}.npz')
            with np.load(path,allow_pickle=False) as a:
                value=float(roc_auc_score(a['label'],a['probability']));count+=len(a['label'])
            raw_rows.append(dict(database=db,context=k,arm=arm,seed=seed,auc=value))
            max_error=max(max_error,abs(value-record['auc']))
assert len(raw_rows)==300 and count==229050 and max_error<1e-12
oracle={}
for db in ['rel-f1','rel-trial']:
    for k in [64,128,256,512,1024]:
        differences=[]
        for seed in range(10):
            a=next(r['auc'] for r in raw_rows if (r['database'],r['context'],r['arm'],r['seed'])==(db,k,'RDBPFN',seed))
            b=next(r['auc'] for r in raw_rows if (r['database'],r['context'],r['arm'],r['seed'])==(db,k,'TabICLv1.1',seed))
            differences.append(a-b)
        got=report['paired']['by_task_context'][db+'/'+str(k)]
        np.testing.assert_allclose(got['per_seed'],differences,atol=1e-12,rtol=0)
        assert abs(got['mean']-np.mean(differences))<1e-12 and abs(got['sample_sd']-np.std(differences,ddof=1))<1e-12
rng=np.random.default_rng(170)
for _ in range(100):
    rows=copy.deepcopy(raw_rows);values=rng.uniform(0,1,len(rows))
    for r,v in zip(rows,values):r['auc']=float(v)
    rng.shuffle(rows);got=paired_design(rows)
    for k in [64,128,256,512,1024]:
        # Equal task weight, never pooled probabilities or row-weighted task averages.
        means=[]
        for db in ['rel-f1','rel-trial']:
            a=[r['auc'] for r in rows if r['database']==db and r['context']==k and r['arm']=='RDBPFN']
            b=[r['auc'] for r in rows if r['database']==db and r['context']==k and r['arm']=='TabICLv1.1']
            means.append(float(np.mean(a)-np.mean(b)))
        assert abs(got['macro_by_context'][str(k)]-np.mean(means))<1e-12
# Rank metrics against a direct pairwise definition on ties and shuffled composite keys.
for n in range(2,102):
    y=np.r_[0,1,rng.integers(0,2,n-2)];s=rng.integers(0,7,n)/6;keys=np.column_stack([np.arange(n)//2,np.arange(n)%2]);order=rng.permutation(n)
    expected=np.mean((s[y==1,None]>s[None,y==0])+.5*(s[y==1,None]==s[None,y==0]))
    assert abs(keyed_auc(keys,y,keys[order],s[order])-expected)<1e-12
# Full truth table, with a separate bit-mask oracle for each documented prerequisite set.
keys=['artifacts','metrics','dfs','temporal','exposure','fresh_training','matched_pipeline','heldout_databases','repeatability']
requirements={'saved_replay':[0,1],'full_pipeline':[0,1,2,3],'fresh_pretraining':[0,1,5,4],'general_advantage':[0,1,6,7,3,4],'exact_repeatability':[0,1,8]}
for mask in range(512):
    facts={key:bool(mask&(1<<i)) for i,key in enumerate(keys)}
    for claim,indices in requirements.items():
        missing=[keys[i] for i in indices if not mask&(1<<i)]
        assert claim_gate(claim,facts)==dict(status='NOT_ESTABLISHED' if missing else 'READY_FOR_REVIEW',missing=missing)
# Reauthenticate even when manipulated metadata is rehashed: semantic identities matter.
corruptions=0
with tempfile.TemporaryDirectory(prefix='l170-corrupt-') as tmp:
    root=Path(tmp)
    for name in manifest['files']:
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/name,dest)
    pins=json.loads((root/'evidence/l169/audit-manifest.json').read_text())
    rn='evidence/l169/remaining-complete/receipt.json';rp=root/rn;receipt_bytes=rp.read_bytes();first=json.loads(receipt_bytes)['records'][0]
    pn='evidence/l169/remaining-complete/'+first['filename'];pp=root/pn;prediction_bytes=pp.read_bytes()
    for mode in ['missing','duplicate','wrong_keys','wrong_labels','wrong_support','bad_hash']:
        changed=copy.deepcopy(pins);receipt=json.loads(receipt_bytes)
        if mode=='missing':receipt['records'].pop()
        elif mode=='duplicate':receipt['records'].append(receipt['records'][0])
        elif mode=='bad_hash':pp.write_bytes(b'corrupt')
        else:
            with np.load(pp) as raw:arr={k:raw[k] for k in raw.files}
            if mode=='wrong_keys':arr['keys'][0]=arr['keys'][1]
            elif mode=='wrong_labels':arr['label'][0]=1-arr['label'][0]
            else:arr['support_keys'][0]=arr['support_keys'][1]
            np.savez_compressed(pp,**arr);changed['files'][pn]=hashlib.sha256(pp.read_bytes()).hexdigest();receipt['records'][0]['sha256']=changed['files'][pn]
        rp.write_text(json.dumps(receipt));changed['files'][rn]=hashlib.sha256(rp.read_bytes()).hexdigest()
        try:audit169(root,changed,keyed_auc,sample_context,scaling_curve,scaling_claim)
        except ValueError:corruptions+=1
        else:raise AssertionError('Corrupt evidence accepted: '+mode)
        rp.write_bytes(receipt_bytes);pp.write_bytes(prediction_bytes)
r=dict(status='PASS',runs=len(raw_rows),predictions=count,max_metric_error=max_error,random_pairing_cases=100,pairwise_auc_cases=100,claim_gate_cases=2560,rejected_bad_learners=3,rejected_corrupt_packets=corruptions,full_replay_report_parity='EXACT',additional_cloud_spend_usd=0)
(P/'_verify_l170_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
