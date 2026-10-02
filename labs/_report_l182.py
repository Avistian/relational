"""Authenticate fresh predictions; independently score all30 runs and preserve pairing."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from relkit.composite_l182 import keyed_auc

def summarize182(packet,score):
    expected={(a,s) for a in ['RDBPFN','RDBPFN_single','TabICLv1.1'] for s in range(10)}
    seen=set();by={a:{} for a,s in expected};supports={};base_keys=None;base_labels=None
    for row in packet:
        identity=(row['arm'],row['seed'])
        if identity not in expected or identity in seen:raise ValueError('Missing, duplicate or unexpected run')
        seen.add(identity);k=row['keys'];y=row['label'];support=row['support_keys'];seed=row['seed']
        if len(k)!=702 or len(support)!=512 or len(set(map(tuple,support)))!=512:raise ValueError('Incomplete population')
        if set(map(tuple,k))&set(map(tuple,support)):raise ValueError('Support/test overlap')
        if base_keys is None:base_keys=k;base_labels=y
        if k!=base_keys or y!=base_labels:raise ValueError('Mismatched test identities/labels')
        if seed in supports and support!=supports[seed]:raise ValueError('Unpaired support draws')
        supports[seed]=support;by[row['arm']][seed]=score(k,y,k,row['probability'])
    if seen!=expected:raise ValueError('Incomplete grid')
    targets={'RDBPFN':.7219,'RDBPFN_single':.6640,'TabICLv1.1':.7176};models={};vectors={}
    for arm,values in sorted(by.items()):
        v=np.array([values[s] for s in range(10)]);vectors[arm]=v
        delta=float(v.mean()-targets[arm])
        models[arm]=dict(mean=float(v.mean()),sample_sd=float(v.std(ddof=1)),paper=targets[arm],delta=delta,per_seed=v.tolist(),status='CLOSE' if abs(delta)<=.02 else 'OUTSIDE_TOLERANCE')
    paired={}
    for other in ['RDBPFN_single','TabICLv1.1']:
        v=vectors['RDBPFN']-vectors[other]
        paired[other]=dict(mean=float(v.mean()),sample_sd=float(v.std(ddof=1)),min=float(v.min()),max=float(v.max()),positive=int(sum(v>0)),per_seed=v.tolist())
    return dict(experiment='L182 fresh Table9 F1/driver-dnf 512support three-arm released-checkpoint evaluation',status='COMPLETE_SELECTED_REPRODUCTION',runs=30,predictions=21060,test_queries=702,models=models,paired=paired,tolerance=.02,hybrid_training='NOT_RUN',fresh_pretraining='NOT_RUN',whole_paper='NOT_RUN',historical_identity='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')

if __name__=='__main__':
 P=Path(__file__).resolve().parent;E=P/'evidence/l182';data=np.load(P/'evidence/l166/prepared.npz');packet=[];errors=[];bodies=[]
 for folder in ['pilot-1','full-1']:
  receipt=json.loads((E/folder/'receipt.json').read_text());bodies.append(json.loads((E/folder/'cost.json').read_text())['worker_body_seconds'])
  for row in receipt['records']:
   path=E/folder/f"{row['arm']}-{row['seed']}.npz";assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
   x=np.load(path);np.testing.assert_array_equal(x['keys'],data['test_keys']);np.testing.assert_array_equal(x['label'],data['y_test']);np.testing.assert_array_equal(x['support_keys'],data['train_keys'][data['support'][row['seed']]])
   value=keyed_auc(x['keys'],x['label'],x['keys'],x['probability']);oracle=roc_auc_score(x['label'],x['probability']);assert abs(value-oracle)<1e-12 and abs(value-row['auc'])<1e-12
   errors.append(abs(value-oracle));record={k:x[k].tolist() for k in x.files};record.update(arm=row['arm'],seed=row['seed']);packet.append(record)
 report=summarize182(packet,keyed_auc)
 for bad in [packet[:-1],packet+[packet[0]]]:
  try:summarize182(bad,keyed_auc)
  except ValueError:pass
  else:raise AssertionError('Bad run grid accepted')
 inherited=json.loads((P/'evidence/l166/predictions.json').read_text());lookup={(r['arm'],r['seed']):r for r in inherited}
 changes=[float(np.max(abs(np.array(r['probability'])-np.array(lookup[(r['arm'],r['seed'])]['probability'])))) for r in packet]
 budget=json.loads((E/'budget.json').read_text())
 audit=dict(status='PASS',rows=21060,independent_sklearn_max_error=max(errors),old_vs_fresh_max_probability_difference=max(changes),fresh_worker_body_seconds=sum(bodies),reservation_plus_overhead_usd=sum(r['upper_usd'] for r in budget['reservations'])+budget['overhead_reserve_usd'],invoice='NOT_ITEMIZED',fresh_modal_apps=['ap-9Npqn0drBLYxIqicxc53z3','ap-mpZjQE71VmLQPL6zFFquSi'])
 for name,obj in [('predictions',packet),('report',report),('evaluation-audit',audit)]:
  (E/(name+'.json')).write_text(json.dumps(obj,indent=None if name=='predictions' else 2)+'\n')
 print(json.dumps(report,indent=2));print(audit)
