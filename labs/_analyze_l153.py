"""Independently rescore actual rankings; decide affordability before five-seed fits."""
import gzip,hashlib,json,math
from pathlib import Path
import numpy as np
from relkit.recommendation_l153 import ranking_metrics,portfolio_entry
P=Path(__file__).resolve().parent;E=P/'evidence/l153';b=json.loads((P/'_budget_l153.json').read_text());meta=json.loads((E/'prepared/prepared.json').read_text());pilot=json.loads((E/'pilot/result.json').read_text())
truth={s:json.load(gzip.open(E/f'prepared/{s}-truth.json.gz','rt')) for s in ['val','test']}
assert {t['time'] for t in truth['val']}=={1577836800000000000}
assert {t['time'] for t in truth['test']}=={1609459200000000000}
for s in truth:assert len(truth[s])==meta['counts'][s] and len({(r['entity'],r['time']) for r in truth[s]})==len(truth[s])
checked=0;independent={};records=[]
for directory in [E/'pilot']+sorted(E.glob('seed-*')):
 if not directory.is_dir() or not (directory/'result.json').exists():continue
 r=json.loads((directory/'result.json').read_text());z=np.load(directory/'predictions.npz');scores={}
 for split in r['scores']:
  key_to_truth={(t['entity'],t['time']):set(t['positives']) for t in truth[split]};pred=z[split+'_pred'];keys=list(zip(map(int,z[split+'_entity']),map(int,z[split+'_time'])))
  assert len(keys)==len(set(keys))==len(key_to_truth) and set(keys)==key_to_truth.keys()
  assert all(len(set(map(int,row)))==10 for row in pred) and pred.min()>=0 and pred.max()<meta['num_dst_nodes']
  hits=np.array([[int(v) in key_to_truth[key] for v in row] for key,row in zip(keys,pred)],dtype=np.float64)
  counts=np.array([len(key_to_truth[key]) for key in keys]);aps=((hits.cumsum(axis=1)/np.arange(1,11))*hits).sum(axis=1)/np.minimum(counts,10)
  scores[split]=dict(map=float(aps.mean()),hit=float(hits.any(axis=1).mean()),recall=float((hits.sum(axis=1)/counts).mean()),n=len(keys))
  rows=[dict(entity=e,time=t,ranking=list(map(int,row))) for (e,t),row in zip(keys,pred)]
  live=ranking_metrics(truth[split],list(reversed(rows)),10,meta['num_dst_nodes'])
  for metric in ['map','hit','recall']:assert abs(scores[split][metric]-live[metric])<1e-12
  assert abs(scores[split]['map']-r['scores'][split])<1e-12;checked+=len(keys)
 independent[directory.name]=scores
 if r['status']=='COMPLETE':records.append(r)
trace=pilot['trace'][0];full_train_seconds=trace['train_seconds']/32*min(pilot['loader_batches'],2001)*20
full_eval_seconds=trace['validation_seconds']*22
setup=max(0,pilot['seconds']-trace['train_seconds']-2*trace['validation_seconds'])
per_fit=full_train_seconds+full_eval_seconds+setup
projected=per_fit*5*b['rate_usd_second'];safe_per_fit=per_fit*1.25+120
reserved=sum(r['upper_usd'] for r in b['reservations']);remaining=b['budget_usd']-b['overhead_reserve_usd']-reserved
forecast=5*safe_per_fit*b['rate_usd_second']+600*b['rate_usd_second']
decision=dict(decision='PROCEED' if safe_per_fit<4200 and forecast<=remaining else 'STOP',pilot_training_batches=32,full_training_batches=pilot['loader_batches'],pilot_training_seconds=trace['train_seconds'],full_validation_seconds=trace['validation_seconds'],projected_per_fit_seconds=per_fit,projected_five_run_compute_usd=projected,safety_adjusted_per_fit_seconds=safe_per_fit,safety_adjusted_five_runs_and_validation_usd=forecast,remaining_compute_allowance_usd=remaining,assumption='20epochs; full released timestamp batches;22full-validation-equivalent evaluation passes;1.25safety plus120s per fit; conservative first32batch audit/cold-start cost. Scenario estimate, not lower bound.',full_dispatch_timeout_seconds=4200,budget_usd=10)
# Preserve the pre-dispatch cost decision once complete fits exist.
if not records:(E/'cost-decision.json').write_text(json.dumps(decision,indent=2))
portfolio=portfolio_entry(records,hashlib.sha256((P/'sources/l153/protocol.json').read_bytes()).hexdigest(),{s:meta['counts'][s] for s in ['val','test']}) if len(records)==5 else None
costs={p.name:json.loads(p.read_text()) for p in E.glob('*-cost.json')}
r=dict(status='PASS',full_selected_reproduction='COMPLETE' if portfolio else 'INCOMPLETE',paper_comparison=portfolio['paper_comparison'] if portfolio else 'NOT_RUN',pilot=pilot,records=records,portfolio=portfolio,independently_rescored_rankings=checked,independent_metrics=independent,labels=json.loads((E/'prepared/independent-labels.json').read_text()),prepared=meta,worker_body_usd=sum(c['worker_body_usd'] for c in costs.values()),costs=costs,historical_identity='NOT_ESTABLISHED',feature_arrival_legality='NOT_ESTABLISHED',whole_paper='NOT_RUN',fresh_manual_fe='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(E/'summary.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k in ['status','full_selected_reproduction','paper_comparison','independently_rescored_rankings','worker_body_usd']});print(decision)
