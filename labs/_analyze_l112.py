"""Independent accuracy/selection reconstruction: no canonical trainer imports."""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;E=P/'evidence/l112';sources=json.loads((P/'_sources_l112.json').read_text());audit=json.loads((P/'_audit_l112_results.json').read_text());rows=[];cost=0
for seed in range(10):
 d=E/'paper'/f'seed-{seed}';r=json.loads((d/'result.json').read_text());identity=json.loads((d/'identity.json').read_text());done=json.loads((d/'completed.json').read_text());cost+=done['resource_usd']
 assert identity['source_sha256']==sources['source_sha256'] and identity['data']['arrays_sha256']==audit['arrays_sha256']
 assert r['seed']==seed and r['epochs']==len(r['history'])==500 and r['parameters']==110120 and r['status']=='COMPLETE'
 assert hashlib.sha256((d/'predictions.npz').read_bytes()).hexdigest()==r['predictions_sha256']
 best=max(row['valid'] for row in r['history']);idx=next(i for i,row in enumerate(r['history']) if row['valid']==best);assert r['selected_epoch']==idx+1
 z=np.load(d/'predictions.npz');assert len(z['pred'])==169343
 measured={k:sum(int(a==b) for a,b in zip(z['pred'][z[k]],z['y'][z[k]]))/len(z[k]) for k in ['train','valid','test']}
 for k,value in measured.items():assert value==r['scores'][k]==r['history'][idx][k]
 row={'seed':seed,'selected_epoch':idx+1,**{k+'_percent':v*100 for k,v in measured.items()}};rows.append(row)
pilot=json.loads((E/'pilot/seed-100/completed.json').read_text());cost+=pilot['resource_usd']
summary={}
for k,target in [('valid',73.),('test',71.74)]:
 scores=[r[k+'_percent'] for r in rows];mean=statistics.mean(scores)
 summary[k]={'mean_percent':mean,'sample_sd_pp':statistics.stdev(scores),'target_percent':target,'gap_pp':mean-target,'tolerance_pp':.5,'verdict':'CLOSE' if abs(mean-target)<=.5 else 'OUTSIDE_TOLERANCE'}
out={'status':'COMPLETE','seeds':rows,'summary':summary,'total_training_epochs':5000,'independently_scored_nodes':1693430,'successful_resource_usd':cost,'unitemized_costs':'Failed pretraining checker attempt, startup/build/storage; conservatively bounded in budget file','historical_identity':'NOT_ESTABLISHED','full_paper':'NOT_ESTABLISHED','learner_status':'PENDING_WRITTEN_DEFENSE'}
(E/'summary.json').write_text(json.dumps(out,indent=2));b=json.loads((P/'_budget_l112.json').read_text());b.update(status='COMPLETE',successful_resource_usd=cost);(P/'_budget_l112.json').write_text(json.dumps(b,indent=2));print(out)
