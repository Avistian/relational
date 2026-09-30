"""Collect immutable run artifacts, reject incomplete runs, score by full key."""
import json,hashlib,sys
from pathlib import Path
import modal,numpy as np
from relkit.ablation_l148 import keyed_mae,interaction_summary,ARMS
P=Path(__file__).resolve().parent;E=P/'evidence/l148';v=modal.Volume.from_name('l148-ablation-evidence')
def fetch(name):
 raw=b''.join(v.read_file(name));dest=E/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert dest.read_bytes()==raw,'Refusing changed artifact '+name
 dest.write_bytes(raw);return dest
phase=sys.argv[1]
if phase=='prepare':
 for n in ['audit.json','queries.npz','cost.json']:fetch('prepared/'+n)
else:
 pairs=[('full',0)] if phase=='pilot' else [(a,s) for a in ARMS for s in range(5)]
 ref=np.load(E/'prepared/queries.npz');runs=[];verified=0;payload={k:ref[k] for k in ref.files};hashes={}
 for a,s in pairs:
  name=f'{a}-{s}'
  for f in ['result.json','predictions.npz','cost.json','started.json']:
   p=fetch(name+'/'+f);hashes[name+'/'+f]=hashlib.sha256(p.read_bytes()).hexdigest()
  r=json.loads((E/name/'result.json').read_text());z=np.load(E/name/'predictions.npz');assert r['status']=='COMPLETE' and r['epochs']==10 and r['arm']==a and r['seed']==s
  assert len(r['trace'])==10 and all(t['train_queries']==7453 for t in r['trace'])
  assert r['selected_epoch']==1+int(np.argmin([t['val_mae'] for t in r['trace']]))
  for split in ['val','test']:
   keys=list(zip(ref[split+'_entity'],ref[split+'_time']));pk=list(zip(z[split+'_entity'],z[split+'_time']))
   np.testing.assert_array_equal(z[split+'_target'],ref[split+'_target'])
   score=keyed_mae(keys,ref[split+'_target'],pk[::-1],z[split+'_pred'][::-1]);assert abs(score-r['scores'][split])<1e-10
   for field in ['entity','time','target','pred']:payload[f'{a}_{s}_{split}_{field}']=z[split+'_'+field]
   verified+=len(keys)
  path=P/f'results/l148/{name}.pt';path.parent.mkdir(parents=True,exist_ok=True)
  if not path.exists():
   with path.open('wb') as f:
    for b in v.read_file(name+'/selected.pt'):f.write(b)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==r['checkpoint_sha256']
  runs.append(r)
 if phase=='pilot':
  cost=json.loads((E/'full-0/cost.json').read_text());forecast=cost['seconds']*1.25+120
  gate=dict(approved=forecast<=900,measured_seconds=cost['seconds'],forecast_per_fit_seconds=forecast,cutoff_seconds=900,full_experiment_planned_upper_usd=9.43302)
  (E/'pilot-gate.json').write_text(json.dumps(gate,indent=2));print(gate)
 else:
  assert len({r['run_id'] for r in runs})==25
  scores={split:{a:{r['seed']:r['scores'][split] for r in runs if r['arm']==a} for a in ARMS} for split in ['val','test']}
  means={sp:{a:dict(mean=float(np.mean(list(d.values()))),sample_sd=float(np.std(list(d.values()),ddof=1))) for a,d in arms.items()} for sp,arms in scores.items()}
  paired={sp:{a:dict(differences=(diff:=np.array(list(d.values()))-np.array(list(arms['full'].values()))).tolist(),mean=float(diff.mean()),sample_sd=float(diff.std(ddof=1))) for a,d in arms.items() if a!='full'} for sp,arms in scores.items()}
  inter={sp:interaction_summary(arms['full'],arms['encoder'],arms['messages'],arms['combined']) for sp,arms in scores.items()}
  summary=dict(status='COMPLETE',runs=runs,means=means,paired=paired,interaction=inter,verified_predictions=verified,artifact_hashes=hashes,baseline_targets={'val':3.193,'test':4.022},baseline_closeness={sp:'CLOSE' if abs(means[sp]['full']['mean']-target)<=.2+1e-12 else 'OUTSIDE_TOLERANCE' for sp,target in [('val',3.193),('test',4.022)]},whole_paper='NOT_RUN',historical_identity='NOT_ESTABLISHED',extensions='EXPLORATORY_COURSE_EXPERIMENTS')
  (E/'summary.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(E/'audit-inputs.npz',**payload);print(dict(verified=verified,means=means,interaction=inter))
