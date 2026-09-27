"""Collect every full seed, then independently score before reporting a mean."""
import json,hashlib
from pathlib import Path
import modal,numpy as np
from relkit.amazon_l138 import rank_auc
P=Path(__file__).resolve().parent;E=P/'evidence/l138';v=modal.Volume.from_name('l138-amazon-evidence')
all_results=[];checkpoints={};baseline=np.load(E/'recency_predictions.npz');hashes={}
for seed in range(5):
 dest=E/'final'/f'seed-{seed}';dest.mkdir(parents=True,exist_ok=True)
 for name in ['result.json','predictions.npz','data_identity.json','progress.json']:
  raw=b''.join(v.read_file(f'final/seed-{seed}/{name}'));(dest/name).write_bytes(raw);hashes[f'seed-{seed}/{name}']=hashlib.sha256(raw).hexdigest()
 identity=json.loads((dest/'data_identity.json').read_text());assert all(x['match'] and x['sha256']==x['historical'] for x in identity.values())
 checkpoint=P/'results/l138/checkpoints'/f'seed-{seed}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256();size=0
 with checkpoint.open('wb') as f:
  for chunk in v.read_file(f'final/seed-{seed}/selected.pt'):f.write(chunk);digest.update(chunk);size+=len(chunk)
 checkpoints[str(seed)]=dict(volume='l138-amazon-evidence',path=f'final/seed-{seed}/selected.pt',sha256=digest.hexdigest(),bytes=size,local=str(checkpoint.relative_to(P)),published=False)
 result=json.loads((dest/'result.json').read_text());assert result['status']=='COMPLETE' and result['epochs']==10 and result['seed']==seed and len(result['history'])==10
 assert all(r['steps']==2001 for r in result['history'])
 assert result['temporal_audit']['query_occurrences']==15104717 and result['temporal_audit']['future_violations']==0
 expected_epoch=int(np.argmax([r['val']['roc_auc'] for r in result['history']]))+1;assert result['best_epoch']==expected_epoch
 assert result['selection_auc']==max(r['val']['roc_auc'] for r in result['history'])
 z=np.load(dest/'predictions.npz')
 for split in ['val','test']:
  for field in ['target','customer','time']:assert np.array_equal(z[split+'_'+field],baseline[split+'_'+field]),(seed,split,field)
  scores=z[split+'_pred'];assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
  auc=rank_auc(z[split+'_target'],scores);assert abs(auc-result['scores'][split]['roc_auc'])<1e-12
 all_results.append(result)
summary=dict(status='COMPLETE',seeds=list(range(5)),epochs=10,source_artifact_hashes=hashes,checkpoints=checkpoints,verified_predictions=5*761677,audited_query_occurrences=sum(x['temporal_audit']['query_occurrences'] for x in all_results),comparison_scope='DESCRIPTIVE_ONLY: released training row count differs from paper Table2',historical_paper_reproduction='NOT_ESTABLISHED',metrics={})
for split,target in [('val',.7045),('test',.7042)]:
 values=[r['scores'][split]['roc_auc'] for r in all_results];mean=float(np.mean(values));summary['metrics'][split]=dict(values=values,mean=mean,sample_sd=float(np.std(values,ddof=1)),paper_target=target,tolerance=.01,verdict='CLOSE' if abs(mean-target)<=.01 else 'OUTSIDE_TOLERANCE')
(E/'training.json').write_text(json.dumps(summary,indent=2));print(summary)
