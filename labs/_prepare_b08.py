"""Freeze approved course inputs before outcomes; refuse changed protocol."""
import hashlib,json,platform,importlib.metadata
from pathlib import Path
import numpy as np
from relkit.limix_b08 import make_episodes
P=Path(__file__).resolve().parent;E=P/'evidence/b08';D=P/'data/b08';S=P/'sources/b08'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if (E/'course-protocol.json').exists():raise SystemExit('Already frozen; no overwrite')
files={}
for seed in range(3):
 for split,count,batch,offset in [('train',120,4,1000),('test',16,1,2000)]:
  p=D/f'{split}-{seed}.npz';np.savez_compressed(p,**make_episodes(offset+seed,count,batch=batch));files[str(p.relative_to(P))]=digest(p)
protocol=dict(name='B08-OBJECTIVE-ABLATION',frozen_before_results=True,seeds=[0,1,2],objectives=['target','feature','combined'],train=dict(steps=120,batch=4,support=24,query=8,features=4,optimizer='AdamW',lr=.001,weight_decay=.01,objective='Ly; Lx; Ly+Lx',selection='fixed final step; no validation/HPO'),model=dict(blocks=2,width=16,task_slots=4,feature_identity_rank=4,head='scalar MSE',precision='float32 CPU one thread deterministic'),evaluation=dict(episodes=16,queries_per_episode=8,metric='query target MSE and masked feature MSE',mask='one of four features per query; support fully observed',aggregation='pool equal-sized episodes; show per-seed points and sample SD',baseline='support mean for target and each feature'),data_files=files,code_sha256=digest(P/'relkit/limix_b08.py'),versions={k:importlib.metadata.version(k) for k in ['torch','numpy','nbformat','nbclient']},python=platform.python_version(),budget=dict(local_seconds=3600,usd_cap=10,planned_usd=8,reserve_usd=2),deviations=['Synthetic same-family tasks; no real-dataset generalization claim','Small two-block single-head model, LayerNorm, scalar heads and fixed mask pattern; not released LimiX architecture','No mixture-density decoder, mask-indicator prediction or published pretraining corpus','Combined loss is an unweighted sum; gradient magnitude may differ','Reconstruction-only leaves final target head untrained; poor target performance is not surprising'])
(E/'course-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
manifest={str(p.relative_to(S)):digest(p) for p in S.rglob('*') if p.is_file() and p.name!='release.tar.gz'}
(S/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
meta=json.loads((S/'checkpoint.json').read_text());ck=next(x for x in meta['siblings'] if x['rfilename'].endswith('.ckpt'))
gate=dict(target='2509.03505v2 Table23 LimiX-16M Analcatdata BroadwayMult',paper_rmse=.194,rounding_interval=[.1935,.1945],status='INCOMPLETE_SOURCE_PROTOCOL',published_mask_rate=.05,current_commit=json.loads((S/'commit.json').read_text())['sha'],historical_commit=json.loads((S/'historical-commits.json').read_text())[0]['sha'],checkpoint=dict(repo=meta['id'],revision=meta['sha'],file=ck['rfilename'],sha256=ck['lfs']['sha256'],size=ck['size'],downloaded=False,historical_identity='NOT_ESTABLISHED'),blockers=['Original Analcatdata train/test row identities and exact data version not authenticated','Original 5 percent masked-cell identities and repeat/seed aggregation not authenticated','Original per-column typing and fitted MinMax preprocessing for this result not authenticated','Released checkpoint identity at paper evaluation time not authenticated'],inspected=['paper section7.3 and Table23','historical and current complete Python/JSON source snapshots','both demo_missing_value_imputation.py: breast cancer, split42, 30 percent masking; not Table23 Analcatdata','current pinned checkpoint model card and license'],inference='NOT_RUN',pretraining='NOT_RUN',other_paper_datasets='NOT_RUN',full_limix2_benchmarks='NOT_RUN')
(E/'source-gate.json').write_text(json.dumps(gate,indent=2)+'\n')
print('Frozen',len(files),'episode arrays; Table23 source gate closed')
