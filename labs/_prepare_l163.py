"""Freeze primary sources, model revision, all row values and split memberships."""
import hashlib,json,time
from pathlib import Path
from importlib.metadata import version
import numpy as np,requests
P=Path(__file__).resolve().parent;E=P/'evidence/l163';S=P/'sources/l163'
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
ledger=[]
for file,url in [
 ('paper.html','https://arxiv.org/html/2305.15321v1'),
 ('model-api.json','https://huggingface.co/api/models/facebook/bart-base'),
 ('bart-docs.html','https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/bart'),
 ('lab-repositories.json','https://api.github.com/orgs/DataManagementLab/repos?per_page=100&type=public')]:
 r=requests.get(url,timeout=45);r.raise_for_status();(S/file).write_bytes(r.content)
 ledger.append(dict(file=file,url=url,sha256=hashlib.sha256(r.content).hexdigest(),http_status=r.status_code))
api=json.loads((S/'model-api.json').read_text());revision=api['sha']
write_json(S/'source-ledger.json',dict(retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),sources=ledger,model_revision=revision,
 historical_release='NOT_LOCATED_IN_INSPECTED_SOURCES',prior_audit='labs/sources/l159/manifest.json',global_absence='NOT_ESTABLISHED'))
rng=np.random.default_rng(163);rows=[]
for i in range(240):
 price=float(np.round(rng.uniform(10,200),2));weight=float(np.round(rng.uniform(.1,8),2))
 colour=str(rng.choice(['red','blue','green']));condition=str(rng.choice(['new','used','refurbished']))
 target=2*price+5*weight+dict(new=8,used=-4,refurbished=2)[condition]+dict(red=3,blue=-2,green=0)[colour]+float(rng.normal(0,2))
 rows.append(dict(id=f'p{i:03d}',price_usd=price,weight_kg=weight,colour=colour,condition=condition,target=target))
splits=[]
for seed in [0,1,2]:
 order=np.random.default_rng(seed).permutation(240).tolist()
 splits.append(dict(seed=seed,train=order[:144],validation=order[144:192],test=order[192:]))
packet=dict(experiment='L163 Frozen Row Encoder Comparison',scope='SYNTHETIC_COURSE_COMPARISON',rows=rows,splits=splits,
 alphas=[.01,.1,1.,10.,100.],variants=['baseline','reordered','renamed'],model_id='facebook/bart-base',revision=revision,
 batch_size=8,max_length=1024,pooling='final hidden mean over attention_mask=1 including BOS/EOS',dtype='float32',threads=1,
 generator='NumPy default_rng(163); see _prepare_l163.py',target_formula='2*price_usd+5*weight_kg+condition_offset+colour_offset+Normal(0,2)',
 sample_size=240,cloud_spend_usd=0,historical_reproduction='NOT_RUN',historical_fidelity='NOT_ESTABLISHED')
write_json(E/'fixtures.json',packet)
write_json(E/'input-manifest.json',dict(fixtures_sha256=hashlib.sha256((E/'fixtures.json').read_bytes()).hexdigest(),source_ledger_sha256=hashlib.sha256((S/'source-ledger.json').read_bytes()).hexdigest()))
packages=['numpy','scipy','scikit-learn','torch','transformers','huggingface-hub','tokenizers','safetensors','nbformat','nbclient','nbconvert']
(P/'l163-requirements.txt').write_text('# Exact author environment; install CPU torch for your platform.\n'+'\n'.join(n+'=='+version(n) for n in packages)+'\n')
print('Pinned 240 rows, 3 complete splits, model revision',revision)
print('Available weight files:',[x['rfilename'] for x in api['siblings'] if x['rfilename'].endswith(('.bin','.safetensors'))])
