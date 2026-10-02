"""Pin complete inherited sampler evidence and original RT code without model execution."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l180';S=P/'sources/l180';packet=E/'packet';packet.mkdir(exist_ok=True)
hash_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal=json.loads((P/'evidence/l175/artifact-manifest.json').read_text())['files']
selected={}
for seed in range(3):selected[f'contexts-{seed}.npz']=P/f'evidence/l175/audit-3/contexts-{seed}.npz'
for name,src in [('context-audit.json','evidence/l175/audit-3/context-audit.json'),('table_info.json','sources/l175/table_info.json'),('column_index.json','sources/l175/column_index.json'),('label-oracle.npz','evidence/l175/label-oracle.npz'),('verified-audit.json','evidence/l175/verified-audit.json')]:selected[name]=P/src
inherited={}
for name,src in selected.items():
 key=str(src.relative_to(R));assert hash_file(src)==seal[key],key
 shutil.copyfile(src,packet/name);inherited[key]=seal[key]
# Raw tables allow the portable audit to verify the inherited label oracle too.
rawpins=json.loads((P/'evidence/l175/raw-oracle-manifest.json').read_text())
for path,digest in rawpins.items():
 src=P/'evidence/l171/db'/Path(path).name
 assert hash_file(src)==digest,src
 shutil.copyfile(src,packet/src.name)
selected_sources={}
for src in sorted((P/'sources/l175').rglob('*')):
 if src.is_file() and src.suffix in ['.py','.rs','.toml','.md','.json','.lock']:
  key=str(src.relative_to(R))
  if key in seal:assert hash_file(src)==seal[key];inherited[key]=seal[key]
  dest=S/src.relative_to(P/'sources/l175');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
  selected_sources[str(dest.relative_to(P))]=hash_file(dest)
for name,src in [('paper.html',P/'sources/l175/paper.html'),('pricing.html',P/'sources/l177/modal-pricing.html')]:
 if not src.exists():src=P/'sources/l175/pricing.html'
 shutil.copyfile(src,S/name);selected_sources[str((S/name).relative_to(P))]=hash_file(S/name)
for name in ['example_finetune.py','model-card.md']:
 shutil.copyfile(S/name,packet/name)
shutil.copyfile(S/'upstream/rt/main.py',packet/'trainer.py')
config=dict(name='L180 RT-v1 public-encoder fine-tuning checkpoint',task='rel-f1/driver-dnf',paper='2510.06377v1',code_revision='8d83590b5ae7fba9e40e8df463ed2dd9066ce5fb',checkpoint_revision='299701dedae451f3dfa40717b831d9dc17c0e4e7',checkpoint='pretrain_rel-f1_driver-dnf.pt',data_revision='e8b48dc2cfb0a3c9171a8fddaaef14b6240f18ee',seed=0,context_seeds=[0,1,2],paper_rounded_steps=33000,source_steps=32769,per_rank_batch=32,world_size=8,global_batch=256,seq_len=1024,lr=1e-4,wd=0.0,hours_per_run='1.5',rates={'A100_40GB':'0.000583','A100_80GB':'0.000694'},cap_usd='10',cloud_authorized_usd=0,local_cap_seconds=1800)
(packet/'config.json').write_text(json.dumps(config,indent=2)+'\n')
manifest=dict(files={str(p.relative_to(packet)):hash_file(p) for p in sorted(packet.rglob('*')) if p.is_file()},inherited_sealed_files=inherited,raw_table_provenance=rawpins)
(E/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(S/'source-ledger-l180.json').write_text(json.dumps(dict(files=selected_sources,source_parent='sources/l175/source-ledger.json',paper_url='https://arxiv.org/html/2510.06377v1',pricing_url='https://modal.com/pricing',rates_checked='2026-10-02',checkpoint_bytes='NOT_DOWNLOADED',post_gate_training='UNVALIDATED'),indent=2)+'\n')
print('Pinned',len(manifest['files']),'packet files and',len(selected_sources),'source files')
