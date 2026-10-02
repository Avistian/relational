"""Rebuild the declared audit packet from archived inputs; never fetch new model/data revisions."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l178';S=P/'sources/l178'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=json.loads((E/'input-manifest.json').read_text())
for name,digest in base['files'].items():assert sha(P/name)==digest,name
paths=[P/n for n in base['files']]+[E/'input-manifest.json']
paths+=list(S.rglob('*.py'))+[S/'relgnn/LICENSE']+list(S.glob('torch-frame-*'))
paths+=[E/n for n in ['config.json','gradient-preflight.json','gradient-independent.json','gradient-input.parquet','fastdfs-preflight.json','preflight-baseline.parquet','preflight-mutated.parquet','support-schedule.npz']]
paths+=[P/n for n in ['_gradient_preflight_l178.py','_preflight_l178.py','_verify_gradient_l178.py','_parity_l141.py','_run_l166.py','_full_l141.py','sources/l166/upstream/LICENSE']]
paths+=list((P/'sources/l141').glob('examples__*.py'))+list((P/'sources/l166/upstream/model_pretrain').rglob('*.py'))
manifest=dict(experiment=base['experiment'],files={p.relative_to(P).as_posix():sha(p) for p in sorted(set(paths)) if p.is_file()},scope='Complete published prediction replay, full raw label reconstruction and stopped local prerequisite evidence; not a fresh model comparison')
old=json.loads((E/'audit-input-manifest.json').read_text());assert manifest==old,'Pin set changed; retain the original approved packet and inspect before replacing'
print('Authenticated identical audit packet:',len(manifest['files']),'files')
