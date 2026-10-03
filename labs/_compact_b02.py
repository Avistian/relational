"""Create a lossless subset for independent final-prediction audit; raw runs retained."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/b02';C=E/'compact';C.mkdir(exist_ok=True)
S=P/'sources/b02/upstream';F=E/'fresh/run-audited/california';V=F/'eval-online-ensembles/greedy/evaluation'
def cp(src,name):
 p=C/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,p)
for name in ['x_num.npy','y.npy']:cp(S/'data/california'/name,'data/'+name)
for part in ['train','val','test']:cp(S/f'data/california/splits/default/{part}.npy','data/'+part+'.npy')
for name in ['config.json','report.json','experiments.json','observed_ensemble.npz']:cp(F/'main'/name,'runs/main/'+name)
for name in ['config.json','report.json']:cp(V/name,'runs/evaluation/'+name)
for i in range(5):cp(V/str(i)/'observed_ensemble.npz',f'runs/{i}/observed_ensemble.npz')
cp(S/'experiments/tabpack-cosine/california/eval-online-ensembles/greedy/evaluation/report.json','released-report.json')
manifest={str(p.relative_to(C)):hashlib.sha256(p.read_bytes()).hexdigest() for p in C.rglob('*') if p.is_file() and p.name!='compact-lock.json'}
(C/'compact-lock.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Compact evidence',len(manifest),'files')
