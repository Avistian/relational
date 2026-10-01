"""Preserve failed validation and independently score the completed portable gate."""
import argparse,hashlib,json
from pathlib import Path
import modal,numpy as np,nbformat
P=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--attempt',default='notebook-final');p.add_argument('--failed',action='store_true');a=p.parse_args();v=modal.Volume.from_name('l155-regression-evidence');E=P/'evidence/l155'/a.attempt;E.mkdir(parents=True,exist_ok=True)
names=['failure.txt','cost.json'] if a.failed else ['execution.json','cost.json','l155-report.json']+[f'l155-gnn-full/seed-{s}/{n}' for s in range(5) for n in ['predictions.npz','result.json','audit.json','temporal-audit.json','checkpoint-audit.json','diagnostics.json']]
for name in names:
 dest=E/name;dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('wb') as f:
  for chunk in v.read_file(a.attempt+'/'+name):f.write(chunk)
if a.failed:print('Preserved failed attempt',a.attempt);raise SystemExit
notebook=nbformat.read(P/'solutions/0155-compare-manual-fe.ipynb',4);code='\n\n'.join(c.source for c in notebook.cells if c.cell_type=='code');execution=json.loads((E/'execution.json').read_text());assert hashlib.sha256(code.encode()).hexdigest()==execution['code_sha256']
count=0;rows=[]
for seed in range(5):
 root=E/f'l155-gnn-full/seed-{seed}';d=np.load(root/'predictions.npz');r=json.loads((root/'result.json').read_text());ref=np.load(P/f'evidence/l155/paper/seed-{seed}/predictions.npz')
 assert r['epochs']==10 and len(r['trace'])==10 and all(t['train_queries']==7453 for t in r['trace'])
 assert r['selected_epoch']==min(r['trace'],key=lambda x:x['val_mae'])['epoch']
 for split in ['val','test']:
  for key in ['entity','time','target']:np.testing.assert_array_equal(d[split+'_'+key],ref[split+'_'+key])
  score=float(np.mean(abs(d[split+'_target']-d[split+'_pred'])));assert abs(score-r['scores'][split])<1e-12;count+=len(d[split+'_pred'])
 rows.append(dict(seed=seed,scores=r['scores']))
r=dict(status='PASS',code_sha256=execution['code_sha256'],code_cells=execution['cells'],predictions=count,fits=5,records=rows,cost=json.loads((E/'cost.json').read_text()),scope='Full portable gate; excluded from primary means',live_colab='NOT_CHECKED')
(P/'_notebook_gnn_l155_results.json').write_text(json.dumps(r,indent=2));print(r)
