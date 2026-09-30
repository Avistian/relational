"""Final evidence review: source pins, full fits, selection, identities and notebooks."""
import hashlib,json,math,statistics,subprocess,sys
from pathlib import Path
import numpy as np,nbformat
from relkit.weakness_l149 import rank_catalog
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/l149'
subprocess.run([sys.executable,str(P/'_check_l149.py')],check=True)
def read(p):return json.loads(p.read_text())
b=read(P/'_budget_l149.json');g=read(E/'training.json');f=read(E/'fe/summary.json');a=read(E/'errors.json');cat=read(E/'catalog.json')
assert g['status']=='COMPLETE_SELECTED_TRAINING' and f['status']=='COMPLETE_RELEASED_PIPELINE_REPLAY'
assert read(E/'sources.json')['status']=='PASS'
for name,digest in b['source_hashes'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
n=0
for seed in range(5):
 gr=read(E/f'final/lr005-full/seed-{seed}/result.json');fr=read(E/f'fe/paper/seed-{seed}/result.json')
 assert len(gr['trace'])==10 and all(t['train_queries']==7453 for t in gr['trace'])
 assert gr['selected_epoch']==min(gr['trace'],key=lambda t:t['val_mae'])['epoch']
 assert len(fr['trace'])==10 and fr['selected_trial']==min(fr['trace'],key=lambda t:(t['val_mae'],t['number']))['number']
 for arm,rel in [('gnn',f'final/lr005-full/seed-{seed}'),('fe',f'fe/paper/seed-{seed}')]:
  x=np.load(E/rel/'predictions.npz')
  for split in ['val','test']:
   keys=list(zip(x[split+'_entity'],x[split+'_time']));assert len(set(keys))==len(keys)
   mean=math.fsum(abs(float(y)-float(p)) for y,p in zip(x[split+'_target'],x[split+'_pred']))/len(keys)
   assert abs(mean-a['splits'][split]['seeds'][seed][arm])<1e-12;n+=len(keys)
assert n==12590
sql=read(E/'fe/sql-audit.json');assert sql['status']=='PASS' and sum(v['rows'] for v in sql['splits'].values())==8712
assert read(P/'_audit_l149_results.json')['status']=='PASS'
assert read(P/'_source_check_fe_l149_results.json')['maximum_prediction_error']=={'val':0.,'test':0.}
for metric in ['AUROC','MAE']:assert rank_catalog([r for r in cat['rows'] if r['metric']==metric])==cat['rankings'][metric]
for row in cat['rows']:
 gr=row['geometry']['rdl']['right'];fr=row['geometry']['fe']['right']
 geometric=(gr-fr)/(424.63606-167.45)*.4 if row['metric']=='AUROC' else (fr-gr)/(798.48086-614.1642)*.4
 reported=next(r['gap'] for r in cat['rankings'][row['metric']] if r['task']==row['task'])
 assert abs(geometric-reported)<1e-14
assert len(cat['rows'])==15 and hashlib.sha256((E/'figure3.svg').read_bytes()).hexdigest()==cat['source_sha256']
assert a['splits']['test']['selected_slice']=='high_history'
for name in ['all','high_history']:
 interval=a['splits']['test']['slices'][name]['interval'];assert interval['low']<0<interval['high']
nb=nbformat.read(P/'solutions/0149-weakest-relbench-tasks.ipynb',4)
first_nb=nbformat.read(E/'notebook-first.ipynb',4)
def training_cells(book):
 start=next(i for i,c in enumerate(book.cells) if c.cell_type=='markdown' and c.source.startswith('## Optional complete training lanes'))
 return [c.source for c in book.cells[start:] if c.cell_type=='code']
assert training_cells(nb)==training_cells(first_nb)
code='\n\n'.join(c.source for c in nb.cells if c.cell_type=='code');sha=hashlib.sha256(code.encode()).hexdigest()
for name,key in [('_execution_l149_results.json','executed_code_sha256'),('_notebook_fe_l149_results.json','code_sha256'),('_notebook_gpu_l149_results.json','code_sha256')]:
 r=read(P/name);assert r['status']=='PASS' and r[key]==sha
 assert r.get('code_cells')==25
assert sum('raise NotImplementedError("TODO:' in c.source for c in nbformat.read(P/'0149-weakest-relbench-tasks.ipynb',4).cells)==3
reserved=sum(x['upper_usd'] for x in b['reservations'])+b['overhead_reserve_usd'];assert reserved<=b['budget_usd']
current=read(P/'_notebook_gpu_l149_results.json');first=read(E/'notebook-first-validation.json');cost=g['worker_resource_usd']+current['resource_usd']+first['resource_usd']
report=dict(status='PASS',primary_predictions=n,independent_labels=8712,full_gnn_primary_fits=5,full_fe_primary_search_trials=50,fe_selected_refits=5,source_graph_rows=74063,published_tasks=15,plot_values='PLOT_DERIVED',historical_identity='NOT_ESTABLISHED',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE',notebook_code_sha256=sha,final_gpu_notebook_full_fits=5,final_fe_notebook_full_trials=50,first_revision_validation='Retained; extra full validation due to stricter ranking-unit contract; training code unchanged',reserved_plus_overhead_usd=reserved,measured_worker_resource_estimate_usd=cost,invoice='NOT_ITEMIZED',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_verify_l149_results.json').write_text(json.dumps(report,indent=2));print(report)
