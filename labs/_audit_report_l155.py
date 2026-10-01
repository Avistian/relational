"""Independent SQL paired-error oracle and adversarial evidence checks."""
import copy,hashlib,json,sqlite3,tempfile,shutil
from pathlib import Path
import numpy as np
from _report_l155 import make_report
from _check_l155 import check_pairs,check_effort,check_ratio
from relkit.effort_l155 import paired_losses,summarize_effort,effort_ratio
P=Path(__file__).resolve().parent;E=P/'evidence/l155';manifest=json.loads((E/'input-manifest.json').read_text());report=make_report(E,manifest)
count=0;error=0
for seed in range(5):
 f=np.load(E/f'fe/paper/seed-{seed}/predictions.npz');g=np.load(E/f'paper/seed-{seed}/predictions.npz')
 for split in ['val','test']:
  c=sqlite3.connect(':memory:')
  for name,a in [('fe',f),('gnn',g)]:
   c.execute(f'create table {name}(entity integer,cutoff integer,y real,p real,primary key(entity,cutoff))')
   c.executemany(f'insert into {name} values(?,?,?,?)',[(int(i),int(t),float(y),float(p)) for i,t,y,p in zip(a[split+'_entity'],a[split+'_time'],a[split+'_target'],a[split+'_pred'])])
  n,b=c.execute('select count(*),avg(abs(fe.y-fe.p)-abs(fe.y-gnn.p)) from fe join gnn using(entity,cutoff) where fe.y=gnn.y').fetchone();c.close()
  row=next(r for r in report['rows'] if r['seed']==seed and r['split']==split);error=max(error,abs(b-row['benefit_mae']));assert error<1e-12 and n==(499 if split=='val' else 760);count+=n*2
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)
 for name in manifest['files']:
  dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(E/name,dest)
 (root/'effort-log.json').write_text('{}')
 try:make_report(root,manifest)
 except AssertionError:pass
 else:raise AssertionError('Changed evidence admitted')
# Actual learner functions are required, not silently replaced by reference imports.
for name,param in [('pairs','pair'),('effort','effort'),('ratio','ratio')]:
 def marker(*a,**kw):raise RuntimeError('live learner invoked')
 try:make_report(E,manifest,**{param:marker})
 except RuntimeError as e:assert str(e)=='live learner invoked'
 else:raise AssertionError(name+' unused')
r=dict(status='PASS',sqlite_prediction_rows=count,maximum_sql_difference=error,changed_bytes='REJECTED',live_functions=3)
(P/'_audit_report_l155_results.json').write_text(json.dumps(r,indent=2));print(r)
