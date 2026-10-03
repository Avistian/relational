"""Independent arithmetic and adversarial evidence checks; no new predictions."""
import copy,hashlib,json,math,sqlite3,tempfile,shutil,statistics
from pathlib import Path
from decimal import Decimal
import numpy as np
from bs4 import BeautifulSoup
from relkit.stress_l195 import paired_mae,interval_verdict,claim_scope,keyed_auc,cluster_interval
from _replay_l195 import replay195
P=Path(__file__).resolve().parent;E=P/'evidence/l195';Q=E/'packet';m=json.loads((E/'input-manifest.json').read_text());r=json.loads((E/'report.json').read_text())
rows=0
for split,n in [('val',499),('test',760)]:
 deltas=[]
 for seed in range(5):
  with np.load(Q/f'l149/gnn/{seed}/predictions.npz') as z:g={k:z[k] for k in z.files}
  with np.load(Q/f'l149/fe/{seed}/predictions.npz') as z:f={k:z[k] for k in z.files}
  db=sqlite3.connect(':memory:');db.execute('create table truth(id integer,t integer,y real,primary key(id,t))');db.execute('create table g(id integer,t integer,p real,primary key(id,t))');db.execute('create table f(id integer,t integer,p real,primary key(id,t))')
  keys=list(zip(f[split+'_entity'].tolist(),f[split+'_time'].tolist()))
  db.executemany('insert into truth values(?,?,?)',[(a,b,float(y)) for (a,b),y in zip(keys,f[split+'_target'])])
  for name,arr in [('g',g),('f',f)]:db.executemany('insert into '+name+' values(?,?,?)',list(zip(arr[split+'_entity'].tolist(),arr[split+'_time'].tolist(),arr[split+'_pred'].tolist()))[::-1])
  measured=db.execute('select avg(abs(y-g.p)),avg(abs(y-f.p)),avg(abs(y-f.p)-abs(y-g.p)),count(*) from truth join g using(id,t) join f using(id,t)').fetchone();db.close()
  saved=r['regression'][split]['runs'][seed]
  assert measured[3]==n
  for val,key in zip(measured,['gnn_mae','fe_mae','gnn_advantage']):assert abs(val-saved[key])<1e-12
  deltas.append(np.abs(f[split+'_target']-f[split+'_pred'])-np.abs(f[split+'_target']-g[split+'_pred'].astype(float)));rows+=2*n
 # Independent bootstrap using explicit driver lists and original draw sequence.
 mean=np.mean(deltas,axis=0);entities=f[split+'_entity'];groups=sorted(set(entities.tolist()));blocks=[mean[entities==x].tolist() for x in groups]
 samples=np.random.default_rng(137).integers(0,len(groups),size=(2000,len(groups)))
 estimates=[math.fsum(math.fsum(blocks[j]) for j in draw)/sum(len(blocks[j]) for j in draw) for draw in samples]
 lo,hi=np.quantile(estimates,[.025,.975]);ci=r['regression'][split]['conditional_interval']
 assert max(abs(lo-ci['low']),abs(hi-ci['high']))<1e-12
for folder in ['pilot-1','full-1']:
 for p in (Q/'l182/model'/folder).glob('*.npz'):
  with np.load(p) as z:y=z['label'];prob=z['probability']
  pos=prob[y==1];neg=prob[y==0];score=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))
  arm,seed=p.stem.rsplit('-',1);assert abs(score-r['icl']['models'][arm]['per_seed'][int(seed)])<1e-12;rows+=len(y)
assert rows==33650==r['prediction_rows']
soup=BeautifulSoup((Q/'l194/packet/paper.html').read_text(),'html.parser');counts=[0,0,0];paper_rows={};table_id=0
for table in soup.find_all('table'):
 lines=[[c.get_text(' ',strip=True) for c in row.find_all(['td','th'])] for row in table.find_all('tr')]
 header=next((x for x in lines if 'Dataset' in x and 'RDBLearn' in x),None)
 if header is None:continue
 table_id+=1;dataset=''
 for x in lines[lines.index(header)+1:]:
  if len(x)!=len(header):continue
  dataset=x[0] or dataset;task=x[1].split(' (')[0].lower();paper_rows[(table_id,dataset.lower(),task)]=(Decimal(x[header.index('RDBLearn')]),Decimal(x[header.index('AutoGluon+DFS')]))
tasks=json.loads((Q/'l194/packet/tasks.json').read_text())
for t in tasks:
 a,b=paper_rows[(t['paper_table'],t['dataset'],t['task'])];d=(a-b)*(1 if t['metric']=='AUROC' else -1);counts[0 if d>0 else 1 if d<0 else 2]+=1
 x=next(x for x in r['rdblearn']['tasks'] if x['task']==t['id']);assert float(a)==x['paper_model'] and float(b)==x['paper_comparator']
assert counts==[17,3,1] and len(paper_rows)==21
rejected=0
def fails(fn):
 global rejected
 try:fn()
 except (ValueError,AssertionError):rejected+=1
 else:raise AssertionError('Invalid evidence accepted')
truth=[(1,1,0),(1,2,1),(2,1,1)];pred=[(2,1,.5),(1,1,.5),(1,2,1)]
assert abs(keyed_auc(truth,pred)-.75)<1e-12
for bad in [pred[:-1],pred+[pred[0]],[(1,1,2),(1,2,.3),(2,1,.4)],[(1,1,float('nan')),(1,2,.3),(2,1,.4)]]:fails(lambda bad=bad:keyed_auc(truth,bad))
fails(lambda:keyed_auc([(1,1,0)],[(1,1,.3)]))
for incomplete in [False,0,None,'true']:
 assert claim_scope('pipeline',dict(authenticated=True,complete=incomplete,measured=True,comparable=True))=='INSUFFICIENT_EVIDENCE'
for low in np.linspace(-.5,.5,11):
 for high in np.linspace(low,.6,7):
  for margin in [0,.1,.5]:
   expected='ABOVE_MARGIN' if low>margin else 'BELOW_NEGATIVE_MARGIN' if high<-margin else 'WITHIN_MARGIN' if margin>0 and low>=-margin and high<=margin else 'UNRESOLVED'
   assert interval_verdict(float(low),float(high),margin)==expected
args=[paired_mae,interval_verdict,claim_scope,keyed_auc,cluster_interval]
for index,wrong in [(0,lambda *x:dict(keys=[],candidate_mae=0,baseline_mae=0,advantage=[0]*len(x[0]))),(2,lambda *x:'PROVEN')]:
 a=args.copy();a[index]=wrong;fails(lambda a=a:replay195(Q,m,*a))
# The interval learner output must affect the report; the verifier checks its semantics.
a=args.copy();a[1]=lambda *x:'ABOVE_MARGIN';wrong=replay195(Q,m,*a)
assert wrong!=r and wrong['regression']['test']['zero_margin_verdict']!='UNRESOLVED'
with tempfile.TemporaryDirectory() as tmp:
 q=Path(tmp)/'packet';shutil.copytree(Q,q)
 p=q/'l149/gnn/0/predictions.npz';p.write_bytes(p.read_bytes()+b'changed');fails(lambda:replay195(q,m,*args));p.unlink();fails(lambda:replay195(q,m,*args))
assert replay195(Q,m,*args)==r
result=dict(status='PASS',independently_scored_predictions=rows,regression_oracle='SQLite key joins and AVG ABS',auc_oracle='Positive-negative pair comparisons with half-credit ties',bootstrap_oracle='Explicit driver blocks, 2000 original draws seed137',independent_paper_rows=len(paper_rows),decimal_sign_counts=counts,margin_states=231,rejected_invalid_cases=rejected,learner_mutations=3,full_report_parity='EXACT')
(P/'_verify_l195_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
