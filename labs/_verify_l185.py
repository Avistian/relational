"""Independent enumerated population and saved-row audit; fresh complete rerun."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from relkit.causal_l185 import full_experiment
P=Path(__file__).resolve().parent;E=P/'evidence/l185'

def independent_auc(y,s):
 n=int(y.sum());m=len(y)-n
 return float((rankdata(s)[y==1].sum()-n*(n+1)/2)/(n*m))

def population():
 # Enumerate exact binary states independently of the generator and estimator.
 rows=[]
 for u,b,a in itertools.product((0,1),repeat=3):
  mass=.5*(.95 if b==u else .05)*((.9 if u else .1) if a else (.1 if u else .9))
  risk={(0,0):.05,(0,1):.10,(1,0):.90,(1,1):.95}[u,a]
  rows.append({'U':u,'B':b,'A':a,'mass':mass,'risk':risk})
 f=pd.DataFrame(rows)
 def conditional(col,value):
  q=f[f[col]==value];return float((q.mass*q.risk).sum()/q.mass.sum())
 return {'badge0_observed':conditional('B',0),'badge1_observed':conditional('B',1),
         'naive_action_difference':conditional('A',1)-conditional('A',0),
         'badge_do_risk':float((f.mass*f.risk).sum()),'action_ate':.05,'badge_ate':0.0}

if __name__=='__main__':
 report=json.loads((E/'report.json').read_text())
 for name,digest in json.loads((E/'run-manifest.json').read_text()).items():assert hashlib.sha256((P.parent/name).read_bytes()).hexdigest()==digest,name
 truth=population()
 assert np.isclose(truth['badge1_observed'],.9005) and np.isclose(truth['badge0_observed'],.0995)
 assert np.isclose(truth['badge_do_risk'],.5) and np.isclose(truth['naive_action_difference'],.73)
 predictions_checked=0
 for seed in range(5):
  d=E/f'seed-{seed}';c=pd.read_parquet(d/'companies.parquet');q=pd.read_parquet(d/'customers.parquet')
  a=pd.read_parquet(d/'actions.parquet');y=pd.read_parquet(d/'outcomes.parquet')
  f=q.join(c.set_index('company_id'),on='company_id').join(a.set_index('customer_id'),on='customer_id').join(y.set_index('customer_id'),on='customer_id')
  assert len(f)==20000 and f.customer_id.nunique()==20000
  assert f.groupby('company_id').split.nunique().eq(1).all()
  assert f.groupby('split').company_id.nunique().to_dict()=={'test':200,'train':600,'validation':200}
  assert (f.available_at<=f.cutoff).all() and (f.action_at<=f.cutoff).all() and (f.outcome_at>f.cutoff).all()
  pred=pd.read_parquet(d/'predictions.parquet');pot=pd.read_parquet(d/'potentials.parquet')
  row=report['per_seed'][seed]
  for (split,model),g in pred.groupby(['split','model']):
   expected=f[f.split==split].sort_values('customer_id');g=g.sort_values('customer_id')
   assert list(zip(g.customer_id,g.cutoff))==list(zip(expected.customer_id,expected.cutoff))
   np.testing.assert_array_equal(g.Y,expected.Y)
   cols={'badge':['B'],'action':['A'],'demand_action':['U','A']}[model]
   train=f[f.split=='train'];lookup={}
   for values in expected[cols].drop_duplicates().itertuples(index=False,name=None):
    mask=np.ones(len(train),dtype=bool)
    for col,value in zip(cols,values):mask &= train[col].to_numpy()==value
    lookup[values]=float(train.loc[mask,'Y'].mean()) if mask.any() else float(train.Y.mean())
   scores=[lookup[v] for v in expected[cols].itertuples(index=False,name=None)]
   np.testing.assert_array_equal(g.score,scores)
   assert abs(independent_auc(g.Y.to_numpy(),g.score.to_numpy())-row[split+'_'+model+'_auroc'])<1e-12
   predictions_checked+=len(g)
  test=f[f.split=='test'].sort_values('customer_id');pot=pot.sort_values('customer_id')
  assert list(zip(test.customer_id,test.cutoff))==list(zip(pot.customer_id,pot.cutoff))
  for name in ['badge0','badge1']:np.testing.assert_array_equal(pot[name],test.Y)
  np.testing.assert_array_equal(pot.action0,(test.E<np.where(test.U==1,.90,.05)).astype(int))
  np.testing.assert_array_equal(pot.action1,(test.E<np.where(test.U==1,.95,.10)).astype(int))
  means={}
  for u in [0,1]:
   stratum=test[test.U==u];means[u]=(stratum[stratum.A==1].Y.mean()-stratum[stratum.A==0].Y.mean())*len(stratum)/len(test)
  assert abs(sum(means.values())-row['adjusted_action_effect'])<1e-12
  assert row['paired_badge_effect']==0 and row['badge1_policy_gain']==0
  # Tolerances frozen before run: sampling tolerance, not paper-score tolerance.
  assert abs(row['paired_action_effect']-.05)<.025
  assert abs(row['adjusted_action_effect']-.05)<.025
 outputs,fresh=full_experiment();assert fresh==report
 for seed,(tables,pred,pot,_) in enumerate(outputs):
  for name,table in {**tables,'predictions':pred,'potentials':pot}.items():pd.testing.assert_frame_equal(table.reset_index(drop=True),pd.read_parquet(E/f'seed-{seed}'/(name+'.parquet')),check_exact=True)
 result={'status':'PASS','prediction_rows_checked':predictions_checked,'paired_customers_checked':20000,'fresh_complete_rerun':'EXACT','population':truth}
 (P/'_verify_l185_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
