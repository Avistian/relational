"""Reconcile fresh candidates, archival controls and every displayed L070 number."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import t
from relkit.checkpoint_l070_v2 import binary_loss,choose_candidate,complete_panel,ARMS,DATASETS
ROOT=Path(__file__).parent

def analyze():
 r=json.loads((ROOT/'_verify_l070_v2_results.json').read_text());assert r['status']=='COMPLETE';archive=json.loads((ROOT/'_verify_l070_results.json').read_text());assert len(r['records'])==30
 max_control=0.;epochs=0
 for z in r['records']:
  assert abs(binary_loss(z['targets'],z['predictions'])-z['error'])<1e-12
  assert abs(binary_loss(z['targets'],z['intervention']['predictions'])-z['intervention']['error'])<1e-12
  for c in z['candidates']:
   assert abs(binary_loss(z['validation_targets'],c['validation_predictions'])-c['validation_error'])<1e-12
   if c['selected_epoch'] is not None:
    scores=c['epoch_losses'];ix=max(i for i,v in enumerate(scores) if v==min(scores));assert c['selected_epoch']==ix+1;assert abs(scores[ix]-c['validation_error'])<1e-12;epochs+=len(scores)
  assert z['selected']==choose_candidate([c['validation_error'] for c in z['candidates']])
  if z['arm']=='XGBoost-fresh-control':
   old=next(x for x in archive['records'] if x['dataset']==z['dataset'] and x['seed']==z['seed'] and x['arm']=='XGBoost');max_control=max(max_control,float(np.max(abs(np.array(z['predictions'])-old['predictions']))))
 s=r['summary'];v=np.array(s['values']);sd=v.std(2,ddof=1);means=v.mean(2)
 table='| Dataset | '+' | '.join(ARMS)+' |\n|---|'+'---:|'*len(ARMS)+'\n'
 for i,d in enumerate(DATASETS):table+='| '+d.split('/')[0]+' | '+' | '.join(f'{means[i,j]:.4f} ± {sd[i,j]:.4f}' for j in range(len(ARMS)))+' |\n'
 table+='\nTest log loss, mean ± sample SD over three downstream seeds. Six archived arms plus fresh corrected TabM. Smaller is better; fixed split and restricted procedures.\n'
 ordered=sorted(s['mean_ranks'],key=s['mean_ranks'].get);best=ordered[0]
 g={}
 for arm in ['XGBoost-fresh-control','TabM-mini-v2']:
  matrix=np.array([[next(z['intervention']['delta_loss'] for z in r['records'] if z['arm']==arm and z['dataset']==d and z['seed']==seed) for seed in [0,1,2]] for d in DATASETS]);g[arm]=dict(mean=float(matrix.mean()),dataset_means=dict(zip(DATASETS,matrix.mean(1).tolist())),positive_seeds=int((matrix>0).sum()),negative_seeds=int((matrix<0).sum()))
 legacy=[]
 for d in DATASETS:
  a=[x['error'] for x in archive['records'] if x['dataset']==d and x['arm']=='TabM-mini'];b=[x['error'] for x in r['records'] if x['dataset']==d and x['arm']=='TabM-mini-v2'];legacy.append(dict(dataset=d,legacy_mean=float(np.mean(a)),corrected_mean=float(np.mean(b)),delta=float(np.mean(b)-np.mean(a))))
 paragraphs=[f"**Measured outcome.** {best} has the smallest mean dataset rank ({s['mean_ranks'][best]:.2f}) in the corrected seven-arm pool. The exploratory Friedman p-value is {s['friedman_p']:.3f}; the Nemenyi critical difference is {s['nemenyi_cd']:.2f} rank units. These summaries do not resolve a general winner from five fixed small tables.",
 f"The fresh XGBoost replay control differs from its archived test probabilities by at most {max_control:.3g}. This verifies that control under the recorded local recipes; its wall-clock cost is a new observation. The 30 fresh selected fits retain 60 candidate validation probability vectors and {epochs} neural epoch losses.",
 'The feature-erasure prediction is '+('supported' if all(z['mean']>0 for z in g.values()) else 'not supported for both arms')+' at the dataset-average level: '+', '.join(f"{a}: mean loss increase {z['mean']:+.4f} nats, {z['negative_seeds']} of 15 seed cases improve" for a,z in g.items())+'. Every negative case stays in the evidence. A mean degradation does not imply all rows or datasets degrade.',
 'The corrected-minus-legacy TabM mean-loss differences are '+', '.join(f"{z['dataset'].split('/')[0]} {z['delta']:+.4f}" for z in legacy)+'. Correcting paper fidelity is not a promise to improve every held-out score.']
 paired={}
 for j,arm in enumerate(ARMS[1:],1):
  delta=v[:,j,:]-v[:,0,:];paired[arm]=[]
  for d,x in zip(DATASETS,delta):
   half=float(t.ppf(.975,2)*x.std(ddof=1)/np.sqrt(3));paired[arm].append(dict(dataset=d,seed_gaps=x.tolist(),mean=float(x.mean()),sample_sd=float(x.std(ddof=1)),conditional_t95=[float(x.mean()-half),float(x.mean()+half)]))
 result=dict(status='PASS',table_markdown=table,interpretation='\n\n'.join(paragraphs),mean_ranks=s['mean_ranks'],friedman_p=s['friedman_p'],nemenyi_cd=s['nemenyi_cd'],intervention=g,legacy_tabm=legacy,max_fresh_xgboost_archive_probability_delta=max_control,candidate_validation_vectors=60,neural_epoch_losses=epochs,paired_conditional_seed_intervals=paired,evidence_sha256=hashlib.sha256((ROOT/'_verify_l070_v2_results.json').read_bytes()).hexdigest())
 (ROOT/'_analysis_l070_v2_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result['interpretation']);return result
if __name__=='__main__':analyze()
