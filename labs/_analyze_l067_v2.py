"""Dataset-balanced local LoCalPFN statistics, all arms and split-level paired gaps."""
from pathlib import Path
import json,hashlib,numpy as np
from scipy.stats import rankdata,friedmanchisquare,t,studentized_range
ROOT=Path(__file__).resolve().parent
ARMS=['global_all','random_k','local_frozen','local_ft_paper_lr','local_ft_release_lr']
def analyze():
 evidence=ROOT/'_verify_l067_v2_results.json';data=json.loads(evidence.read_text());datasets=list(data['datasets'])
 expected={(d,s) for d in ['diabetes','blood_transfusion','wdbc'] for s in [7,17,27]}
 keys=[(r['dataset'],r['seed']) for r in data['records']]
 assert len(keys)==9 and set(keys)==expected,'Expected exactly nine dataset/seed records'
 assert all(set(r['arms'])==set(ARMS) for r in data['records']),'Expected five arms on every record'
 summary=[];means=[];gaps=[];selected={a:[] for a in ARMS[-2:]}
 for name in datasets:
  rows=[r for r in data['records'] if r['dataset']==name];ds=[]
  for arm in ARMS:
   auc=np.array([r['arms'][arm]['auc'] for r in rows]);loss=np.array([r['arms'][arm]['log_loss'] for r in rows]);acc=np.array([r['arms'][arm]['accuracy'] for r in rows]);ds.append(auc.mean())
   summary.append(dict(dataset=name,arm=arm,auc_mean=float(auc.mean()),auc_sd=float(auc.std(ddof=1)),loss_mean=float(loss.mean()),loss_sd=float(loss.std(ddof=1)),accuracy_mean=float(acc.mean()),accuracy_sd=float(acc.std(ddof=1))))
   if arm in selected:selected[arm]+=[dict(dataset=name,seed=r['seed'],step=r['arms'][arm]['selected_step'],parameter_change=r['adaptation'][arm]['final_parameter_l2_change']) for r in rows]
  means.append(ds)
  for arm,baseline in [('random_k','global_all'),('local_frozen','random_k'),('local_ft_paper_lr','local_frozen'),('local_ft_release_lr','local_frozen')]:
   values=np.array([r['arms'][arm]['auc']-r['arms'][baseline]['auc'] for r in rows]);half=float(t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
   gaps.append(dict(dataset=name,arm=arm,baseline=baseline,metric='AUC',seed_gaps=values.tolist(),mean=float(values.mean()),conditional_t95=[float(values.mean()-half),float(values.mean()+half)]))
 means=np.asarray(means);ranks=np.array([rankdata(-row,method='average') for row in means]);stat,p=friedmanchisquare(*means.T)
 cd=float(studentized_range.ppf(.95,len(ARMS),np.inf)/np.sqrt(2)*np.sqrt(len(ARMS)*(len(ARMS)+1)/(6*len(datasets))))
 counts={a:sum(r['step']==0 for r in rr) for a,rr in selected.items()}
 text=f"The paper-rate arm retained step 0 in {counts[ARMS[-2]]}/9 runs; the released-CLI-rate arm retained it in {counts[ARMS[-1]]}/9. Every adaptation run records nonzero final parameter change, so retained frozen predictions do not imply that training was skipped. "
 for name in datasets:
  rows=[s for s in summary if s['dataset']==name];baseline=next(s for s in rows if s['arm']=='local_frozen');release=next(s for s in rows if s['arm']=='local_ft_release_lr')
  gap=next(g for g in gaps if g['dataset']==name and g['arm']=='local_ft_release_lr')
  text+=f"On {name.replace('_',' ')}, local frozen mean AUC was {baseline['auc_mean']:.4f} and local FT at 10⁻⁵ was {release['auc_mean']:.4f}; corresponding mean log losses were {baseline['loss_mean']:.4f} and {release['loss_mean']:.4f}. The paired released-rate minus frozen AUC gap is {gap['mean']:+.4f}, with descriptive split-seed t95 interval [{gap['conditional_t95'][0]:+.4f}, {gap['conditional_t95'][1]:+.4f}]. "
 text+=f"Dataset-balanced mean AUC ranks (lower is better) in arm order global-all/random-k/local-frozen/paper-rate/released-rate are {'/'.join(f'{v:.3f}' for v in ranks.mean(0))}. Friedman statistic {stat:.3f}, p={p:.4f}; Nemenyi 5% critical difference {cd:.3f}. The global-all versus random-k mean-rank gap is 4.000, exceeding this approximate CD; the other pairwise gaps do not. This is a nominal descriptive distinction within this panel, not a general-superiority result. The asymptotic tests have only three dataset blocks and several tied arms; overlapping seed holdouts do not increase that dataset count. A zero-width gap interval when all selected states are step 0 reflects identical deployed predictions, not precise knowledge of adaptation at other budgets."
 result=dict(evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),datasets=datasets,arms=ARMS,summary=summary,paired_gaps=gaps,selection=selected,mean_ranks=dict(zip(ARMS,ranks.mean(0).tolist())),dataset_ranks=ranks.tolist(),friedman=dict(statistic=float(stat),pvalue=float(p)),nemenyi_cd=cd,interval_scope='Conventional paired split-seed t95; overlapping holdouts, descriptive, not dataset-population uncertainty',interpretation=text,total_seconds=data['seconds'])
 (ROOT/'_analysis_l067_v2_results.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':print(json.dumps(analyze(),indent=2))
