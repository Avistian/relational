"""Recompute local panel summaries/ranks with dataset, never seed, as unit."""
from pathlib import Path
import json,numpy as np
from scipy.stats import rankdata,friedmanchisquare,studentized_range
from relkit import driftpfn_l068_v2 as c
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 result=json.loads((ROOT/'_verify_l068_v2_results.json').read_text());summary=c.dataset_summary(result['records']);assert summary==result['summary']
 arms=['base_no_time','base_time','drift','drift_zero','noT2V'];datasets=['electricity','parking','chess'];stats={}
 for metric,sign in [('accuracy',-1),('auc',-1),('log_loss',1)]:
  matrix=np.array([[next(r[metric+'_mean'] for r in summary if (r['dataset'],r['arm'],r['split'])==(d,a,'ood')) for a in arms] for d in datasets]);ranks=np.array([rankdata(sign*r) for r in matrix]);test=friedmanchisquare(*matrix.T)
  stats[metric]=dict(datasets=datasets,arms=arms,values=matrix.tolist(),ranks=ranks.tolist(),mean_ranks=ranks.mean(0).tolist(),friedman_statistic=float(test.statistic),friedman_p=float(test.pvalue),nemenyi_cd=float(studentized_range.ppf(.95,len(arms),np.inf)/np.sqrt(2)*np.sqrt(len(arms)*(len(arms)+1)/(6*len(datasets)))))
 def get(d,a,m):return next(r[m+'_mean'] for r in summary if (r['dataset'],r['arm'],r['split'])==(d,a,'ood'))
 gaps=[get(d,'drift','log_loss')-get(d,'base_time','log_loss') for d in datasets]
 rng=np.random.default_rng(68);boot=np.mean(rng.choice(gaps,size=(10000,3),replace=True),1)
 interpretation=('The complete-model panel is mixed on real data. On Chess, drift worsens mean OOD accuracy from base-with-time 0.7158 to 0.6762 and log loss from 0.7033 to 0.7721. On Parking it raises accuracy from 0.5101 to 0.6644 and lowers loss from 1.0233 to 0.8757, but zeroing time improves the drift checkpoint further (accuracy 0.6866, loss 0.8521): clock sensitivity is not uniformly beneficial. Electricity illustrates metric disagreement: drift slightly improves accuracy (0.6456 to 0.6616) and log loss (0.6135 to 0.6096), while AUC falls from 0.7492 to 0.7467. Blobs is much stronger evidence for extrapolating this synthetic mechanism: drift accuracy 0.9487 versus base-with-time 0.5961; zero-time falls to 0.4448. The separately pretrained NoT2V checkpoint reaches 0.9908 on Blobs, supporting the distinction between the temporal prior and Time2Vec without isolating pretraining variance. These are local matched-wrapper results, not the paper benchmark. Across three real datasets the mean drift-minus-base-time log-loss gap is '+f'{np.mean(gaps):.4f}'+', with a coarse dataset-bootstrap 95% percentile interval '+str([round(float(x),4) for x in np.quantile(boot,[.025,.975])])+'. Three datasets cannot support a broad superiority conclusion; the bootstrap cannot invent unseen tasks.')
 out=dict(status='PASS',summary=summary,stats=stats,paired_real_loss_gap=dict(datasets=datasets,values=gaps,mean=float(np.mean(gaps)),dataset_bootstrap_percentile95=np.quantile(boot,[.025,.975]).tolist(),draws=10000,seed=68),interpretation=interpretation,seconds=result['seconds'],paper_reproduction='INCOMPARABLE')
 (ROOT/'_analysis_l068_v2_results.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(interpretation);print(stats)
