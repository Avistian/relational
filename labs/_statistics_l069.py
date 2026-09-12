"""Independent dataset-unit reconstruction of L069 summary statistics."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def check():
 evidence=ROOT/'_verify_l069_v2_results.json';r=json.loads(evidence.read_text());a=json.loads((ROOT/'_analysis_l069_v2_results.json').read_text());gaps=[];ranks=[];details=[]
 for dataset in ['cmc','winequality-red','winequality-white']:
  means=[]
  for arm in ['v2','xgboost']:
   seed_means=[]
   for seed in [42,2023,789]:
    values=[v['auc'] for v in r['records'] if v['axis']=='novelty' and v['condition']=='leave-one-class-out' and v['dataset']==dataset and v['arm']==arm and v['seed']==seed];seed_means.append(float(np.mean(values)))
   means.append(float(np.mean(seed_means)));details.append(dict(dataset=dataset,arm=arm,seed_means=seed_means))
  gaps.append(means[0]-means[1]);ranks.append([1.,2.] if means[0]>means[1] else [2.,1.] if means[0]<means[1] else [1.5,1.5])
 np.testing.assert_allclose(gaps,a['continuous_auc_dataset_gaps'],atol=1e-14);np.testing.assert_allclose(np.mean(ranks,axis=0),a['mean_ranks'],atol=1e-14)
 exact=np.array([np.mean([gaps[i] for i in sample]) for sample in itertools.product(range(3),repeat=3)]);interval=np.quantile(exact,[.025,.975],method='inverted_cdf');np.testing.assert_allclose(interval,a['dataset_bootstrap95'],atol=1e-14)
 for summary in a['feature_summary']:
  base=[v for v in r['records'] if v['axis']=='features' and v['dataset']==summary['dataset'] and v['arm']==summary['arm'] and v['condition']=='0%'];shift=[v for v in r['records'] if v['axis']=='features' and v['dataset']==summary['dataset'] and v['arm']==summary['arm'] and v['condition']=='100%']
  for field,metric in [('clean','accuracy'),('balanced_accuracy','balanced_accuracy'),('macro_f1','macro_f1')]:assert abs(summary[field]-np.mean([v[metric] for v in base]))<1e-12
  auc=[v['auc'] for v in base if v['auc'] is not None];assert summary['auc_n']==len(auc) and abs(summary['auc']-np.mean(auc))<1e-12
  assert abs(summary['all_removed']-np.mean([v['accuracy'] for v in shift]))<1e-12
 report=dict(status='PASS',evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),analysis_sha256=hashlib.sha256((ROOT/'_analysis_l069_v2_results.json').read_bytes()).hexdigest(),datasets=3,exact_bootstrap_samples=27,exact_bootstrap95=interval.tolist(),dataset_gaps=gaps,details=details,scope='Reconstructed class-then-seed means from raw records, two-arm dataset ranks, all 27 paired dataset bootstrap draws and clean/shifted objective summaries including available-AUC seed counts; descriptive small-panel uncertainty.')
 (ROOT.parent/'reviews/lesson-quality-audit-047-070/069-statistics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':check()
