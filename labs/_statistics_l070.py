"""Reconstruct L070 hybrid panels and dataset-unit uncertainty from raw records."""
import itertools,json
from pathlib import Path
import numpy as np
from scipy.stats import chi2,studentized_range
from _archive_l070 import ROOT,digest

def ranks(x):
    return np.array([1+np.sum(x<v)+(np.sum(x==v)-1)/2 for v in x])

def check():
    path=ROOT/'_verify_l070_v2_results.json';r=json.loads(path.read_text());old=json.loads((ROOT/'_verify_l070_results.json').read_text())['records'];checks=[]
    selected=[v for v in old if v['dataset'] in r['config']['datasets'] and v['seed'] in r['config']['seeds']];new=[v for v in r['records'] if v['arm']=='TabM-mini-v2']
    for key,records in [('summary',[v for v in selected if v['arm']!='TabM-mini']+new),('historical_summary',selected),('legacy_diagnostic',selected+new)]:
        s=r[key];values=np.array([[[next(v['error']for v in records if (v['dataset'],v['arm'],v['seed'])==(d,a,k))for k in s['seeds']]for a in s['arms']]for d in s['datasets']]);means=values.mean(2);rr=np.stack([ranks(row)for row in means]);n,k=rr.shape
        np.testing.assert_array_equal(values,s['values']);np.testing.assert_allclose(means,s['means'],atol=1e-14);np.testing.assert_allclose(rr,s['dataset_ranks'],atol=1e-14)
        np.testing.assert_allclose(rr.mean(0),[s['mean_ranks'][a]for a in s['arms']],atol=1e-14)
        if len(s['seeds'])>1:np.testing.assert_allclose(values.std(2,ddof=1),s['sample_sd'],atol=1e-14)
        ties=sum(sum(c**3-c for c in np.unique(row,return_counts=True)[1])for row in means);correction=1-ties/(n*(k**3-k));q=(12/(n*k*(k+1))*np.sum(rr.sum(0)**2)-3*n*(k+1))/correction
        if n>=3 and k>=3:assert abs(chi2.sf(q,k-1)-s['friedman_p'])<1e-13
        cd=studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n));assert abs(cd-s['nemenyi_cd'])<1e-13
        checks.append(dict(panel=key,datasets=n,arms=k,friedman_statistic=q,asymptotic_p=s['friedman_p'],nemenyi_cd=cd))
    s=r['summary'];values=np.asarray(s['values']);bootstrap=[]
    for i,arm in enumerate(s['arms'][1:],1):
        effect=r['paired_effects'][arm];gaps=(values[:,i,:]-values[:,0,:]).mean(1);n=len(gaps)
        random=np.random.default_rng(effect['seed']);mc=np.mean(gaps[random.integers(0,n,size=(effect['repetitions'],n))],axis=1)
        np.testing.assert_allclose(gaps,effect['dataset_means'],atol=1e-14);assert abs(gaps.mean()-effect['mean'])<1e-14;np.testing.assert_allclose(np.quantile(mc,[.025,.975]),effect['percentile95'],atol=1e-14)
        exact=np.array([np.mean(gaps[list(draw)])for draw in itertools.product(range(n),repeat=n)])
        bootstrap.append(dict(arm=arm,mean_gap=float(gaps.mean()),monte_carlo95=effect['percentile95'],exact_empirical95=np.quantile(exact,[.025,.975],method='inverted_cdf').tolist(),exact_resamples=len(exact)))
    report=dict(status='PASS',evidence_sha256=digest(path),archive_sha256=digest(ROOT/'_verify_l070_results.json'),checker_sha256=digest(__file__),panels=checks,paired_bootstraps=bootstrap,scope='Reconstructed main seven-arm, historical seven-arm and diagnostic eight-arm panels from raw selected records; manual midranks and tie-corrected Friedman statistic, independent critical-distance formula, paired Monte Carlo replay and all 3125 empirical five-dataset bootstrap draws. Exact finite-distribution and Monte Carlo quantiles are separately labeled; neither establishes population coverage.')
    (ROOT.parent/'reviews/lesson-quality-audit-047-070/070-statistics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return report

if __name__=='__main__':check()
