"""Download and independently reconstruct all metrics from authenticated predictions."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from _run_l108 import fingerprint,sha,ARMS
P=Path(__file__).resolve().parent;R=P.parent

def ap(p,n):return float(average_precision_score(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n]))
def stats(values):
 a=np.asarray(values,float);return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)),'values':a.tolist()}
def collect(download=False):
 digest=fingerprint();root=P/'evidence/l108';root.mkdir(parents=True,exist_ok=True)
 if download:subprocess.run([str(R/'.venv/bin/modal'),'volume','get','l108-sampling-evidence',f'/{digest}/full',str(root)+'/', '--force'],check=True)
 records=[];hashes={};per={a:{l:[] for l in ['all','new']} for a in ARMS};release={l:[] for l in ['all','new']};maxerr=0.;total=0
 pins=json.loads((P/'_inputs_l108.json').read_text())
 for seed in range(10):
  folder=root/'full'/f'seed-{seed}';r=json.loads((folder/'result.json').read_text());assert r['status']=='COMPLETE'
  assert r['identity']['fingerprint']==digest and r['identity']['seed']==seed and r['identity']['checkpoint']==pins['seeds'][str(seed)]['selected.pt']
  assert r['predictions_sha256']==sha(folder/'predictions.npz')
  records.append(r)
  for file in ['predictions.npz','result.json']:hashes[f'full/seed-{seed}/{file}']=sha(folder/file)
  with np.load(folder/'predictions.npz') as z:
   for lane,expected in [('all',23620),('new',11714)]:
    assert len(z[f'release_{lane}_e'])==expected;total+=expected
    batch=z[f'release_{lane}_batch'];vals=[]
    for b in np.unique(batch):
     mask=batch==b;vals.append(ap(z[f'release_{lane}_p'][mask],z[f'release_{lane}_n'][mask]))
    released=float(np.mean(vals));assert abs(released-r['release'][lane]['batch_mean_ap'])<1e-12
    release[lane].append(100*released)
    maxerr=max(maxerr,r['release'][lane]['max_prediction_error'])
    baseline=ap(z[f'uniform20_{lane}_p'],z[f'uniform20_{lane}_n'])
    for arm in ARMS:
     for field in ['e','negative','batch']:np.testing.assert_array_equal(z[f'{arm}_{lane}_{field}'],z[f'release_{lane}_{field}'])
     p,n=z[f'{arm}_{lane}_p'],z[f'{arm}_{lane}_n'];assert len(p)==expected and np.isfinite(p).all() and np.isfinite(n).all()
     measured=ap(p,n);m=r['metrics'][arm][lane]
     assert abs(measured-m['changed_ap'])<1e-12 and abs(100*(measured-baseline)-m['delta_pp'])<1e-10
     assert m['audit']['nonpast_records']==m['audit']['expired_records']==0
     per[arm][lane].append({'ap':100*measured,'delta_pp':100*(measured-baseline),'seconds':m['seconds'],'questions_per_second':m['events_per_second'],'cuda_peak_mib':m['cuda_peak_allocated_bytes']/2**20})
 summary={arm:{lane:{key:stats([x[key] for x in rows]) for key in rows[0]} for lane,rows in lanes.items()} for arm,lanes in per.items()}
 rel={lane:{**stats(vals),'target':{'all':95.34,'new':93.99}[lane],'status':'CLOSE' if abs(np.mean(vals)-{'all':95.34,'new':93.99}[lane])<=.5 else 'NOT_CLOSE'} for lane,vals in release.items()}
 result={'status':'COMPLETE','seeds':list(range(10)),'fingerprint':digest,'release':rel,'summary':summary,'release_prediction_max_error':maxerr,'positive_questions_across_populations_and_seeds':total,'intervention_positive_questions':4*total,'artifacts':hashes,'seconds_total':sum(r['elapsed_seconds'] for r in records),'index_array_bytes':records[0]['index_array_bytes'],'metric_reconstruction':'PASS','full_paper':'NOT_ESTABLISHED','historical_identity':'INCOMPARABLE','fresh_training':'NOT_RUN_IN_L108','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
 (P/'_analysis_l108_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['artifacts','summary']},indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');a=p.parse_args();collect(a.download)
