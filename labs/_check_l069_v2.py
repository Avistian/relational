"""Focused operator checks for live L069 TODOs and error accounting."""
import json
from pathlib import Path
import numpy as np
from relkit import openenv_l069_v2 as c
from _build_l069 import CHECKS
ROOT=Path(__file__).resolve().parent

def check(namespace=None):
 s=vars(c) if namespace is None else namespace
 for name,code in CHECKS.items():exec(code,s)
 # A present context label absent from the test makes macro OVR AUC unavailable.
 r=s['metric_record'](np.array([1,2,1,2]),np.array([[.1,.8,.1],[.1,.1,.8],[.1,.6,.3],[.1,.3,.6]]),[0,1,2]);assert r['auc'] is None
 rows=[dict(axis='f',dataset='A',arm='v2',condition='0',seed=i,auc=v,accuracy=.7) for i,v in [(42,.8),(2023,.6),(789,None)]]
 summary=s['dataset_summary'](rows)[0];assert summary['auc_n']==2 and summary['auc_seed_ids']==[42,2023] and summary['accuracy_n']==3
 p=np.array([[.2,.8],[.9,.1]]);a=s['all_row_metrics']([20,30],p,[10,20]);assert a['unsupported_n']==1 and a['n']==2
 augmented=np.array([[3.,99.,4.]]);np.testing.assert_array_equal(s['align_schema'](augmented,['A','new','B'],['A','B']),[[3,4]])
 for available,training in [(['A','A'],['A']),(['A'],['A','B'])]:
  try:s['align_schema'](np.ones((2,len(available))),available,training)
  except ValueError:pass
  else:raise AssertionError('Ambiguous schema accepted')
 return dict(status='PASS',six_live_tasks=list(CHECKS),extra_checks=['undefined multiclass AUC uses None','per-metric seed IDs/counts','unsupported class map and full denominator','incremental feature reorder/ignore','duplicate and missing schema rejected'],kernel_identity=s['kernel_identity'](s,ROOT)['sha256'])
if __name__=='__main__':
 r=check();(ROOT/'_check_l069_v2_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
