"""Independent scalar score reconstruction, source parity and adversarial contracts."""
import ast,copy,csv,hashlib,json,logging,math,textwrap
from pathlib import Path
from typing import *
import numpy as np
from scipy import fft
from scipy.signal import find_peaks
from scipy.stats import rankdata,gmean
from relkit.forecast_b19b import source_periods,source_seasonal_features
from _reproduce_b19b import replay
P=Path(__file__).resolve().parent;E=P/'evidence/b19b';S=P/'sources/b19b'
r=json.loads((E/'diagnostic.json').read_text());n=0
assert len(r['predictions'])==324 and len(r['scores'])==27
assert len({(x['seed'],x['origin'],x['time'],x['arm']) for x in r['predictions']})==324
for s in r['scores']:
 y=r['series'][s['seed']]['target'];h=y[:s['origin']];p=[x for x in r['predictions'] if all(x[k]==s[k] for k in ['seed','origin','arm'])]
 assert [x['time'] for x in p]==list(range(s['origin'],s['origin']+12))
 scale=sum(abs(a-b) for a,b in zip(h[12:],h[:-12]))/(len(h)-12)
 mase=sum(abs(x['y']-x['quantiles'][4]) for x in p)/12/scale
 wql=sum(2*max(q*(x['y']-v),(q-1)*(x['y']-v)) for x in p for q,v in zip(r['levels'],x['quantiles']))/9/sum(abs(x['y']) for x in p)
 assert math.isclose(mase,s['mase'],rel_tol=1e-12) and math.isclose(wql,s['wql'],rel_tol=1e-12)
 assert all(x['y']==y[x['time']] for x in p)
 if s['arm']=='seasonal_naive':assert all(x['quantiles'][4]==y[x['time']-12] for x in p)
 n+=1
records=json.loads((E/'released-scores.json').read_text());report=json.loads((E/'reproduction.json').read_text());keys=sorted(records['Seasonal-Naive']);models=list(records)
for row in report['rows']:
 for metric in ['mase','wql']:
  value=float(gmean([records[row['model']][k][metric]/records['Seasonal-Naive'][k][metric] for k in keys]))
  assert math.isclose(value,row['relative_'+metric],rel_tol=1e-12)
 rank=float(np.mean([rankdata([records[m][k]['wql'] for m in models])[models.index(row['model'])] for k in keys]))
 assert math.isclose(rank,row['mean_wql_rank'],rel_tol=1e-12)
# Original CSV -> every compact row, not only aggregate consistency.
for f in report['files']:
 for x in csv.DictReader((S/f['path']).open()):
  assert records[f['model']][x['dataset']]==dict(mase=float(x['eval_metrics/MASE[0.5]']),wql=float(x['eval_metrics/mean_weighted_sum_quantile_loss']))
for mutate in ['missing_task','missing_model','nan','zero']:
 bad=copy.deepcopy(records)
 if mutate=='missing_task':bad['TiRex'].pop(keys[0])
 elif mutate=='missing_model':bad.pop('TiRex')
 elif mutate=='nan':bad['TiRex'][keys[0]]['wql']=float('nan')
 else:bad['TiRex'][keys[0]]['wql']=0
 try:replay(bad)
 except ValueError:pass
 else:raise AssertionError(mutate)
# Execute the archived author's seasonal extraction function, without importing its model/API dependencies.
source=(S/'wrapper/tabpfn_time_series/features/auto_features.py').read_text();tree=ast.parse(source)
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef));method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='find_seasonal_periods');detrend=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='detrend')
ns=dict(np=np,fft=fft,find_peaks=find_peaks,logger=logging.getLogger('source'))
exec('from __future__ import annotations\n'+ast.get_source_segment(source,detrend)+'\n'+textwrap.dedent(ast.get_source_segment(source,method)),ns)
comparisons=[]
for label,h in [('season12',np.sin(2*np.pi*np.arange(96)/12)),('two_periods',np.sin(2*np.pi*np.arange(96)/12)+.3*np.sin(2*np.pi*np.arange(96)/8)),('constant',np.ones(96)),('course',np.array(r['series'][0]['target'][:96]))]:
 expected=ns['find_seasonal_periods'](h,max_top_k=5,detrend_type='linear',exclude_zero=True)
 actual=source_periods(h);assert np.allclose(actual,expected,rtol=1e-12,atol=1e-12)
 train,test,periods=source_seasonal_features(h,12);assert train.shape==(96,10) and test.shape==(12,10)
 for i,p in enumerate(periods):assert np.allclose(test[:,2*i],np.sin(2*np.pi*np.arange(96,108)/p))
 comparisons.append(dict(case=label,periods=periods,shape=[list(train.shape),list(test.shape)]))
out=dict(status='PASS',scalar_scored_conditions=n,complete_keyed_predictions=324,original_csv_records=1261,independent_aggregate_fields=39,rejected_score_corruptions=4,source_seasonal_parity=comparisons,source_scope='Default linear detrend, Hann multiplication, right padding, peak threshold; not full model prediction parity')
(E/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
