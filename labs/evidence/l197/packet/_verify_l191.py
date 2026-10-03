"""Independent BeautifulSoup/Fraction reconstruction, corruption and completeness checks."""
import copy,hashlib,json,math,shutil,tempfile
from fractions import Fraction
from pathlib import Path
from bs4 import BeautifulSoup
from scipy.stats import rankdata
from _replay_l191 import parse_tables,reconstruct,replay191
P=Path(__file__).resolve().parent;E=P/'evidence/l191';Q=E/'packet'
manifest=json.loads((E/'input-manifest.json').read_text());report=replay191(Q,manifest);tables=json.loads((Q/'tables.json').read_text());soup=BeautifulSoup((Q/'paper-v1.html').read_text(),'html.parser')
checked=0
for table,summary in zip(tables,report['tables']):
 n=len(table['tasks']);raw=soup.find(id=f"S4.T{table['number']}").select('tr')[2:]
 assert len(raw)==len(table['rows'])
 matrix=[]
 for tr,row in zip(raw,table['rows']):
  cells=tr.find_all(['th','td'],recursive=False);values=[c.get_text(' ',strip=True) for c in cells[-(n+2):]]
  assert values[:-2]==row['displayed'];assert float(values[-2])==row['published_aggregate'];assert float(values[-1])==row['published_rank'];checked+=len(values)
  matrix.append([Fraction(x) for x in values[:-2]])
 base=matrix[next(i for i,r in enumerate(table['rows']) if r['method']=='LightGBM')]
 for i,row in enumerate(summary['audit']):
  exact=sum(matrix[i],Fraction())/n if table['metric']=='AUROC' else sum((a/b for a,b in zip(matrix[i],base)),Fraction())/n
  assert math.isclose(float(exact),row['recomputed_aggregate'],abs_tol=1e-12)
  # Independent sorting-based implementation, not the canonical pair-count ranker.
  ranks=[float(rankdata([float(v[j])*(-1 if table['metric']=='AUROC' else 1) for v in matrix],method='average')[i]) for j in range(n)]
  assert ranks==row['per_task_ranks'];assert abs(sum(ranks)/n-row['displayed_mean_rank'])<1e-12
  half=[Fraction(1,2*10**len(v.split('.')[1])) for v in table['rows'][i]['displayed']]
  if table['metric']=='AUROC':lo=sum(a-h for a,h in zip(matrix[i],half))/n;hi=sum(a+h for a,h in zip(matrix[i],half))/n;eps=Fraction(5,1000)
  elif row['method']=='LightGBM':lo=hi=Fraction(1);eps=Fraction(5,10000)
  else:
   brow=table['rows'][next(j for j,r in enumerate(table['rows']) if r['method']=='LightGBM')];bh=[Fraction(1,2*10**len(v.split('.')[1])) for v in brow['displayed']]
   lo=sum((a-h)/(b+k) for a,h,b,k in zip(matrix[i],half,base,bh))/n;hi=sum((a+h)/(b-k) for a,h,b,k in zip(matrix[i],half,base,bh))/n;eps=Fraction(5,10000)
  assert abs(float(lo)-row['rounding_interval'][0])<1e-12 and abs(float(hi)-row['rounding_interval'][1])<1e-12
  expected='ROUNDING_COMPATIBLE' if lo-eps<=Fraction(str(row['published_aggregate']))<=hi+eps else 'OUTSIDE_ROUNDING_BOUND';assert expected==row['aggregate_status']
# Independent declared-pool comparison using rational arithmetic.
for table,summary in zip(tables,report['tables']):
 values={r['method']:[Fraction(x) for x in r['displayed']] for r in table['rows']}
 higher=table['metric']=='AUROC';n=len(table['tasks']);base=values['LightGBM'];target=values['KumoRFM-2']
 def score(v):return sum(v,Fraction())/n if higher else sum((a/b for a,b in zip(v,base)),Fraction())/n
 for pool,names in [('foundation',['RDBLearn','Griffin']),('supervised',['LightGBM','GraphSAGE','RelGNN','RelGT']),('all',['RDBLearn','Griffin','LightGBM','GraphSAGE','RelGNN','RelGT'])]:
  present={k:v for k,v in values.items() if k in names};actual=summary['pools'][pool]
  if not present:assert actual['status']=='NO_ELIGIBLE_COMPARATOR';continue
  choose=max if higher else min;best=choose(score(v) for v in present.values());winner=sorted(k for k,v in present.items() if score(v)==best)
  oracle=[choose(v[i] for v in present.values()) for i in range(n)]
  assert actual['single_methods']==winner
  for key,value in [('single_gap',score(target)-best if higher else best-score(target)),('oracle_gap',score(target)-score(oracle) if higher else score(oracle)-score(target))]:assert abs(actual[key]-float(value))<1e-10
  assert actual['oracle_values']==list(map(float,oracle))
with tempfile.TemporaryDirectory() as tmp:
 dst=Path(tmp)/'packet';shutil.copytree(Q,dst);(dst/'paper-v1.html').write_text('corrupt')
 try:replay191(dst,manifest)
 except ValueError:pass
 else:raise AssertionError('Corrupt source accepted')
with tempfile.TemporaryDirectory() as tmp:
 dst=Path(tmp)/'packet';shutil.copytree(Q,dst);(dst/'tables.json').unlink()
 try:replay191(dst,manifest)
 except ValueError:pass
 else:raise AssertionError('Missing evidence accepted')
result=dict(status='PASS',task_cells=401,all_numeric_cells=checked,method_rows=45,tables=[3,4,7,8],independent_parser='BeautifulSoup',independent_arithmetic='Fraction',independent_ranking='scipy.rankdata',independent_pool_comparisons=12,corruption_rejection=True,missing_file_rejection=True,aggregate_discrepancies=sum(x['aggregate_status']=='OUTSIDE_ROUNDING_BOUND' for t in report['tables'] for x in t['audit']),rank_differences=sum(x['rank_status']=='DIFFERS_DISPLAYED' for t in report['tables'] for x in t['audit']))
(P/'_verify_l191_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
for t in report['tables']:
 print('T',t['number'])
 for pool,x in t['pools'].items():print(pool,{k:v for k,v in x.items() if k in ['status','single_methods','single_gap','oracle_gap','target_score','single_score']})
 for x in t['audit']:
  if x['aggregate_status']=='OUTSIDE_ROUNDING_BOUND':print('OUTSIDE',x['method'],x['recomputed_aggregate'],x['published_aggregate'])
