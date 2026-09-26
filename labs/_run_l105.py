"""Run all three predeclared full-data conditions, with independent SQLite oracle."""
import argparse,hashlib,importlib.util,json,platform,sqlite3,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stream_l105',P/'relkit/stream_l105.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def run(raw,output):
 begin=time.perf_counter();d=m.load_wikipedia(raw);output.mkdir(parents=True,exist_ok=True)
 # Parse with a second mechanism; skip feature payload, retain exact typed raw IDs.
 rows=[]
 with Path(raw).open() as f:
  next(f)
  for line in f:
   u,v,t,_=line.split(',',3);rows.append((int(u),int(v),float(t)))
 assert np.array_equal(np.array(rows),np.column_stack([d['u'],d['v'],d['t']]))
 db=sqlite3.connect(':memory:');db.execute('CREATE TABLE events(u INTEGER,v INTEGER,t REAL)')
 db.executemany('INSERT INTO events VALUES (?,?,?)',rows)
 reports=[];oracles=[]
 for width in m.WIDTHS:
  snap=m.aggregate_events(d,width);report=m.audit_stream(d,width)
  sql=db.execute('SELECT CAST(t / ? AS INTEGER), u, v, COUNT(*), MIN(t), MAX(t) FROM events GROUP BY 1,2,3 ORDER BY 1,2,3',(width,)).fetchall()
  measured=np.column_stack([snap[k] for k in ['bin','u','v','count','first','last']])
  np.testing.assert_array_equal(measured,np.array(sql))
  # Independent grouped timestamp walk: query multiplicities are explicit.
  bins=defaultdict(Counter)
  for u,v,t in rows:bins[int(t//width)][t]+=1
  hidden=withheld=nonpast=ties=0
  for bucket,times in bins.items():
   previous=0;total=sum(times.values())
   for t,count in sorted(times.items()):
    hidden+=previous*count;withheld+=previous*count
    nonpast+=(total-previous)*count;ties+=count*(count-1)//2;previous+=count
  assert report['hidden_strict_pairs']==hidden
  assert report['withheld_past_sum']==withheld
  assert report['nonpast_exposure_sum']==nonpast
  assert report['tied_timestamp_pairs']==ties
  assert int(snap['count'].sum())==len(rows)
  # Every query's completed-window count: independent cumulative bin lookup.
  total=0;past_by_bin={}
  for bucket in sorted(bins):past_by_bin[bucket]=total;total+=sum(bins[bucket].values())
  expected=np.array([past_by_bin[int(t//width)] for t in d['t']])
  ends=np.unique(snap['end']);cumulative=np.array([sum(bins[b].values()) for b in sorted(bins)]).cumsum()
  pos=np.searchsorted(ends,d['t'],side='right');actual=np.r_[0,cumulative][pos]
  np.testing.assert_array_equal(actual,expected)
  np.savez_compressed(output/f'snapshots-{width}.npz',**snap)
  reports.append(report);oracles.append({'width_seconds':width,'sql_group_rows':len(sql),'all_query_release_counts':len(rows),'independent_timestamp_pairs':'EXACT','sql_group_aggregates':'EXACT'})
 db.close()
 np.savez_compressed(output/'events.npz',**d)
 result={'status':'PASS','experiment':'L105 full Wikipedia representation audit','paper_result_reproduction':'NOT_APPLICABLE: course-defined audit, no published score target','dataset':{'url':m.DATA_URL,'sha256':sha(raw),'bytes':Path(raw).stat().st_size,'events':len(rows),'users':len(np.unique(d['u'])),'items':len(np.unique(d['v'])),'time_min_seconds':float(d['t'][0]),'time_max_seconds':float(d['t'][-1]),'unique_timestamps':len(np.unique(d['t'])),'assumption':'availability equals event time; ingestion histories unavailable'},'protocol':{'width_seconds':list(m.WIDTHS),'origin_seconds':0,'windows':'[start,end)','query_population':'all event timestamps, with event multiplicity','strict_history':'event time < query','snapshot_release':'max(end, max constituent availability) over the entire window','snapshot_semantics':'window interaction graphs, not cumulative state graphs','features_and_labels':'excluded; endpoint/time projection only','order_metric':'all pairs of events within the same bin with distinct times; global, not node-local','access_metric':'global stream records, not neural dependencies','training':'none','seeds':'not applicable: deterministic census'},'records':reports,'independent_oracles':oracles,'code_sha256':{str(p.relative_to(P)):sha(p) for p in [P/'relkit/stream_l105.py',Path(__file__)]},'artifact_sha256':{p.name:sha(p) for p in sorted(output.glob('*.npz'))},'runtime':{'python':sys.version,'numpy':np.__version__,'sqlite':sqlite3.sqlite_version,'platform':platform.platform()},'elapsed_seconds':time.perf_counter()-begin}
 (P/'_analysis_l105_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,default=P/'data/l102/wikipedia.csv');parser.add_argument('--output',type=Path,default=P/'evidence/l105');args=parser.parse_args();run(args.raw,args.output)
