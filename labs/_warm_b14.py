"""Original maximum-depth feature construction, no neural inference."""
import sys,json,time,traceback,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/b14/relarena';E=P/'evidence/b14';sys.path.insert(0,str(S/'src'))
from relarena.dataset import RelBenchDatasetTask,concat_tables
from relarena.cache import resolve_cache_config
from relarena.featurization.dfs import build_dfs_features
start=time.monotonic();result=dict(status='INCOMPLETE',frames=[])
try:
 source=RelBenchDatasetTask('rel-f1','driver-dnf',download=True);inner=source.inner_split();outer=source.outer_split();hist=concat_tables(outer.train_table,outer.val_table)
 cache=resolve_cache_config(Path('/tmp/b14-dfs-cache'),on_miss='fill')
 for phase,db,hist,query in [('inner',inner.db_state,inner.train_table,inner.eval_table),('outer',outer.db_state,hist,outer.eval_table)]:
  for name,table in [('support',hist),('query',query)]:
   t=time.monotonic();frame,extra=build_dfs_features(source.task,db,table,depth=4,max_depth=4,history_table=hist,keep_anchor_columns=True,cache=cache,run_identity=source.run_identity(phase))
   path=E/(phase+'-'+name+'.parquet');frame.to_parquet(path)
   result['frames'].append(dict(phase=phase,name=name,rows=len(frame),columns=len(frame.columns),seconds=time.monotonic()-t,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 result['status']='PASS'
except Exception as e:result.update(status='INCOMPLETE_SOURCE_FEATURES',error=repr(e),traceback=traceback.format_exc())
result['seconds']=time.monotonic()-start;(E/'warm-receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
