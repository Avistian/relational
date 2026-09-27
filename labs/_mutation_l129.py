"""Prove the semantic checks reject plausible mistakes."""
import ast,json
from pathlib import Path
from _check_l129 import check_past,check_align,check_select
P=Path(__file__).resolve().parent;source=(P/'relkit/manual_fe_l129.py').read_text()
mutants=[('future at cutoff','< r[\'event\'] < cutoff','< r[\'event\'] <= cutoff','past_summary',check_past),('late arrival','and r[\'available\'] <= cutoff','and True','past_summary',check_past),('include lower endpoint','cutoff - lookback <','cutoff - lookback <=','past_summary',check_past),('positional output','return [float(mapping[k]) for k in query_keys]','return [float(p) for p in predictions]','align_predictions',check_align),('test selection',"r['val_mae'], r['number']","r.get('test_mae',0), r['number']",'choose_trial',check_select),('last tie',"r['val_mae'], r['number']","r['val_mae'], -r['number']",'choose_trial',check_select)]
rejected=[]
for name,old,new,fn,check in mutants:
 assert old in source;ns={};exec(source.replace(old,new),ns)
 try:check(ns[fn])
 except (AssertionError,ValueError):rejected.append(name)
 else:raise AssertionError(name+' survived')
r=dict(status='PASS',rejected=rejected);(P/'_mutation_l129_results.json').write_text(json.dumps(r,indent=2));print(r)
