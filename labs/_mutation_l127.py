"""Check that learner tests reject plausible experiment-reporting mistakes."""
import json,types
from pathlib import Path
from _check_l127 import check_contract,check_selection,check_summary
P=Path(__file__).resolve().parent;source=(P/'relkit/benchmark_l127.py').read_text()
mutations=[('changed_epochs','epochs=10,seeds=', 'epochs=1,seeds=',check_contract,'experiment_contract'),
 ('last_validation_tie',"return min(trace,key=lambda row:row['val_mae'])['epoch']","return min(reversed(trace),key=lambda row:row['val_mae'])['epoch']",check_selection,'select_checkpoint'),
 ('test_selection',"return min(trace,key=lambda row:row['val_mae'])['epoch']","return min(trace,key=lambda row:row['test_mae'])['epoch']",check_selection,'select_checkpoint'),
 ('population_sd','statistics.stdev(r[s] for r in ordered)','statistics.pstdev(r[s] for r in ordered)',check_summary,'summarize_seeds'),
 ('permit_reused_identity',"if len({r['run_uuid'] for r in records})!=len(records):","if False:",check_summary,'summarize_seeds'),
 ('allow_incomplete_epochs',"r['epochs']!=10","False",check_summary,'summarize_seeds')]
results={}
for name,before,after,test,fn in mutations:
 assert before in source,name
 ns={};exec(compile(source.replace(before,after),name,'exec'),ns)
 try:test(ns[fn])
 except (AssertionError,ValueError,KeyError):results[name]='REJECTED'
 else:raise AssertionError('Surviving mutation '+name)
r=dict(status='PASS',mutations=results)
(P/'_mutation_l127_results.json').write_text(json.dumps(r,indent=2));print(r)
