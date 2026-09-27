"""Reject plausible mistakes in the learner's checkpoint functions."""
import json
from pathlib import Path
from _check_l130 import check_queries,check_mae,check_verdict
P=Path(__file__).resolve().parent;source=(P/'relkit/checkpoint_l130.py').read_text()
mutations=[
 ('entity_only_identity','keys.append((int(entity), int(time)))','keys.append((int(entity), 0))',check_queries,'validate_queries'),
 ('allow_duplicate_queries','if len(set(keys)) != expected_count:','if False:',check_queries,'validate_queries'),
 ('positional_scoring',"abs(lookup[key] - float(q['target']))", "abs(float(prediction_rows[keys.index(key)]['target']) - float(q['target']))",check_mae,'keyed_mae'),
 ('population_sd','statistics.stdev(values)','statistics.pstdev(values)',check_verdict,'reproduction_verdict'),
 ('reused_run',"if len({r['run_uuid'] for r in records}) != len(records):",'if False:',check_verdict,'reproduction_verdict'),
 ('incomplete_epochs',"r['epochs'] != 10",'False',check_verdict,'reproduction_verdict'),
 ('always_close',"abs(mean-target) <= tolerance",'True',check_verdict,'reproduction_verdict')]
results={}
for name,before,after,test,fn in mutations:
 assert before in source
 ns={};exec(compile(source.replace(before,after),name,'exec'),ns)
 try:test(ns[fn])
 except (AssertionError,ValueError,KeyError):results[name]='REJECTED'
 else:raise AssertionError('Surviving mutant '+name)
r=dict(status='PASS',mutations=results);(P/'_mutation_l130_results.json').write_text(json.dumps(r,indent=2));print(r)
