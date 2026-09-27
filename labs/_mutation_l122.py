"""Reject plausible wrong constructions through the same learner CHECKs."""
import json
from pathlib import Path
from _check_l122 import check_keys,check_edges,check_graph
P=Path(__file__).resolve().parent
source=(P/'relkit/reg_l122.py').read_text()
mutations=[
 ('raw_id_as_index','lookup[key] = position','lookup[key] = key',check_keys,'key_index'),
 ('accept_duplicate_pk','if pd.isna(key) or key in lookup:','if pd.isna(key):',check_keys,'key_index'),
 ('null_is_row_zero','if pd.isna(key):\n            continue','if pd.isna(key):\n            key = primary_keys[0]',check_edges,'relation_edges'),
 ('reverse_forward_edges','pairs.append((source, lookup[key]))','pairs.append((lookup[key], source))',check_edges,'relation_edges'),
 ('lose_fk_role',"'f2p_' + fk","'f2p_any'",check_graph,'construct_reg'),
 ('bad_reverse_direction','data[kind].edge_index.flip(0).contiguous()','data[kind].edge_index.clone()',check_graph,'construct_reg')]
rejected=[]
for name,before,after,check,fn in mutations:
 assert before in source,name
 ns={};exec(compile(source.replace(before,after),name,'exec'),ns)
 try:check(ns[fn])
 except (AssertionError,ValueError,RuntimeError):rejected.append(name)
 else:raise AssertionError('Surviving mutant: '+name)
report={'status':'PASS','rejected_mutants':rejected,'scope':'Task CHECKs reject semantic errors; not just template checks'}
(P/'_mutation_l122_results.json').write_text(json.dumps(report,indent=2));print(report)
