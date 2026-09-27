"""Reject semantic bugs rather than test implementation text."""
import json
from pathlib import Path
from relkit import tasks_l124 as m
from _check_l124 import check_labels,check_validate,check_alignment
cases=[('inclusive-left',check_labels,'make_labels','results.date.gt(t)&','results.date.ge(t)&'),('exclusive-right',check_labels,'make_labels','results.date.le(end)','results.date.lt(end)'),('future-cohort-as-past',check_labels,'make_labels',"if cohort=='past':active &= results.date.le(t)","if False:active &= results.date.le(t)"),('entity-only-identity',check_validate,'validate_task',"rows.duplicated(['driverId','date'])","rows.duplicated(['driverId'])"),('immature-label',check_validate,'validate_task','if fit_time is not None and','if False and'),('positional-scoring',check_alignment,'aligned_mae','zip(joined.prediction,joined.position)','zip(predictions.prediction,rows.position)')]
source=Path(m.__file__).read_text();rejected=[]
for name,check,fn,old,new in cases:
 assert old in source
 ns={};exec(source.replace(old,new),ns)
 try:check(ns[fn])
 except (AssertionError,ValueError):rejected.append(name)
 else:raise AssertionError('Survived: '+name)
r={'status':'PASS','rejected':rejected};Path(__file__).with_name('_mutation_l124_results.json').write_text(json.dumps(r,indent=2));print(r)
