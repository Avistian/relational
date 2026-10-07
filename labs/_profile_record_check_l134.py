"""Regression: reject malformed observations before timing/count aggregation."""
import json,runpy
from pathlib import Path
from _check_l134 import check_summary
from _profile_provenance_l134 import ARCHIVE,verify_profile_provenance
from relkit.scale_l134 import profile_summary
P=Path(__file__).resolve().parent
old=runpy.run_path(str(ARCHIVE))['profile_summary']
try:
    check_summary(old)
except AssertionError as exc:
    assert 'Negative batch timing accepted' in str(exc)
else:
    raise AssertionError('The archived implementation must expose the regression')
check_summary(profile_summary)
scale=json.loads((P/'evidence/l134/scale/scale.json').read_text())
for config in scale['configurations']:
    assert old(config['batches'])==profile_summary(config['batches'])==config['summary']
report=dict(status='PASS',archived_implementation='REJECTED_BY_NEW_CHECK',current_implementation='PASS',invalid_record_cases=10,saved_configurations_unchanged=len(scale['configurations']),provenance=verify_profile_provenance())
(P/'_profile_record_check_l134_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
