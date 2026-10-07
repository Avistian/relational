"""Regression: preserve inclusive score boundaries and all measured summaries."""
import json
from pathlib import Path
from _check_l152 import check_summary
from relkit.regression_l152 import portfolio_summary
P=Path(__file__).resolve().parent
old={};exec((P/'sources/l152/regression_before_boundary.py').read_text(),old)
try:check_summary(old['portfolio_summary'])
except AssertionError as error:assert 'Inclusive tolerance boundary' in str(error)
else:raise AssertionError('Historical implementation should expose the boundary regression')
check_summary(portfolio_summary)
rows=json.loads((P/'evidence/l152/summary.json').read_text())['records']
assert portfolio_summary(rows)==old['portfolio_summary'](rows)
print('PASS: historical RED, current GREEN, all measured summary fields unchanged')
