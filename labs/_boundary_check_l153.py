"""Historical RED/current GREEN for inclusive recommendation tolerance."""
from pathlib import Path
from _check_l153 import check_portfolio
from relkit.recommendation_l153 import portfolio_entry
old={};exec((Path(__file__).parent/'sources/l153/recommendation_before_boundary.py').read_text(),old)
try:check_portfolio(old['portfolio_entry'])
except AssertionError as error:assert 'Inclusive tolerance boundary' in str(error)
else:raise AssertionError('Historical boundary should fail')
check_portfolio(portfolio_entry)
print('PASS: historical RED/current GREEN; measured pilot remains incomplete and has no paper verdict')
