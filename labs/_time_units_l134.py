"""Regression: epoch counts are meaningless without a declared unit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
reference=1577836800
rows=[]
for unit in ['ns','us','ms','s']:
 s=pd.Series(pd.to_datetime(['2020-01-01'])).astype('datetime64['+unit+']')
 naive=int((s.astype('int64')//10**9).iloc[0])
 corrected=int(s.astype('datetime64[s]').astype('int64').iloc[0])
 assert corrected==reference
 rows.append(dict(unit=unit,naive_divide_1e9=naive,explicit_seconds=corrected))
assert sum(r['naive_divide_1e9']==reference for r in rows)==1
Path(__file__).with_name('_time_units_l134_results.json').write_text(json.dumps(dict(status='PASS',cases=rows),indent=2));print(rows)
