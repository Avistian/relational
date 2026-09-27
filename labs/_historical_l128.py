"""Independently compare recovered task rows with unmodified pre-flip SQL."""
import importlib.util,io,json,zipfile
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from relbench.datasets import get_dataset
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('historical_f1',P/'sources/l128/f1-historical.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
dataset=get_dataset('rel-f1',download=True);task=m.DriverDNFTask(dataset,cache_dir=None)
report={}
with zipfile.ZipFile(P/'sources/l128/driver-dnf.zip') as archive:
 for split in ['train','val','test']:
  actual=task._get_table(split).df
  expected=pq.read_table(io.BytesIO(archive.read('driver-dnf/'+split+'.parquet')),use_threads=False).to_pandas()
  expected['did_not_finish']=1-expected.did_not_finish
  cols=['driverId','date','did_not_finish']
  a=actual[cols].sort_values(cols[:2]).reset_index(drop=True);b=expected[cols].sort_values(cols[:2]).reset_index(drop=True)
  pd.testing.assert_frame_equal(a,b,check_dtype=False)
  report[split]=dict(rows=len(a),positives=int(a.did_not_finish.sum()),all_keys_and_labels='EXACT_PRE_FLIP_SQL')
r=dict(status='PASS',splits=report,archive_identity='NOT_ESTABLISHED',row_order='Contemporary archive; historical unavailable')
(P/'_historical_l128_results.json').write_text(json.dumps(r,indent=2));print(r)
