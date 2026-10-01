"""Record bounded local typed-feature work beside existing fresh BART timing."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,time
from pathlib import Path
from relkit.rows_l163 import typed_features
P=Path(__file__).resolve().parent;E=P/'evidence/l163';p=json.loads((E/'fixtures.json').read_text());enc=json.loads((E/'encoding-receipt.json').read_text());rows=p['rows'];events=[]
for split in p['splits']:
 start=time.perf_counter();x,meta=typed_features([rows[i] for i in split['train']],rows);elapsed=time.perf_counter()-start
 assert x.shape==(240,8)
 events.append(dict(seed=split['seed'],seconds=elapsed,rows=240,width=8,operation='fit train moments/vocabularies; transform all240 rows'))
r=dict(typed=events,bart_seconds_per_variant=enc['seconds'],bart_rows_per_variant=240,bart_width=768,bart_seconds_with_fetch=enc['elapsed_with_fetch_seconds'],caveat='Single local timings, not a controlled speed benchmark. Typed work includes train fitting and transformation; BART times include tokenization/forward/pooling after model loading. Imports/process startup and head fitting excluded.',cloud_spend_usd=0)
(E/'timing.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
