"""Independent AP routine: tied-score randomized checks and a batch/pooled witness."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
P=Path(__file__).resolve().parent;source=(P/'_analyze_l110.py').read_text();tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='independent_ap');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'ap-function','exec'),ns);ap=ns['independent_ap']
rng=np.random.RandomState(110);error=0.;cases=0
for n in [2,3,10,100,1000]:
 for _ in range(100):
  y=rng.randint(0,2,n);y[0]=1;p=rng.randint(0,10,n)/10
  got=ap(y,p);expected=average_precision_score(y,p);error=max(error,abs(got-expected));assert abs(got-expected)<1e-14;cases+=1
batch_mean=(ap(np.array([1,0]),np.array([.9,.8]))+ap(np.array([1,0]),np.array([.2,.1])))/2
pooled=ap(np.array([1,0,1,0]),np.array([.9,.8,.2,.1]));assert batch_mean==1 and abs(pooled-5/6)<1e-14
out={'status':'PASS','random_tied_score_cases':cases,'maximum_error':error,'batch_mean_witness':batch_mean,'pooled_witness':pooled,'analyzer_sha256':hashlib.sha256(source.encode()).hexdigest()};(P/'_metrics_l110_results.json').write_text(json.dumps(out,indent=2));print(out)
