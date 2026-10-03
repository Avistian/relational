"""Bound the entire nine-job course matrix; retain failed attempts, never downscale."""
import subprocess,time,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/b09';ledger=E/'budget.json'
# Verify once per invocation; cold hash work is charged to preparation.
subprocess.run([sys.executable,str(P/'_preflight_b09.py')],check=True)
b=json.loads(ledger.read_text()) if ledger.exists() else dict(cap_s=3600,preparation_allowance_s=300,validation_reserve_s=180,paid_usd=0,attempts=[])
def save():ledger.write_text(json.dumps(b,indent=2))
for name in (sys.argv[1:] or ['exaone','nori','tabfm']):
 if name=='tabfm':
  import psutil
  weight_bytes=Path('/tmp/b09-weights/tabfm/regression/model.safetensors').stat().st_size
  estimate=2*weight_bytes+1024**3
  available=psutil.virtual_memory().available
  if estimate>available:
   gate=dict(status='INCOMPLETE_RESOURCE_GATE',model=name,estimated_load_peak_bytes=estimate,available_bytes=available,reason='Conservative unmodified-loader bound: constructed float32 model plus checkpoint mapping and 1GiB runtime reserve. No smaller model or precision substituted.')
   (E/'resource-gate.json').write_text(json.dumps(gate,indent=2));print(gate,flush=True);continue
 for seed in range(3):
  if (E/f'{name}-{seed}.json').exists():continue
  left=b['cap_s']-b['preparation_allowance_s']-b['validation_reserve_s']-sum(a['seconds'] for a in b['attempts'])
  if left<=0:save();sys.exit('INCOMPLETE_BUDGET_GATE')
  start=time.monotonic();log=E/f'{name}-{seed}-attempt-{len(b["attempts"])}.log'
  with log.open('w') as f:
   try:r=subprocess.run([sys.executable,str(P/'_worker_b09.py'),name,str(seed)],stdout=f,stderr=subprocess.STDOUT,timeout=min(left,900),env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'));status='OK' if r.returncode==0 else 'FAILED'
   except subprocess.TimeoutExpired:status='TIMEOUT'
  b['attempts'].append(dict(model=name,seed=seed,status=status,seconds=time.monotonic()-start,log=str(log.relative_to(P))));save();print(name,seed,status,flush=True)
  if status!='OK':break
