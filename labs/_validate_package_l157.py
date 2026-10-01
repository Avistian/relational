"""Independent artifact acceptance from a fresh archive extraction."""
import hashlib,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
P=Path(__file__).resolve().parent;D=P/'releases/l157-f1-audit'
with tempfile.TemporaryDirectory(prefix='l157-release-') as tmp:
 root=Path(tmp)
 with zipfile.ZipFile(P/'releases/l157-f1-audit.zip') as z:z.extractall(root)
 sys.path.insert(0,str(root/'labs'))
 from relkit.contribution_l157 import verify_manifest
 release=json.loads((root/'release-manifest.json').read_text());count=verify_manifest(root,release)
 assert not any(p.suffix=='.pt' or p.name in ['db.zip','driver-position.zip','feats.sql'] for p in root.rglob('*'))
 for args in [['cli.py','check'],['labs/_check_l157.py'],['cli.py','replay']]:
  subprocess.run([sys.executable,*args],cwd=root,check=True,capture_output=True,text=True,timeout=180)
 E=root/'labs/evidence/l157';report=json.loads((E/'report.json').read_text());predictions=0;checkpoints=0
 for lane in ['paper','fit_horizon']:
  for seed in range(5):
   d=E/lane/f'seed-{seed}';r=json.loads((d/'result.json').read_text());a=np.load(d/'predictions.npz')
   weights=P/'results/l157'/lane/f'seed-{seed}/selected.pt';assert hashlib.sha256(weights.read_bytes()).hexdigest()==r['checkpoint_sha256'];checkpoints+=1
   for split,n in [('val',499),('test',760)]:
    assert len(a[split+'_pred'])==n
    assert abs(mean_absolute_error(a[split+'_target'],a[split+'_pred'])-r['scores'][split])<1e-12;predictions+=n
 assert predictions==12590
 # A corrupted evidence file must fail, even without the higher-level release check.
 victim=E/'paper/seed-0/predictions.npz';original=victim.read_bytes();victim.write_bytes(original[:-1]+bytes([original[-1]^1]))
 bad=subprocess.run([sys.executable,'cli.py','replay'],cwd=root,capture_output=True,text=True);assert bad.returncode and 'Changed evidence' in bad.stderr
 victim.write_bytes(original)
 # Deleting a run is not silently converted to a partial average.
 victim.unlink();bad=subprocess.run([sys.executable,'cli.py','replay'],cwd=root,capture_output=True,text=True);assert bad.returncode
 r=dict(status='PASS',files=count,predictions=predictions,checkpoints=checkpoints,independent_metric='sklearn MAE',working_directory='EMPTY_EXTRACTION',corruption_rejected=True,missing_run_rejected=True,raw_inputs_and_weights_bundled=False,archive_sha256=hashlib.sha256((P/'releases/l157-f1-audit.zip').read_bytes()).hexdigest())
(P/'_package_l157_results.json').write_text(json.dumps(r,indent=2));print(r)
