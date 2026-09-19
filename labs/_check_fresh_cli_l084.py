"""Check a two-worker CLI in an empty copied workspace; invoke with the clean runtime."""
from pathlib import Path
import tempfile,shutil,subprocess,json,hashlib,sys
lab=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='l084-fresh-cli-') as tmp:
 root=Path(tmp);(root/'relkit').mkdir()
 for name in ['_run_l084.py','_sources_l084.json']:shutil.copy2(lab/name,root/name)
 shutil.copy2(lab/'relkit/gat_l084.py',root/'relkit/gat_l084.py')
 subprocess.run([sys.executable,str(root/'_run_l084.py'),'--seeds','2','--workers','2','--max-epochs','3','--output',str(root/'smoke.json')],check=True)
 r=json.loads((root/'smoke.json').read_text());assert len(r['runs'])==2 and all(len(v['trace'])==3 for v in r['runs'])
 result={'status':'PASS','fresh_empty_data_cache':True,'interpreter':sys.executable,'spawned_workers':2,'smoke_seeds':2,'epochs_per_seed':3,'paper_result_status':'INCOMPARABLE; CLI packaging test only','runner_sha256':hashlib.sha256((lab/'_run_l084.py').read_bytes()).hexdigest()}
 (lab/'_fresh_cli_l084_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
