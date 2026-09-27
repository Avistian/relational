"""Create an isolated fresh training operator; no paid work is launched."""
import argparse,json,shutil,subprocess,sys,uuid
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();dest=a.directory.resolve();assert not dest.exists();dest.mkdir(parents=True)
volume='l137-repeat-'+uuid.uuid4().hex[:12]
files=['_run_l117.py','_run_l135.py','_run_l137.py','_prepare_l137.py','_collect_l137.py','_pilot_check_l137.py','_analyze_l137.py','_audit_l137.py','requirements-l117-runtime.txt','relkit/rdl_l117.py','relkit/batch_audit_l123.py','relkit/tuning_l135.py','relkit/tuning_train_l135.py']
for name in files:
 target=dest/'labs'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/name,target)
shutil.copytree(P/'sources/l117',dest/'labs/sources/l117',ignore=shutil.ignore_patterns('__pycache__'))
(dest/'modal').mkdir();shutil.copyfile(R/'modal/l137_repro.py',dest/'modal/l137_repro.py')
for path in [dest/'modal/l137_repro.py',dest/'labs/_collect_l137.py']:path.write_text(path.read_text().replace('l137-error-analysis-evidence',volume))
subprocess.run([sys.executable,str(dest/'labs/_prepare_l137.py')],cwd=dest,check=True)
(dest/'replay-origin.json').write_text(json.dumps(dict(source=str(R),volume=volume,scope='Fresh exact historical training protocol. New budget requires authorization; not an extension of the completed author run.'),indent=2))
print('Isolated runnable training replay:',dest,'; no paid work launched')
