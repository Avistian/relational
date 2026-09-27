"""Make an isolated, independently budgeted replay without modifying existing evidence."""
import argparse,json,shutil,subprocess,sys,uuid
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();dest=a.directory.resolve()
assert not dest.exists(),'Fresh directory required'
dest.mkdir(parents=True);volume='l135-repeat-'+uuid.uuid4().hex[:12]
files=['_run_l117.py','_run_l135.py','_protocol_l135.json','_prepare_l135.py','_collect_l135.py','_freeze_l135.py','_pilot_check_l135.py','_analyze_l135.py','_audit_l135.py','_check_l135.py','_verify_l135.py','requirements-l117-runtime.txt','relkit/rdl_l117.py','relkit/batch_audit_l123.py','relkit/tuning_l135.py','relkit/tuning_train_l135.py']
for name in files:
 target=dest/'labs'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/name,target)
shutil.copytree(P/'sources/l117',dest/'labs/sources/l117',ignore=shutil.ignore_patterns('__pycache__'))
(dest/'labs/sources/l135').mkdir(parents=True);shutil.copyfile(P/'sources/l135/paper.html',dest/'labs/sources/l135/paper.html')
(dest/'modal').mkdir();shutil.copyfile(R/'modal/l135_repro.py',dest/'modal/l135_repro.py')
for path in [dest/'modal/l135_repro.py',dest/'labs/_collect_l135.py']:
 path.write_text(path.read_text().replace('l135-tuning-evidence',volume))
subprocess.run([sys.executable,str(dest/'labs/_prepare_l135.py')],cwd=dest,check=True)
(dest/'replay-origin.json').write_text(json.dumps(dict(source=str(R),volume=volume,scope='Fresh exact protocol; only artifact volume name differs. Independent USD10 cap.'),indent=2))
print('Created isolated replay:',dest,'; volume:',volume,'; no paid work launched')
