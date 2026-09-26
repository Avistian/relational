"""Completed-seed reuse needs identical source/data/runtime; partial epochs are not resumable."""
import json,shutil,subprocess,sys,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='l110-resume-') as tmp:
 root=Path(tmp);(root/'relkit').mkdir();(root/'data').mkdir();(root/'data/l102').symlink_to(P/'data/l102',target_is_directory=True)
 shutil.copy(P/'relkit/checkpoint_l110.py',root/'relkit/checkpoint_l110.py');shutil.copy(P/'_run_l110.py',root/'_run_l110.py')
 command=[sys.executable,str(root/'_run_l110.py'),'--preset','smoke','--arm','clean','--seeds','19','--output',str(root/'output')]
 subprocess.run(command,check=True,capture_output=True,text=True)
 second=subprocess.run(command,check=True,capture_output=True,text=True);assert 'Completed seed already present' in second.stdout
 model=root/'relkit/checkpoint_l110.py';s=model.read_text().replace("return (event_time < query_time) & (observed_time < query_time)","return event_time < query_time");model.write_text(s)
 third=subprocess.run(command,capture_output=True,text=True);assert third.returncode!=0 and 'Changed run identity' in third.stderr
r={'status':'PASS','unchanged_completed_seed':'REUSED','changed_eligibility_helper':'REJECTED','fingerprint':'Exact source bytes plus data identity and runtime versions; stricter than semantic hash','partial_epoch_resume':'NOT_SUPPORTED'}
(P/'_resume_l110_results.json').write_text(json.dumps(r,indent=2));print(r)
