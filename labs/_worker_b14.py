"""Fresh original RelArena trial/full grid; only the wrapper writes evidence."""
import sys,json,time,hashlib,traceback,dataclasses,importlib.metadata
from pathlib import Path
import numpy as np
P=Path('/out');P.mkdir(exist_ok=True);sys.path.insert(0,'/source/src');phase=sys.argv[1];start=time.monotonic()
r=dict(phase=phase,status='INCOMPLETE',trials=[])
try:
 import torch
 torch.set_num_threads(4)
 lock=json.loads(Path('/input-lock.json').read_text())
 for name,h in lock.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
 from relarena.dataset import RelBenchDatasetTask
 from relarena.checksums.checksum import _db_checksums,_label_checksums
 from relarena.models.tabpfn_rel.model import TabPFNRelLocalModel,TABPFN_REL_LOCAL_SPACE
 from relarena.cache import resolve_cache_config
 from relarena.tuner import run_trial
 from relarena.runner import run_experiment
 source=RelBenchDatasetTask('rel-f1','driver-dnf',download=False)
 cs={**_db_checksums(source),**_label_checksums(source)}
 expected=json.loads(Path('/source/src/relarena/checksums/relbench_v1_checksums.json').read_text())['rel-f1/driver-dnf'];assert cs==expected
 r['data_checksums']=cs
 if phase=='pilot':
  trials=[run_trial(TabPFNRelLocalModel,TABPFN_REL_LOCAL_SPACE.default_overrides,'default',source.task,source.inner_split(),seed=0,cache=resolve_cache_config('/dfs-cache',on_miss='raise'),run_identity=source.run_identity('inner'))]
  selected=None
 else:
  summary=run_experiment(TabPFNRelLocalModel,'rel-f1','driver-dnf',seed=0,n_trials=3,download=False,cache_predictions=True,cache_dir='/dfs-cache',require_all_trials=True)
  trials=summary.trials;selected=summary.tuned.config_id;r['selected']=selected
 task=source.task
 for trial in trials:
  row=dataclasses.asdict(trial);row.pop('val_pred');row.pop('test_pred')
  for split,arr in [('val',trial.val_pred),('test',trial.test_pred)]:
   if arr is None:continue
   frame=task.get_table(split,mask_input_cols=False).df
   np.savez(P/(trial.config_tag+'-'+split+'.npz'),entity=frame[task.entity_col].to_numpy(),date=frame[task.time_col].astype('int64').to_numpy(),label=frame[task.target_col].to_numpy(),prediction=arr)
  r['trials'].append(row)
 assert all(t.ok and t.val_pred is not None for t in trials)
 if phase=='full':assert len(trials)==3 and summary.tuned.test_pred is not None and summary.default.test_pred is not None
 r['status']='COMPLETE';r['gpu']=torch.cuda.get_device_name();r['seed']=0
except Exception as e:r.update(error=repr(e),traceback=traceback.format_exc())
r['seconds']=time.monotonic()-start;r['packages']={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','torch','tabpfn','relbench','fastdfs']};(P/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
