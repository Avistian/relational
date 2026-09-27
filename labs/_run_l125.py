"""Full released F1 feature-path execution; never labeled Table 2 reproduction."""
import hashlib,json,platform,shutil
from pathlib import Path
import numpy as np
import torch,torch_frame,pandas
from relkit.frame_l125 import load_f1_archive,encode_f1_tables,gradient_step
P=Path(__file__).resolve().parent;E=P/'evidence/l125';E.mkdir(parents=True,exist_ok=True)
cache=Path.home()/'.cache/relbench/rel-f1'
for src,dst in [('db.zip','f1-db.zip'),('tasks/driver-position.zip','f1-task.zip')]:
 if not (E/dst).exists():shutil.copyfile(cache/src,E/dst)
raw=(E/'f1-db.zip').read_bytes();tables=load_f1_archive(raw)
models,frames,exports,report=encode_f1_tables(tables)
arrays={}
for name,item in exports.items():arrays[name+'_ids']=np.asarray(item['ids']);arrays[name+'_vectors']=item['vectors'].numpy()
np.savez_compressed(E/'encoded-reg.npz',**arrays)
result={'status':'PASS','experiment':'L125 complete F1 feature export + one training-query gradient step','paper_result':'NOT_RUN','tables':report,'total_rows':sum(r['rows'] for r in report.values()),'gradient':gradient_step(tables,models,frames,(E/'f1-task.zip').read_bytes()),'provenance':{n:hashlib.sha256((E/n).read_bytes()).hexdigest() for n in ['f1-db.zip','f1-task.zip','encoded-reg.npz']},'runtime':{'python':platform.python_version(),'torch':torch.__version__,'frame':torch_frame.__version__,'pandas':pandas.__version__},'weights':'Random initialization seed125; exported before the separate gradient step','fit_cutoff':'2004-09-03 for dated tables; full static tables have no creation history','text':'Fixed 16-bin token counts, not pretrained embeddings','paid_compute_usd':0}
(E/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
