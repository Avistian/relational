import hashlib,json,platform,sys
import numpy,pandas,sklearn
from pathlib import Path
import torch
from relkit.tgat_l103 import load_wikipedia
from relkit.tgat_comparison_l103 import common_data,run_comparison
P=Path(__file__).resolve().parent;torch.set_num_threads(1)
n,e,d,a=load_wikipedia(P/'data/l102');d=common_data(d)
records=[]
for seed in range(3):records+=run_comparison(n,e,d,seed=seed,output=P/'results/l103/comparison')
report={'status':'COMPLETE','scope':'Matched one-layer course comparison, 2000 train / 400 validation / 400 test, 3 epochs, 3 seeds; not paper reproduction','split':{name:hashlib.sha256(x['e'].tobytes()).hexdigest() for name,x in d.items()},'counts':{name:len(x['u']) for name,x in d.items()},'records':records,'identity':{'implementation_sha256':{f:hashlib.sha256((P/'relkit'/f).read_bytes()).hexdigest() for f in ['tgat_l103.py','tgn_l102.py','tgat_comparison_l103.py']},'python':sys.version,'torch':torch.__version__,'numpy':numpy.__version__,'pandas':pandas.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'command':'OMP_NUM_THREADS=1 .venv/bin/python labs/_compare_l103.py'}}
(P/'_comparison_l103_results.json').write_text(json.dumps(report,indent=2)+'\n')
