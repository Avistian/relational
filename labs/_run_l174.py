"""Run twelve fresh fits in a new directory; refuse to overwrite evidence."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from relkit.finetune_l174 import load_adaptation_packet,train_adaptation,adaptation_split,AdaptationModel
from relkit.multitask_l173 import tensor_packet,predict,score_outputs
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--packet',type=Path,default=P/'evidence/l174');args=parser.parse_args()
if args.output.exists():raise SystemExit('Refusing to overwrite existing output')
a,t,states,m=load_adaptation_packet(args.packet)
args.output.mkdir(parents=True)
for name in list(m['files'])+['manifest.json']:shutil.copyfile(args.packet/name,args.output/name)
torch.set_num_threads(1)
split=adaptation_split(a['dates'][a['cell_ids']]);train=np.flatnonzero(split==0);test=np.flatnonzero(split==2);data=tensor_packet(a)
baselines=[]
for seed,state in enumerate(states):
    model=AdaptationModel(t,'freeze');model.load_state_dict(state);out=predict(model,data,test)
    np.savez_compressed(args.output/f'unchanged-{seed}.npz',cell_ids=a['cell_ids'][test],task=a['task'][test],target=a['target'][test],output=out)
    baselines.append(dict(name='unchanged',seed=seed,test=score_outputs(out,a['target'][test],a['task'][test],t)))
out=np.zeros((len(test),max(s['classes'] for s in t)),dtype=np.float32)
for i,s in enumerate(t):
    y=a['target'][train[a['task'][train]==i]];mask=a['task'][test]==i
    if s['kind']=='number':out[mask,0]=y.mean(dtype=np.float64)
    else:
        counts=np.bincount(y.astype(int),minlength=s['classes'])+1
        out[mask,:s['classes']]=np.log(counts/counts.sum())
np.savez_compressed(args.output/'constant.npz',cell_ids=a['cell_ids'][test],task=a['task'][test],target=a['target'][test],output=out)
baselines.append(dict(name='constant',test=score_outputs(out,a['target'][test],a['task'][test],t)))
(args.output/'baselines.json').write_text(json.dumps(baselines,indent=2)+'\n')
runs=[]
for seed in range(3):
    for arm in ['freeze','full','adapter','scratch']:
        runs.append(train_adaptation(a,t,states[seed],arm,seed,args.output/f'{arm}-{seed}'))
        (args.output/'report.json').write_text(json.dumps(dict(status='COMPLETE' if len(runs)==12 else 'INCOMPLETE',runs=runs,numpy=np.__version__,torch=torch.__version__,whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE'),indent=2)+'\n')
