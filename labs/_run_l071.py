"""Public-data follow-up. Preset names describe resources, not paper fidelity."""
import argparse,json
from pathlib import Path
from relkit.vime_l071 import run_experiment
PRESETS={
 'smoke':dict(dataset='digits',seeds=(0,),budgets=(50,),pre_epochs=2,epochs=3),
 'closer':dict(dataset='mnist',seeds=(0,1,2),budgets=(100,1000,6000),pre_epochs=30,epochs=60),
 'paper':dict(dataset='mnist',seeds=tuple(range(10)),budgets=(100,1000,6000),pre_epochs=100,epochs=100),
}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preset',choices=PRESETS,default='smoke');p.add_argument('--device',default='cpu');p.add_argument('--output',default='l071-followup.json');a=p.parse_args()
 result=run_experiment(**PRESETS[a.preset],device=a.device)
 result['preset']=a.preset
 result['paper_gaps']=['60/40 unlabeled/reserve split differs from paper 90/10',
 'No paper architecture/hyperparameter search','Paired five-arm control is not the original baseline roster',
 'PyTorch port; no historical optimizer or random-state parity','Missing Income, Blog, clinical and genomic experiments']
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'verified_here':result['summary'],'paper_claim':'Cited, not reproduced','scale_up':a.preset,'verdict':'INCOMPARABLE'},indent=2))
