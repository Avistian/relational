"""Run the frozen complete named experiment, or explicitly separate smoke training."""
import argparse,hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import torch
from relkit.tgn_l102 import load_wikipedia,run_training,PAPER_AP,CLOSE_TOLERANCE_PP
P=Path(__file__).resolve().parent

def main():
 a=argparse.ArgumentParser();a.add_argument('--preset',choices=['smoke','paper'],default='paper');a.add_argument('--seeds',default='0,1,2,3,4,5,6,7,8,9');a.add_argument('--threads',type=int,default=1);a.add_argument('--device',default='cpu');a.add_argument('--output',type=Path);args=a.parse_args()
 torch.set_num_threads(args.threads)
 out=args.output or P/'results/l102'/args.preset;out.mkdir(parents=True,exist_ok=True)
 nodes,edges,data,audit=load_wikipedia(P/'data/l102')
 code_sha=hashlib.sha256((P/'relkit/tgn_l102.py').read_bytes()).hexdigest()
 identity={'implementation_sha256':code_sha,'data':audit,'preset':args.preset,'torch':torch.__version__,
           'numpy':np.__version__,'python':sys.version,'platform':platform.platform(),'threads':args.threads,'device':args.device}
 identity_path=out/'identity.json'
 if identity_path.exists():
  assert json.loads(identity_path.read_text())==identity,'Run identity changed; use a fresh output directory'
 else:identity_path.write_text(json.dumps(identity,indent=2))
 if args.preset=='smoke':
  # Deliberately different experiment. Keep complete widths and real input features.
  data={k:{field:values[:min(len(values),600 if k=='train' else 200)] for field,values in d.items()} if k!='full' else d for k,d in data.items()}
 results=[]
 for seed in map(int,args.seeds.split(',')):
  done=out/f'seed-{seed}.json'
  if done.exists(): result=json.loads(done.read_text())
  else:result=run_training(nodes,edges,data,seed=seed,epochs=2 if args.preset=='smoke' else 50,
                           output=out,device=args.device)
  results.append(result)
 report={'scope':'Rossi v3 Table 2 Wikipedia TGN-attn two AP cells' if args.preset=='paper' else 'Real-data teaching smoke; INCOMPARABLE to paper',
         'identity':identity,'complete_seeds':[r['seed'] for r in results],
         'status':'COMPLETE' if args.preset=='paper' and sorted(r['seed'] for r in results)==list(range(10)) else 'PARTIAL' if args.preset=='paper' else 'MEASURED',
         'historical_identity':'INCOMPARABLE','full_paper_reproduction':'NOT_ESTABLISHED',
         'records':results,'summary':{}}
 for lane,key in [('all','test'),('new','new_test')]:
  scores=np.array([r[key]['ap']*100 for r in results])
  report['summary'][lane]={'mean_ap_percent':float(scores.mean()),'sample_sd_pp':float(scores.std(ddof=1)) if len(scores)>1 else None,
          'paper_ap_percent':PAPER_AP[lane],'gap_pp':float(scores.mean()-PAPER_AP[lane]),'close_tolerance_pp':CLOSE_TOLERANCE_PP,
          'numerical_verdict':('CLOSE' if abs(scores.mean()-PAPER_AP[lane])<=CLOSE_TOLERANCE_PP else 'OUTSIDE_TOLERANCE') if report['status']=='COMPLETE' else 'NOT_ESTABLISHED'}
 (out/'summary.json').write_text(json.dumps(report,indent=2))
 target=P/('_paper_l102_results.json' if args.preset=='paper' else '_experiment_l102_results.json')
 target.write_text(json.dumps(report,indent=2));print(json.dumps(report['summary'],indent=2))
if __name__=='__main__':main()
