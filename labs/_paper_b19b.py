"""Full97-task candidate-release operator. Default only writes a plan; no model calls."""
import argparse,ast,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources/b19b'
def plan():
 text=(S/'wrapper/gift_eval/dataset_definition.py').read_text();ns={}
 for n in ast.parse(text).body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['SHORT_DATASETS','MED_LONG_DATASETS'] for t in n.targets):exec(compile(ast.Module(body=[n],type_ignores=[]),'<roster>','exec'),ns)
 short=ns['SHORT_DATASETS'];long=ns['MED_LONG_DATASETS'];assert len(short)+2*len(long)==97
 return dict(status='NOT_RUN',scope='All97 TabPFN-TS tasks only; other12 model outputs remain saved releases',configurations=len(short),tasks=97,config=[dict(dataset=d,terms=['short','medium','long'] if d in long else ['short']) for d in sorted(short)],deviations=['Explicit local backend instead of archived CLI client default','User-provided authenticated checkpoint; historical actual-run equivalence unestablished','Original environment and confidence interval method unavailable'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',action='store_true');ap.add_argument('--execute',action='store_true');ap.add_argument('--checkpoint',type=Path);ap.add_argument('--checkpoint-sha256');ap.add_argument('--data',type=Path);ap.add_argument('--output',type=Path,default=Path('b19b-fresh'));ap.add_argument('--seconds',type=int);args=ap.parse_args();p=plan()
 if not args.execute:print(json.dumps(p,indent=2));return
 if not all([args.checkpoint,args.checkpoint_sha256,args.data,args.seconds]) or args.seconds<=0:ap.error('Execution requires preinstalled environment, --checkpoint, --checkpoint-sha256, --data and positive --seconds')
 if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=args.checkpoint_sha256:raise ValueError('CHECKPOINT_HASH_MISMATCH')
 manifest=json.loads((S/'manifest.json').read_text())
 for e in manifest['files']:
  if hashlib.sha256((S/e['file']).read_bytes()).hexdigest()!=e['sha256']:raise ValueError('SOURCE_HASH_MISMATCH')
 args.output.mkdir(parents=True,exist_ok=False);(args.output/'plan.json').write_text(json.dumps(p,indent=2));deadline=time.monotonic()+args.seconds;attempts=[]
 # Packages must already exist; no installs, remote inference, telemetry or automatic data downloads.
 env=dict(os.environ,WANDB_MODE='disabled',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TABPFN_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='1')
 child="""import os,sys,runpy
sys.path.insert(0,sys.argv[1])
from tabpfn_time_series import defaults
from tabpfn_time_series import predictor
defaults.TABPFN_DEFAULT_CONFIG['model_path']=sys.argv[2]
import gift_eval.tabpfn_ts_wrapper as wrapper
original=wrapper.TabPFNTimeSeriesPredictor
checkpoint=sys.argv[2]
wrapper.TabPFNTimeSeriesPredictor=lambda **kw: original(tabpfn_mode=wrapper.TabPFNMode.LOCAL,tabpfn_config={'model_path':checkpoint})
sys.argv=['evaluate.py','--dataset',sys.argv[3],'--terms',sys.argv[4],'--dataset_storage_path',sys.argv[5],'--output_dir',sys.argv[6]]
runpy.run_module('gift_eval.evaluate',run_name='__main__')
"""
 for config in p['config']:
  remaining=deadline-time.monotonic()
  if remaining<=0:break
  cmd=[sys.executable,'-c',child,str(S/'wrapper'),str(args.checkpoint.resolve()),config['dataset'],','.join(config['terms']),str(args.data.resolve()),str(args.output.resolve())]
  started=time.monotonic();proc=subprocess.Popen(cmd,env=env,start_new_session=True)
  try:code=proc.wait(timeout=remaining)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
  attempts.append(dict(dataset=config['dataset'],returncode=code,seconds=time.monotonic()-started));(args.output/'attempts.json').write_text(json.dumps(attempts,indent=2))
  if code:break
 status='COMPLETED_COMMANDS_REQUIRES_SCORE_AUDIT' if len(attempts)==len(p['config']) and all(x['returncode']==0 for x in attempts) else 'INCOMPLETE'
 (args.output/'status.json').write_text(json.dumps(dict(status=status,actual_checkpoint_sha256=args.checkpoint_sha256,historical_parity='NOT_ESTABLISHED')))
if __name__=='__main__':main()
