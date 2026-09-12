"""Fresh-process and adversarial live-graph checks for L070."""
import json,os,subprocess,sys
from pathlib import Path
from relkit import checkpoint_l070_v2 as core
ROOT=Path(__file__).parent

def check():
 s=vars(core).copy();first=core.kernel_identity(s,ROOT)['sha256'];original=core.XGBClassifier.predict_proba
 try:
  core.XGBClassifier.predict_proba=lambda self,x:None
  assert core.kernel_identity(s,ROOT)['sha256']!=first
 finally:core.XGBClassifier.predict_proba=original
 assert core.kernel_identity(s,ROOT)['sha256']==first
 # Function-valued defaults and closure values are semantic inputs too.
 def factory(v):
  def f(x):return x+v
  return f
 assert core.stable_code(factory(1))!=core.stable_code(factory(2))
 command=[sys.executable,'-c','from pathlib import Path; from relkit import checkpoint_l070_v2 as c; print(c.kernel_identity(vars(c),Path("labs"))["sha256"])']
 env=os.environ.copy();env['PYTHONPATH']=str(ROOT.resolve())
 outputs=[subprocess.check_output(command,cwd=ROOT.parent,env=env,text=True).strip() for _ in range(2)]
 assert len(set(outputs))==1 and outputs[0]==first
 return dict(status='PASS',kernel_identity=first,fresh_process_hashes=outputs,checks=['actual-XGBoost-method','closure-value','two-fresh-processes'],scope='Known live code and runtime graph; no resume promised')
if __name__=='__main__':
 r=check();(ROOT/'_identity_l070_v2_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
