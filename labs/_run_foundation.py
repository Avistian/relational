"""Single explicit entrypoint for the L058–L070 evidence tracks.

Historical inference uses subprocess-only package directories; installing v1
cannot change a running notebook's imported v2 implementation. No cloud launch
occurs here. A true paper-reproduction preset is intentionally unsupported.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def isolated(mode,script,arguments):
    packages={'v1':['requests==2.32.5','tabpfn==0.1.11','scikit-learn==1.6.1','pandas==2.2.3'],
              'v2':['tabpfn==2.0.9','tabicl==0.1.4','scikit-learn==1.6.1','pandas==2.2.3','einops==0.8.1','huggingface-hub==0.34.4'],
              'current':['scikit-learn==1.6.1','pandas==2.2.3','tabpfn==8.5.0','tabicl==2.2.0','pydantic==2.12.5','pydantic-core==2.41.5',
                         'pydantic-settings==2.12.0','annotated-types==0.8.0','typing-inspection==0.4.4',
                         'python-dotenv==1.2.3','safetensors==0.7.0','einops==0.8.1','huggingface-hub==0.34.4']}
    folder=ROOT/'data/cache/foundation/packages'/mode;marker=folder/'course-packages.json'
    if not marker.exists() or json.loads(marker.read_text())!=packages[mode]:
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--upgrade','--target',str(folder),*packages[mode]],check=True)
        marker.write_text(json.dumps(packages[mode]))
    env=os.environ.copy();env['PYTHONPATH']=str(folder)+os.pathsep+str(ROOT);env['OMP_NUM_THREADS']='1'
    subprocess.run([sys.executable,str(ROOT/script),*arguments],env=env,cwd=ROOT,check=True)


def run(lesson,preset='lab',output=None,current=False):
    if preset=='paper':raise ValueError('No full paper-fidelity preset exists. Use a declared local track and list the remaining original-protocol work.')
    out=Path(output).resolve() if output else ROOT/'data/cache/foundation'/f'l{lesson:03}-{preset}.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    if current:
        if lesson!=70:raise ValueError('The current-version extension belongs to L070')
        from _fetch_foundation import fetch_current
        fetch_current();isolated('current','_run_current_foundation.py',['--output',str(out)]);return out
    if lesson == 61:
        from relkit.pfn_l061_v2 import run_experiment
        run_experiment(preset=preset, output=str(out), device='cpu')
        return out
    if lesson == 62:
        from relkit.tabpfn_l062_v2 import run_experiment
        import torch
        if out.exists():raise FileExistsError('Choose a fresh output; no resume/overwrite')
        torch.set_num_threads(1)
        result=run_experiment(ROOT,preset=preset)
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return out
    if lesson in [65,66,67,69]:
        from _fetch_foundation import fetch
        fetch('v1' if lesson==67 else ('tabicl' if lesson==66 else 'v2'))
        isolated('v1' if lesson==67 else 'v2','_run_pretrained_foundation.py',
                 ['--lesson',str(lesson),'--output',str(out)])
        return out
    if lesson == 60:
        from _fetch_l055 import fetch
        if preset != 'smoke':
            fetch()
        from relkit.checkpoint_l060_v2 import run_checkpoint
        run_checkpoint(preset=preset, output=str(out))
        return out
    if lesson in [64,70]:
        if lesson in [64,70]:
            from _fetch_foundation import fetch
            fetch('v2');fetch('tabicl')
            isolated('v2','_run_foundation.py',['--lesson','70','--preset',preset,'--output',str(out),'--worker'])
        if lesson==64:
            from relkit.benchmark_core import paired_summary
            result=json.loads(out.read_text());result['lesson']=64
            result['records']=[r for r in result['records'] if r['arm'] in ['XGBoost','TabPFN-v2']]
            result['summary']={'random':paired_summary(result['records'])}
            result['scope']='Focused Nature-v2 versus XGBoost slice of the newly run historical checkpoint'
            out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return out
    from relkit.foundation_experiments import (survey_audit,validation_audit,posterior_experiment,
        scm_experiment,drift_experiment,temporal_pfn_experiment)
    import torch
    torch.set_num_threads(1)
    if lesson==58:result=survey_audit()
    elif lesson==59:result=validation_audit()
    elif lesson==63:result=scm_experiment()
    elif lesson==68:
        result=drift_experiment();result['pfn_ablation']=temporal_pfn_experiment({'smoke':10,'lab':400,'closer':2000}[preset])
    else:raise ValueError('Lesson must be 58–70')
    result['preset']=preset;out.write_text(json.dumps(result,indent=2,allow_nan=False,default=lambda x:x.item())+'\n');return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lesson',type=int,required=True);p.add_argument('--preset',choices=['smoke','lab','closer','paper'],default='lab')
    p.add_argument('--output');p.add_argument('--current',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:
        from relkit.foundation_benchmark import run_checkpoint
        run_checkpoint(a.lesson,a.preset,a.output)
    else:print(run(a.lesson,a.preset,a.output,a.current))
