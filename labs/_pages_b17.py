"""Actual copied Pages build from an isolated Git index; preserve user's index.

Includes the prepared B16 working-tree package because the inherited workflow
already references it. No files are committed/staged in the real index.
"""
import hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent

def run():
    started=time.monotonic();index_path=Path(subprocess.check_output(['git','rev-parse','--git-path','index'],cwd=R,text=True).strip())
    if not index_path.is_absolute():index_path=R/index_path
    before=hashlib.sha256(index_path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='b17-pages-') as td:
        t=Path(td);env=dict(os.environ,GIT_INDEX_FILE=str(t/'index'),GIT_LFS_SKIP_SMUDGE='1')
        def git(*args):return subprocess.run(['git',*args],cwd=R,env=env,check=True,capture_output=True,text=True)
        git('read-tree','HEAD');git('add','-u')
        paths=[]
        for unit,slug,module in [('b16','b16-autograble-graph-selection','partitions_b16'),('b17','b17-reusable-representations','representations_b17')]:
            for root in [P/'evidence'/unit,P/'sources'/unit,P/'figures'/unit]:
                paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
            paths.extend(P.glob('_*'+unit+'*.py'));paths.extend(P.glob('_*'+unit+'*.json'))
            paths.extend([P/f'{unit}-reproduction.md',P/f'{slug}.ipynb',P/'solutions'/f'{slug}.ipynb',P/'html'/f'{slug}.html',P/'relkit'/f'{module}.py',R/'lessons'/f'{slug}.html',R/'lessons/content'/f'{slug}.md',R/'reference'/f'{slug}.html'])
        paths.extend(R/'assets'/n for n in ['b16-evidence.js','partition-selection.css','partition-selection.js','b17-evidence.js','representation-boundary.css','representation-boundary.js'])
        git('add','-f','--',*[str(p.relative_to(R)) for p in paths if p.is_file()])
        root=t/'checkout';root.mkdir();git('checkout-index','--all','--prefix='+str(root)+'/')
        workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
        script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
        proc=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
        if proc.returncode:raise AssertionError(proc.stderr)
        checked=[]
        for p in paths:
            if 'b17' not in str(p) and p.name not in ['representation-boundary.css','representation-boundary.js']:continue
            if p.name in ['local-budget.json','_pages_b17_results.json']:continue
            relative=p.relative_to(R)
            if relative.parts[:2]==('lessons','content'):continue
            target=root/'public'/relative
            assert target.is_file(),str(relative)
            assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(p.read_bytes()).digest(),str(relative)
            checked.append(str(relative))
        manifest=json.loads((root/'public/lessons/manifest.json').read_text());assert any(x['id']=='B17' and x['labPath'] for x in manifest['lessons'])
        after=hashlib.sha256(index_path.read_bytes()).hexdigest();assert before==after,'User index changed'
        out=dict(status='PASS',actual_workflow_build=True,temporary_index=True,user_index_preserved=True,byte_identical_b17_files=len(checked),files=checked,seconds=time.monotonic()-started,publication='NOT_REQUESTED')
        (P/'_pages_b17_results.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='files'})
if __name__=='__main__':run()
