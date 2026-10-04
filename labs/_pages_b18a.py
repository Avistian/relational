"""Actual copied Pages build from an isolated Git index; preserve user's index.

Includes the prepared B16 and B17 working-tree packages because the inherited workflow
already references it. No files are committed/staged in the real index.
"""
import hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent

def run():
    started=time.monotonic();index_path=Path(subprocess.check_output(['git','rev-parse','--git-path','index'],cwd=R,text=True).strip())
    if not index_path.is_absolute():index_path=R/index_path
    before=hashlib.sha256(index_path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='b18a-pages-') as td:
        t=Path(td);env=dict(os.environ,GIT_INDEX_FILE=str(t/'index'),GIT_LFS_SKIP_SMUDGE='1')
        def git(*args):return subprocess.run(['git',*args],cwd=R,env=env,check=True,capture_output=True,text=True)
        git('read-tree','HEAD');git('add','-u')
        paths=[]
        for unit,slug,module in [('b16','b16-autograble-graph-selection','partitions_b16'),('b17','b17-reusable-representations','representations_b17'),('b18','b18-context-sufficiency','context_b18'),('b18a','b18a-context-state','state_b18a')]:
            for root in [P/'evidence'/unit,P/'sources'/unit,P/'figures'/unit]:
                paths.extend(p for p in (root.glob('*') if unit=='b18a' and root.name=='b18a' and root.parent.name=='sources' else root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
            paths.extend(P.glob('_*'+unit+'*.py'));paths.extend(P.glob('_*'+unit+'*.json'))
            paths.extend([P/f'{unit}-reproduction.md',P/f'{slug}.ipynb',P/'solutions'/f'{slug}.ipynb',P/'html'/f'{slug}.html',P/'relkit'/f'{module}.py',R/'lessons'/f'{slug}.html',R/'lessons/content'/f'{slug}.md',R/'reference'/f'{slug}.html'])
        paths.extend(R/'assets'/n for n in ['b16-evidence.js','partition-selection.css','partition-selection.js','b17-evidence.js','representation-boundary.css','representation-boundary.js','b18-evidence.js','context-budget.css','context-budget.js','context-state.css','context-state.js'])
        paths.append(R/'docs/plans/2026-10-04-b18a-design.md')
        git('add','-f','--',*[str(p.relative_to(R)) for p in paths if p.is_file()])
        root=t/'checkout';root.mkdir();git('checkout-index','--all','--prefix='+str(root)+'/')
        workflow=(root/'.github/workflows/pages.yml').read_text();block=workflow.split('      - name: Build site\n        run: |\n',1)[1].split('\n      - name:',1)[0]
        script='set -eu\n'+'\n'.join(line[10:] for line in block.splitlines() if line.startswith('          '))
        proc=subprocess.run(['bash','-c',script],cwd=root,capture_output=True,text=True)
        if proc.returncode:raise AssertionError(proc.stderr)
        checked=[]
        for p in paths:
            if 'b18a' not in str(p) and p.name not in ['context-state.css','context-state.js']:continue
            if p.name in ['local-budget.json','_pages_b18a_results.json']:continue
            relative=p.relative_to(R)
            if relative.parts[:2]==('lessons','content'):continue
            target=root/'public'/relative
            assert target.is_file(),str(relative)
            assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(p.read_bytes()).digest(),str(relative)
            checked.append(str(relative))
        manifest=json.loads((root/'public/lessons/manifest.json').read_text());assert any(x['id']=='B18a' and x['labPath'] for x in manifest['lessons'])
        from html.parser import HTMLParser
        from urllib.parse import urlsplit,unquote
        class Links(HTMLParser):
            def __init__(self):super().__init__();self.urls=[]
            def handle_starttag(self,tag,attrs):self.urls.extend(v for k,v in attrs if k in ('href','src'))
        links=0
        for rel in ['lessons/b18a-context-state.html','reference/b18a-context-state.html']:
            file=root/'public'/rel;parser=Links();parser.feed(file.read_text())
            for url in parser.urls:
                u=urlsplit(url)
                if u.scheme:continue
                target=(file.parent/unquote(u.path)).resolve() if u.path else file
                assert target.is_file(),url
                links+=1
        after=hashlib.sha256(index_path.read_bytes()).hexdigest();assert before==after,'User index changed'
        out=dict(status='PASS',actual_workflow_build=True,temporary_index=True,user_index_preserved=True,byte_identical_b18a_files=len(checked),files=checked,copied_site_local_links=links,seconds=time.monotonic()-started,publication='NOT_REQUESTED')
        (P/'_pages_b18a_results.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='files'})
if __name__=='__main__':run()
