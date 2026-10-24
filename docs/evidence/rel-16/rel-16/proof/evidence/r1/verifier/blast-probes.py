"""Independent CLI/Bash probes at review HEAD. No implementation imports."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
ROOT=Path('/tmp/lead-wt/night-vrel16-r1')
OUT=Path(__file__).resolve().parent
rows=[]
with tempfile.TemporaryDirectory(prefix='rel16-vr1-probe-') as tmp:
    f=Path(tmp)
    for name in ['a.txt','b.txt','plain name.dat']:
        (f/name).write_text('fixture')
    def wf(name,plan):
        p=f/name
        p.write_text('jobs:\n  build:\n    steps:\n      - name: Build site\n        run: |\n'+''.join('          '+line+'\n' for line in plan.splitlines()))
        return p
    base=wf('base.yml','cp *.txt public/ 2>/dev/null || true')
    def shell(plan):
        public=f/'public'
        if public.exists(): shutil.rmtree(public)
        public.mkdir()
        r=subprocess.run(['bash','-eu'],input=plan,cwd=f,capture_output=True,text=True)
        files=sorted(p.relative_to(public).as_posix() for p in public.rglob('*') if p.is_file())
        return dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,files=files)
    baseline=shell('cp *.txt public/')
    assert baseline['files']==['a.txt','b.txt']
    cases=[('quoted_glob','cp "*.txt" public/ 2>/dev/null || true'),('rm_glob','cp *.txt public/\nrm -rf public/*.txt')]
    cases += [('root_'+str(n),'cp *.txt public/\nrm -rf '+operand) for n,operand in enumerate(['public','public/','public//','./public//'])]
    for name,plan in cases:
        other=wf('other.yml',plan)
        observed=shell(plan)
        assert observed['exit_code']==0 and observed['files']==[]
        cli=[]
        for args in [['--workflow',str(other)],['--workflow',str(base),'--compare',str(other)],['--workflow',str(other),'--compare',str(base)]]:
            r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'scripts/pages_publish_list.py'),'--root',str(f),*args],capture_output=True,text=True)
            cli.append(dict(args=args,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
            assert r.returncode==2 and r.stdout=='' and 'unsupported' in r.stderr
        rows.append(dict(case=name,plan=plan,shell=observed,cli=cli))
    for name,plan,expected in [('literal_quoted','cp "plain name.dat" public/',['plain name.dat']),('literal_subtree','mkdir -p public/sub\ncp *.txt public/sub/\nrm -rf public/sub',[])]:
        other=wf('control.yml',plan)
        observed=shell(plan)
        r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'scripts/pages_publish_list.py'),'--root',str(f),'--workflow',str(other)],capture_output=True,text=True)
        assert observed['exit_code']==0 and observed['files']==expected and r.returncode==0 and json.loads(r.stdout)['files']==expected
        rows.append(dict(case=name,plan=plan,shell=observed,cli=dict(exit_code=r.returncode,report=json.loads(r.stdout))))
(OUT/'blast-probes.json').write_text(json.dumps(dict(baseline=baseline,cases=rows),indent=2)+'\n')
print('OK independent probes: quoted wildcard, wildcard removal, four normalized root forms; all 18 CLI calls exit 2/empty stdout; literal controls match Bash')
