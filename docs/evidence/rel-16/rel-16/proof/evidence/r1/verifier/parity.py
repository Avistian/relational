"""Verifier inventory: actual Bash/GNU cp destinations, no parser import."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import yaml
ROOT=Path('/tmp/lead-wt/night-vrel16-r1')
OUT=Path(__file__).resolve().parent
workflow=ROOT/'.github/workflows/pages.yml'
steps=yaml.safe_load(workflow.read_text())['jobs']['build']['steps']
plans=[s['run'] for s in steps if s.get('name')=='Build site']
assert len(plans)==1
lines=plans[0].splitlines()
skipped=[line for line in lines if line.strip().startswith(('VER=', 'sed '))]
assert len(skipped)==2 and skipped[0].strip().startswith('VER=$(python3 -c ') and skipped[1].strip().startswith('sed -i ')
with tempfile.TemporaryDirectory(prefix='rel16-r1-verifier-publication-') as tmp:
    public=Path(tmp)/'public'
    script='set -eu\ncp() { command cp --reflink=auto "$@"; }\n'+'\n'.join(line for line in lines if line not in skipped).replace('public/',str(public)+'/')
    (OUT/'workflow-copy.sh.txt').write_text(script)
    shell=subprocess.run(['bash'],input=script,cwd=ROOT,capture_output=True,text=True)
    (OUT/'workflow-copy.stdout.txt').write_text(shell.stdout)
    (OUT/'workflow-copy.stderr.txt').write_text(shell.stderr)
    assert shell.returncode==0,shell.stderr
    actual=[]
    for current,dirs,names in os.walk(public):
        actual += [(Path(current)/name).relative_to(public).as_posix() for name in names+[name for name in dirs if (Path(current)/name).is_symlink()]]
    actual=sorted(actual)
    (OUT/'actual-files.json').write_text(json.dumps(actual,indent=2)+'\n')
cli=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','scripts/pages_publish_list.py','--compare','.github/workflows/pages.yml'],cwd=ROOT,capture_output=True,text=True)
(OUT/'compare.stdout.json').write_text(cli.stdout)
(OUT/'compare.stderr.txt').write_text(cli.stderr)
assert cli.returncode==0,cli.stderr
report=json.loads(cli.stdout)
retained=json.loads((ROOT/'docs/evidence/rel-16/rel-16/evidence/builder/outside-files.json').read_text())
result=dict(source=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT))),shell_exit_code=shell.returncode,compare_exit_code=cli.returncode,generated_count=len(report['files']),actual_count=len(actual),equal=report['files']==actual,retained_inventory_equal=retained==actual,only_generated=sorted(set(report['files'])-set(actual)),only_shell=sorted(set(actual)-set(report['files'])),comparison=report['comparison'],diagnostics={k:report[k] for k in ['missing_sources','missing_references','omitted_references']},workflow_sha256=hashlib.sha256(workflow.read_bytes()).hexdigest(),skipped_content_only_commands=skipped)
(OUT/'parity.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['equal'] and result['retained_inventory_equal'] and result['generated_count']==result['actual_count']==18497
assert all(not v for v in result['diagnostics'].values())
print('OK Bash/current generator/retained inventory all equal: 18497 files; diagnostics empty')
