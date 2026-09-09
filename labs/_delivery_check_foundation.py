"""Audit artifact integrity, saved predictions, notebooks and copied Pages staging.

Run from the repository root. Browser checks are a separate executable operator.
"""
import base64, hashlib, json, re, shutil, subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit
import nbformat
import numpy as np
from bs4 import BeautifulSoup
from sklearn.metrics import log_loss, root_mean_squared_error
from _foundation_config import SLUGS, TASKS, ARCH
from relkit.benchmark_core import paired_summary, validate_partitions

ROOT=Path(__file__).resolve().parents[1]
LABS=ROOT/'labs'


def check(stage=Path('/tmp/relational-foundation-pages')):
    report={'status':'RUNNING','lessons':[],'checks':{},'live_colab':'NOT_RUN',
            'modal':'NOT_RUN','deployment':'Check separately after pushing the verified commit'}
    path=LABS/'_delivery_foundation_results.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    source=json.loads((LABS/'_sources_foundation.json').read_text())
    for name,entry in source['reference_files'].items():
        assert hashlib.sha256((LABS/'sources/foundation'/name).read_bytes()).hexdigest()==entry['sha256'],name
    report['checks']['pinned_source_hashes']='PASS'
    known_hashes={hashlib.sha256(p.read_bytes()).hexdigest() for folder in [LABS,LABS/'relkit',LABS/'sources/foundation'] for p in folder.glob('*.py')}
    rescored=0
    for n in range(58,71):
        slug=f'{n:04}-{SLUGS[n]}'
        for file in [ROOT/'lessons'/f'{slug}.html',ROOT/'reference'/f'{slug}.html',
                     LABS/'html'/f'{slug}.html',LABS/f'l{n:03}-reproduction.md']:
            assert file.exists(),file
        student=nbformat.read(LABS/f'{slug}.ipynb',as_version=4)
        solution=nbformat.read(LABS/'solutions'/f'{slug}.ipynb',as_version=4)
        nbformat.validate(student);nbformat.validate(solution)
        assert len(student.cells)==len(solution.cells)
        blanks=0;images=0;codes=0
        for a,b in zip(student.cells,solution.cells):
            if a.cell_type=='code':
                codes+=1
                if 'raise NotImplementedError(' in a.source:blanks+=1
                assert not a.outputs and a.execution_count is None, 'Student must not include teacher execution'
                assert b.execution_count is not None, (n,b.source[:80])
                assert not any(o.output_type=='error' for o in b.outputs),(n,b.source[:80])
            else:
                for payload in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',a.source):
                    assert base64.b64decode(payload,validate=True).startswith(b'\x89PNG\r\n\x1a\n');images+=1
                assert 'attachment:' not in a.source
        assert blanks==len(TASKS[n]),(n,blanks)
        assert images>=2+(n in ARCH),(n,images)
        evidence=json.loads((LABS/f'_verify_l{n:03}_results.json').read_text())
        for field in ['source_sha256','operator_sha256']:
            if field in evidence:assert evidence[field] in known_hashes,(n,'missing measured operator',field)
        audits=evidence.get('datasets',{})
        for audit in (audits.values() if isinstance(audits,dict) else []):
            if 'ids' in audit:validate_partitions(audit['ids'])
        for row in evidence.get('records',[]):
            if 'predictions' not in row or 'targets' not in row:continue
            y,p=row['targets'],row['predictions']
            score=root_mean_squared_error(y,p) if row.get('metric')=='RMSE' else log_loss(y,p,labels=[0,1])
            value=row.get('error',row.get('log_loss'))
            assert abs(score-value)<1e-6,(n,row.get('arm'),score,value)
            if 'selected' in row:assert int(np.argmin(row['validation_errors']))==row['selected']
            if 'selected_C' in row:assert [.01,1.,100.][int(np.argmin(row['validation_errors']))]==row['selected_C']
            rescored+=1
        if n in [60,70]:
            for regime,summary in evidence['summary'].items():
                rows=[v for v in evidence['records'] if v['dataset'].endswith('/'+regime)]
                actual=paired_summary(rows)
                assert actual==summary,(n,regime,'aggregation mismatch')
        report['lessons'].append(dict(lesson=n,code_cells=codes,portable_pngs=images,status='PASS'))
    report['checks']['saved_predictions_rescored']=rescored
    report['checks']['measured_operator_archives']='PASS'
    report['checks']['partition_disjointness']='PASS'
    report['browser_report']='_browser_foundation_results.json'
    report['checks']['notebooks']='13 executed solutions; blank student TODOs; portable PNG payloads validated'
    # Run the workflow's real copy commands from the repo into a fresh directory.
    # Never symlink the staging tree: relative assets must survive publication.
    workflow=(ROOT/'.github/workflows/pages.yml').read_text()
    script=workflow.split('        run: |\n',1)[1].split('\n      - name: Setup Pages',1)[0]
    lines=[line[10:] for line in script.splitlines() if line.strip()]
    lines=[line for line in lines if not line.startswith(('VER=','sed -i'))]
    if stage.exists():shutil.rmtree(stage)
    script=re.sub(r'\bpublic\b',str(stage),'\n'.join(lines))
    subprocess.run(['bash','-eu','-c',script],cwd=ROOT,check=True)
    broken=[];checked=0
    for n in range(58,71):
        slug=f'{n:04}-{SLUGS[n]}'
        for page in [stage/'lessons'/f'{slug}.html',stage/'reference'/f'{slug}.html',stage/'labs/html'/f'{slug}.html']:
            soup=BeautifulSoup(page.read_text(),'html.parser')
            for tag in soup.select('[href], [src]'):
                link=tag.get('href',tag.get('src'));parsed=urlsplit(link)
                if parsed.scheme or parsed.netloc or not parsed.path:continue
                target=(page.parent/unquote(parsed.path)).resolve();checked+=1
                if not target.exists():broken.append((str(page.relative_to(stage)),link))
    assert not broken,broken
    manifest=json.loads((stage/'lessons/manifest.json').read_text())
    entries=manifest['lessons'] if isinstance(manifest,dict) else manifest
    for n in range(58,71):
        item=next(v for v in entries if v['id']==n)
        assert item['labPath'] and (n not in [60,70] or item['checkpoint'])
    report['checks']['copied_pages_local_links']=checked
    report['checks']['manifest_registration']='PASS'
    report['status']='PASS'
    path.write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(path,stage/'labs'/path.name)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':check()
