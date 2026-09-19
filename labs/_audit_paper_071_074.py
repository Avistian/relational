"""Delivery acceptance for the repaired reproduction lanes, separate from result claims."""
import ast
import hashlib
import json
from pathlib import Path
import nbformat
import numpy as np
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';REPRO=LAB/'reproductions'

def executable_ast(text):
    tree=ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and ast.get_docstring(node,clean=False) is not None:
            node.body=node.body[1:]
    return ast.dump(tree)

def run():
    reports=[]
    for lesson in range(71,75):
        path=next(LAB.glob(f'00{lesson}-*.ipynb'));nb=nbformat.read(path,as_version=4)
        code=[c for c in nb.cells if c.cell_type=='code'];paper=[c for c in code if 'paper-reproduction' in c.metadata.get('tags',[])]
        assert paper and any('RUN_PAPER_REPRO=False' in c.source for c in paper)
        assembled={}
        for cell in paper:
            if not cell.source.startswith('%%writefile '):continue
            first,body=cell.source.split('\n',1)
            file=first.split()[-1].removeprefix('paper-run/')
            assembled[file]=assembled.get(file,'')+body.rstrip('\n')+'\n'
        sources=list(assembled)
        for file,body in assembled.items():
            assert executable_ast(body)==executable_ast((REPRO/file).read_text()),(lesson,file,'stale inline implementation')
        assert not any(c.outputs or c.execution_count is not None for c in code)
        assert sum('raise NotImplementedError' in c.source for c in code if not c.source.startswith('%%writefile '))==3
        page=ROOT/'lessons'/f'{path.stem}.html';soup=BeautifulSoup(page.read_text(),'html.parser')
        assert any(a.get('href')=='../labs/reproductions/README.md' for a in soup.find_all('a'))
        reports.append(dict(lesson=lesson,inline_modules=sources,student_tasks=3,status='PASS'))
    manifest=json.loads((REPRO/'source_manifest.json').read_text())
    for file,digest in manifest['files'].items():
        assert hashlib.sha256((REPRO/file).read_bytes()).hexdigest()==digest,('stale manifest',file)
    targets=json.loads((REPRO/'scarf_targets.json').read_text());assert len(targets['1.0'])==len(targets['0.25'])==69
    measured=[]
    for path in sorted(REPRO.glob('*results.json')):
        result=json.loads(path.read_text());checked=0
        for row in result.get('records',[]):
            if 'prediction' not in row:continue
            y=np.asarray(row['target']);p=np.asarray(row['prediction'])
            assert y.shape==p.shape and np.isfinite(p).all()
            if 'accuracy' in row:assert abs(np.mean(y==p)-row['accuracy'])<1e-7
            if 'r2' in row:assert abs(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)-row['r2'])<1e-10
            checked+=1
        if result.get('smoke'):assert result['verdict']=='INCOMPARABLE'
        if path.name=='scarf-paper-results.json':
            assert len(result['records'])==360
            for summary in result['summary']:
                rows=[r for r in result['records'] if all(r[k]==summary[k] for k in ['data_id','fraction','arm'])]
                assert len(rows)==summary['n']==30
                assert abs(np.mean([r['accuracy'] for r in rows])-summary['mean'])<1e-12
                assert abs(np.std([r['accuracy'] for r in rows],ddof=1)-summary['sd'])<1e-12
        if path.name=='carte-paper-row-results.json':
            assert result['config']['num_model']==15 and result['config']['max_epoch']==500
            assert all(len(r['validation_losses'])==15 for r in result['records'] if 'r2' in r)
        if path.name=='subtab-paper-results.json':
            assert result['train_rows']==60000 and result['test_rows']==10000
            assert result['config']['epochs']==15 and result['config']['evaluation_noise'] is False
            assert len(result['records'])==9
        measured.append(dict(file=path.name,prediction_metrics_recomputed=checked))
    soup=BeautifulSoup((ROOT/'index.html').read_text(),'html.parser')
    assert soup.find(id='lesson-nav')
    assert not any(p.select('a[href^="lessons/"]') for p in soup.find_all('p')), 'Redundant home-page lesson announcements returned'
    workflow=(ROOT/'.github/workflows/pages.yml').read_text();assert 'cp -r labs/reproductions public/labs/' in workflow
    out=dict(status='PASS',lessons=reports,source_manifest_verified=True,home_page='tiles only',
             evidence=measured,
             result_claim='Delivery checks are not paper-result parity',live_colab='NOT_CHECKED',publication='NOT_CHECKED')
    (LAB/'_paper_reproduction_audit_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':run()
