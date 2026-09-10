"""Audit delivery, numerical evidence, embedded figures and actual copied Pages build."""
import ast,base64,hashlib,io,json,re,shutil,subprocess,tempfile
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
import numpy as np
import yaml
from PIL import Image
from bs4 import BeautifulSoup
from scipy.stats import rankdata,friedmanchisquare
from relkit.realmlp_experiment import load_task,error,seed_interval
from relkit.realmlp import last_best
from _ablation_l053 import paired_effect

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent;SLUG='0053-realmlp-strong-defaults'
def check_links(path):
    soup=BeautifulSoup(path.read_text(),'html.parser');checked=0
    for el in soup.find_all(['a','img','script','link']):
        link=el.get('href') or el.get('src')
        if not link:continue
        part=urlsplit(link)
        if part.scheme or part.netloc:continue
        target=(path.parent/unquote(part.path)).resolve() if part.path else path
        assert target.exists(),f'{path.name}: missing {link}'
        if part.fragment and target.suffix=='.html':
            page=BeautifulSoup(target.read_text(),'html.parser')
            assert page.find(id=unquote(part.fragment)),f'{path.name}: missing anchor {link}'
        checked+=1
    return checked

def main():
    result=json.loads((ROOT/'_verify_l053_results.json').read_text());count=0
    for name,d in result['results'].items():
        data=load_task(name,data_root=ROOT/'data/cache/l052')
        assert data['selection']==d['selection'] and data['hashes']==d['hashes']
        for arm,runs in d['runs'].items():
            for run in runs:
                score=error(np.array(run['prediction']),data['y']['test'],data['regression'],data['target_std'])
                np.testing.assert_allclose(score,run['error'],rtol=1e-6,atol=1e-6);count+=1
                if arm=='RealMLP-S':assert run['best_epoch']==last_best(run['history'])+1 and len(run['history'])==64
                if arm=='XGB-tuned':assert run['selected']==int(np.argmin([c['validation_error'] for c in run['search']]))
            stat=seed_interval([r['error'] for r in runs]);assert stat==d['summary'][arm]
    means=np.array([[d['summary'][a]['mean'] for a in result['ranks']['arms']] for d in result['results'].values()])
    expected=np.mean([rankdata(row) for row in means],axis=0)
    np.testing.assert_allclose(expected,list(result['ranks']['means'].values()))
    assert abs(friedmanchisquare(*means.T).pvalue-result['ranks']['friedman_p'])<1e-12
    for path,digest in result['code_hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    larger=json.loads((ROOT/'_paper_repro_l053_closer_summary.json').read_text())
    assert hashlib.sha256((ROOT/'_paper_repro_l053_measured.py').read_bytes()).hexdigest()==larger['contract']['operator']
    data=load_task('california',6000,None,ROOT/'data/cache/l052')
    for run in larger['runs']:
        np.testing.assert_allclose(error(np.array(run['prediction']),data['y']['test'],True,data['target_std']),run['error'],rtol=1e-6)
        assert run['best_epoch']==last_best(run['history'])+1 and len(run['history'])==256
    assert seed_interval([r['error'] for r in larger['runs']])==larger['summary']
    ablation=json.loads((ROOT/'_ablation_l053_results.json').read_text())
    for path,digest in ablation['code_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    ablation_count=0
    for name,baseline in ablation['baseline']['results'].items():
        changed=ablation['robust_only']['results'][name]
        data=load_task(name,data_root=ROOT/'data/cache/l052')
        assert baseline['selection']==changed['selection']==data['selection']
        assert baseline['hashes']==changed['hashes']==data['hashes']
        a=baseline['runs']['RealMLP-S'];b=changed['runs']['RealMLP-S']
        for new,old in zip(a,result['results'][name]['runs']['RealMLP-S']):
            np.testing.assert_array_equal(new['prediction'],old['prediction'])
        for run in a+b:
            np.testing.assert_allclose(error(np.array(run['prediction']),data['y']['test'],
                data['regression'],data['target_std']),run['error'],rtol=1e-6)
            assert run['best_epoch']==last_best(run['history'])+1
            ablation_count+=1
        assert ablation['paired'][name]==dict(metric=baseline['metric'],**paired_effect(a,b))
    student=nbformat.read(ROOT/f'{SLUG}.ipynb',as_version=4);solution=nbformat.read(ROOT/'solutions'/f'{SLUG}.ipynb',as_version=4)
    nbformat.validate(student);nbformat.validate(solution)
    assert sum('raise NotImplementedError' in c.source for c in student.cells if c.cell_type=='code')==4
    assert all(c.execution_count is None and not c.outputs for c in student.cells if c.cell_type=='code')
    scode=[c for c in solution.cells if c.cell_type=='code']
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in scode)
    images=0
    for c in student.cells:
        assert 'attachment:' not in c.source
        for payload in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',c.source):
            data=base64.b64decode(payload,validate=True);im=Image.open(io.BytesIO(data));im.verify();images+=1
    assert images==8
    code='\n'.join(c.source for c in student.cells if c.cell_type=='code')
    for name in ['RealMLPS','NTPLinear','fit_neural','fit_trees','run_suite']:assert ('class '+name if name in ['RealMLPS','NTPLinear'] else 'def '+name) in code
    assert 'model_class=RealMLPS,runner=run_suite,prep_class=RobustSmooth' in code
    assert 'RUN_PAPER_REPRO = False' in code
    assert 'ablation_results=run_suite(RealMLPS,prep_class=RobustOnly,neural_only=True)' in code
    assert "data/cache/l053-student/exit.json" in code
    for path in [ROOT/'relkit/realmlp.py',ROOT/'relkit/realmlp_experiment.py',REPO/'modal/l053_paper_repro.py']:ast.parse(path.read_text())
    manifest=json.loads((REPO/'lessons/manifest.json').read_text());entry=next(x for x in manifest['lessons'] if x['id']==53)
    assert entry['labPath']==f'labs/{SLUG}.ipynb'
    # Recreate a COPY of workflow inputs; never let sed mutate the actual repository.
    stage=Path(tempfile.mkdtemp(prefix='l053-pages-'))
    for d in ['assets','lessons','reference']:shutil.copytree(REPO/d,stage/d)
    for f in ['index.html','notebooks.html','flashcards.html','.nojekyll']:shutil.copy2(REPO/f,stage/f)
    (stage/'labs').mkdir();shutil.copytree(ROOT/'html',stage/'labs/html');shutil.copytree(ROOT/'figures',stage/'labs/figures')
    # The current Pages workflow also publishes the foundation source packages.
    for directory in ['relkit','sources']:shutil.copytree(ROOT/directory,stage/'labs'/directory)
    for pattern in ['*.ipynb','*.json','*.md','*.py']:
        for f in ROOT.glob(pattern):shutil.copy2(f,stage/'labs'/f.name)
    shutil.copytree(REPO/'modal',stage/'modal')
    (stage/'reviews').mkdir()
    shutil.copy2(REPO/'reviews/lessons-047-070-depth-and-architecture.md',stage/'reviews')
    workflow=yaml.safe_load((REPO/'.github/workflows/pages.yml').read_text())
    command=next(s['run'] for s in workflow['jobs']['build']['steps'] if s.get('name')=='Build site')
    subprocess.run(['bash','-e','-c',command],cwd=stage,check=True,capture_output=True,text=True)
    paths=[Path('lessons')/f'{SLUG}.html',Path('reference/realmlp-strong-defaults.html'),Path('labs/html')/f'{SLUG}.html']
    links=sum(check_links(stage/'public'/p) for p in paths)
    gate=json.loads((ROOT/'_gate_l053_results.json').read_text());assert gate['status']=='PASS'
    final=dict(status='PASS',local_prediction_checks=count,larger_prediction_checks=3,ablation_prediction_checks=ablation_count,executed_solution_cells=len(scode),
        student_todos=4,portable_pngs=images,copied_pages_links=links,live_code_gate='PASS',
        browser='SEE_SEPARATE_AUDIT',browser_reason='Current local Chromium audit: reviews/lesson-quality-audit-047-070/053-browser.json',
        live_colab_ui='NOT_CHECKED',modal_execution='NOT_RUN',deployment='NOT_CHECKED',
        figures='Eight portable figures; architecture dimensions and meta-development boundary repaired; separately measured paired clipping intervention added',
        visual_improvement='Compared with L052: explicit per-coordinate NTP products and a fixed baseline under the feature-scale intervention; schedule displays all parameter-group rates.')
    (ROOT/'_delivery_l053_results.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final,indent=2))

if __name__=='__main__':main()
