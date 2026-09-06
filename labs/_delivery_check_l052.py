"""Check generated sources, notebook evidence, portable images and copied Pages links."""
import base64,hashlib,json,re,shutil,tempfile
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
import numpy as np
from bs4 import BeautifulSoup
from _build_l052 import build,SLUG
from relkit.tabr_experiment import score_predictions
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def links(path,root):
    soup=BeautifulSoup(path.read_text(),'html.parser');count=0
    for node in soup.find_all(['a','img','script','link']):
        href=node.get('href') or node.get('src')
        if not href:continue
        part=urlsplit(href)
        if part.scheme or part.netloc:continue
        target=(path.parent/unquote(part.path)).resolve() if part.path else path
        assert target.exists(),f'Missing link from {path}: {href}'
        if part.fragment and target.suffix=='.html':
            other=BeautifulSoup(target.read_text(),'html.parser')
            assert other.find(id=unquote(part.fragment)),f'Missing anchor {href}'
        count+=1
    return count

def check():
    student=nbformat.read(HERE/f'{SLUG}.ipynb',as_version=4)
    solution=nbformat.read(HERE/'solutions'/f'{SLUG}.ipynb',as_version=4)
    for nb,is_solution in [(student,False),(solution,True)]:
        expected=build(is_solution,write=False)
        assert [c.source for c in nb.cells]==[c.source for c in expected.cells],'Builder drift'
    student_code=[c for c in student.cells if c.cell_type=='code']
    assert sum('raise NotImplementedError' in c.source for c in student_code)==4
    assert '@colab-bootstrap' in student_code[0].source
    code=[c for c in solution.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    assert any('class TabRS' in c.source for c in code) and any('def fit_neural' in c.source for c in code)
    images=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)','\n'.join(c.source for c in student.cells))
    assert len(images)==6
    for value in images:assert base64.b64decode(value,validate=True).startswith(b'\x89PNG\r\n\x1a\n')
    assert 'attachment:' not in '\n'.join(c.source for c in student.cells)
    reference=json.loads((HERE/'_verify_l052_results.json').read_text())
    for name,digest in reference['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    n_scores=0
    for name,task in reference['results'].items():
        for arm,runs in task['runs'].items():
            for run in runs:
                got=score_predictions(np.array(run['prediction']),np.array(task['test_target']),task['metric']=='RMSE',task['target_std'])
                assert abs(got-run['score'])<1e-7*max(1,abs(got));n_scores+=1
    live=json.loads((HERE/'data/cache/l052-exit.json').read_text())
    for row in live['metrics']:
        want=reference['results'][row['dataset']]['summary'][row['model']]
        assert abs(row['mean']-want['mean'])<1e-7*max(1,abs(want['mean'])),'Teacher/reference scores disagree'
    paths=['lessons/'+SLUG+'.html','reference/tabr-retrieval.html','labs/html/'+SLUG+'.html']
    local_count=sum(links(ROOT/p,ROOT) for p in paths)
    # Match the deployed asset layout using copies, never symlinks.
    with tempfile.TemporaryDirectory(prefix='l052-pages-') as directory:
        stage=Path(directory)
        for p in ['assets','lessons','reference']:shutil.copytree(ROOT/p,stage/p)
        for p in ['index.html','notebooks.html','flashcards.html']:shutil.copyfile(ROOT/p,stage/p)
        (stage/'labs').mkdir();(stage/'modal').mkdir()
        shutil.copytree(HERE/'html',stage/'labs/html');shutil.copytree(HERE/'figures',stage/'labs/figures')
        for p in HERE.glob('*.ipynb'):shutil.copyfile(p,stage/'labs'/p.name)
        for p in ['_verify_l052_results.json','_source_check_l052_results.json','_sources_l052.json','_data_l052.json','l052-reproduction.md','_paper_repro_l052_closer_summary.json']:
            if (HERE/p).exists():shutil.copyfile(HERE/p,stage/'labs'/p)
        shutil.copyfile(ROOT/'modal/l052_paper_repro.py',stage/'modal/l052_paper_repro.py')
        staged_count=sum(links(stage/p,stage) for p in paths)
    manifest=json.loads((ROOT/'lessons/manifest.json').read_text());entry=next(x for x in manifest['lessons'] if x['id']==52)
    assert entry['labPath']==f'labs/{SLUG}.ipynb' and entry['published']
    result=dict(status='PASS',executed_solution_cells=len(code),student_todos=4,embedded_pngs=len(images),
                prediction_scores_checked=n_scores,teacher_reference_mean_parity=True,local_links=local_count,copied_pages_links=staged_count,
                browser='NOT_CHECKED: no installed browser or browser tool',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
    (HERE/'_delivery_l052_results.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':check()
