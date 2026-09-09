"""Verify L055 evidence, notebook provenance and a copied Pages staging tree."""
import base64,hashlib,json,re,shutil,tempfile
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from _build_l055 import ROOT,SLUG,build
from relkit.temporal_experiment import metric,summarize

result=json.loads((ROOT/'_verify_l055_results.json').read_text())
checks={}
for name,sha in result['source_hashes'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,(name,'source drift')
count=0
for name,task in result['tasks'].items():
    for strategy,records in task.items():
        for rec in records:
            for arm,runs in rec['arms'].items():
                for run in runs:
                    assert run['selected']==int(np.argmin(run['validation_errors']))
                    actual=metric(np.array(rec['test_targets']),np.array(run['predictions']),name=='sberbank-housing')
                    assert abs(actual-run['error'])<1e-6,(name,strategy,arm,actual,run['error'])
                    count+=1
checks['prediction_and_selection_reconciliations']=count
assert summarize(result)==result['summary']
student=nbformat.read(ROOT/(SLUG+'.ipynb'),as_version=4)
solution=nbformat.read(ROOT/'solutions'/(SLUG+'.ipynb'),as_version=4)
assert len([c for c in student.cells if c.cell_type=='code' and 'raise NotImplementedError' in c.source])==4
assert all(not c.outputs and c.execution_count is None for c in student.cells if c.cell_type=='code')
for nb,which in [(student,False),(solution,True)]:
    rebuilt=build(which)
    assert [(c.cell_type,c.source) for c in nb.cells]==[(c.cell_type,c.source) for c in rebuilt.cells]
    for c in nb.cells:
        if c.cell_type=='code' and which:
            assert c.execution_count is not None
            assert not any(o.output_type=='error' for o in c.outputs)
payloads=[]
for c in student.cells:
    if c.cell_type=='markdown':
        assert 'attachment:' not in c.source
        for b64 in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',c.source):
            data=base64.b64decode(b64);Image.open(BytesIO(data)).verify();payloads.append(hashlib.sha256(data).hexdigest())
assert len(payloads)==5
expected={hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'figures/l055').glob('*.png')}
assert set(payloads)==expected
teacher=json.loads((ROOT/'data/cache/l055/student-results.json').read_text())
for name,task in result['tasks'].items():
    for strategy,records in task.items():
        for arm,runs in records[0]['arms'].items():
            for a,b in zip(runs,teacher['tasks'][name][strategy][0]['arms'][arm]):
                assert a['selected']==b['selected']
                np.testing.assert_allclose(a['predictions'],b['predictions'],atol=1e-7,rtol=1e-6)
checks.update(solution_code_cells=sum(c.cell_type=='code' for c in solution.cells),student_todos=4,portable_figures=5,notebook_prediction_parity='PASS')

site=ROOT.parent
manifest=json.loads((site/'lessons/manifest.json').read_text())
entry=next(x for x in manifest['lessons'] if x['id']==55)
assert entry['slug']==SLUG and entry['labPath']=='labs/'+SLUG+'.ipynb'
assert len({x['id'] for x in manifest['lessons']})==len(manifest['lessons'])
for name in ('index.html','notebooks.html'):
    assert f'name="rdl-manifest-version" content="{manifest["version"]}"' in (site/name).read_text()

stage=Path(tempfile.mkdtemp(prefix='l055-pages-'))
for folder in ('assets','lessons','reference'):shutil.copytree(site/folder,stage/folder)
(stage/'labs').mkdir();(stage/'modal').mkdir()
for folder in ('html','figures'):shutil.copytree(ROOT/folder,stage/'labs'/folder)
for file in ROOT.glob('*.ipynb'):shutil.copy2(file,stage/'labs'/file.name)
for name in ['index.html','notebooks.html','flashcards.html']:shutil.copy2(site/name,stage/name)
for name in ['l055-reproduction.md','_verify_l055_results.json','_data_l055.json','_sources_l055.json']:
    assert name in (site/'.github/workflows/pages.yml').read_text()
    shutil.copy2(ROOT/name,stage/'labs'/name)
shutil.copy2(site/'modal/l055_paper_repro.py',stage/'modal/l055_paper_repro.py')
links=0
for relative in ['lessons/'+SLUG+'.html','reference/tabred-temporal-splits.html','labs/html/'+SLUG+'.html']:
    p=stage/relative;soup=BeautifulSoup(p.read_text(),'html.parser')
    assert not soup.find(string=lambda s:s and 'L055_RESULTS' in s)
    for el in soup.find_all(['a','img','script','link']):
        url=el.get('href') or el.get('src')
        if not url:continue
        u=urlsplit(url)
        if u.scheme or u.netloc:continue
        target=(p.parent/unquote(u.path)).resolve() if u.path else p
        assert target.exists(),(relative,url)
        if u.fragment and target.suffix=='.html':
            dest=BeautifulSoup(target.read_text(),'html.parser')
            assert dest.find(id=unquote(u.fragment)),(relative,url,'missing anchor')
        links+=1
checks.update(local_links_in_copied_pages_tree=links,staging_path=str(stage),browser='NOT_CHECKED',live_colab='NOT_CHECKED',deployment='NOT_CHECKED',closer='NOT_RUN',modal='NOT_RUN',paper_figure_2='INCOMPARABLE')
(ROOT/'_delivery_l055_results.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
