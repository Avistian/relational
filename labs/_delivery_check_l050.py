"""Check complete delivery, full evidence arithmetic, live-code output and PNG packaging."""
import base64
import hashlib
import io
import json
import re
from pathlib import Path
from urllib.parse import unquote,urlsplit
import nbformat
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image
from sklearn.metrics import roc_auc_score,accuracy_score
from relkit.checkpoint import select_trial,paired_summary
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;SLUG='0050-q1-checkpoint'
links=0
for file in [ROOT/'lessons'/f'{SLUG}.html',ROOT/'reference/fair-comparison-checkpoint.html',HERE/'html'/f'{SLUG}.html']:
    soup=BeautifulSoup(file.read_text(),'html.parser');ids=[t['id'] for t in soup.select('[id]')]
    assert len(ids)==len(set(ids)),('Duplicate IDs',file)
    for tag,attr in [('a','href'),('img','src'),('script','src'),('link','href')]:
        for node in soup.find_all(tag):
            url=node.get(attr,'');parts=urlsplit(url)
            if not url or parts.scheme or parts.netloc:continue
            target=(file.parent/unquote(parts.path)).resolve() if parts.path else file
            assert target.exists(),(file,url)
            if parts.fragment and target.suffix=='.html':
                dest=soup if target==file else BeautifulSoup(target.read_text(),'html.parser')
                assert dest.find(id=parts.fragment) or dest.find(id=unquote(parts.fragment)) or dest.find(attrs={'name':unquote(parts.fragment)}),(file,url)
            links+=1
student=nbformat.read(HERE/f'{SLUG}.ipynb',4);teacher=nbformat.read(HERE/'solutions'/f'{SLUG}.ipynb',4)
source_images={hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'figures/l050').glob('*.png')}
for nb in [student,teacher]:
    nbformat.validate(nb);embedded=[]
    for cell in nb.cells:
        if cell.cell_type=='markdown':
            assert 'attachment:' not in cell.source
            for data in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',cell.source):
                raw=base64.b64decode(data,validate=True)
                with Image.open(io.BytesIO(raw)) as im:im.verify()
                assert hashlib.sha256(raw).hexdigest() in source_images
                embedded.append(raw)
    assert len(embedded)==5
html=(HERE/'html'/f'{SLUG}.html').read_text()
for raw in embedded:assert base64.b64encode(raw).decode() in html
code=[c for c in student.cells if c.cell_type=='code']
assert '@colab-bootstrap' in code[0].source
assert sum(c.source.startswith('# TODO') and '____' in c.source for c in code)==4
assert all(c.execution_count is None and not c.outputs for c in code)
for cell in teacher.cells:
    if cell.cell_type=='code':
        assert cell.execution_count is not None
        assert not any(o.output_type=='error' for o in cell.outputs)
r=json.loads((HERE/'_verify_l050_results.json').read_text())
assert r['source_sha256']==hashlib.sha256((HERE/'relkit/checkpoint.py').read_bytes()).hexdigest()
for name,row in r['datasets'].items():
    split=[set(row['split'][k]) for k in ['train','valid','test']]
    assert len(set.union(*split))==row['n'] and sum(map(len,split))==row['n']
    for model,runs in row['runs'].items():
        for run in runs:
            assert run['selected_trial']==select_trial([v['validation_auc'] for v in run['trials']])
            assert abs(run['auc']-roc_auc_score(row['y_test'],run['probability']))<1e-12
        assert row['summary'][model]==paired_summary([x['auc'] for x in runs],np.zeros(3))
measured=json.loads((HERE/'_scaleup_l050_evidence.json').read_text())
assert measured['identity']['operator']==hashlib.sha256((HERE/'_paper_repro_l050_measured.py').read_bytes()).hexdigest()
for seed in measured['seed_results']:
    row=seed['datasets']['higgs_small']
    assert seed['source_sha256']==r['source_sha256']
    for model,runs in row['runs'].items():
        run=runs[0]
        assert abs(run['accuracy']-accuracy_score(row['y_test'],np.array(run['probability'])>=.5))<1e-12
        assert run['selected_trial']==select_trial([x['validation_auc'] for x in run['trials']])
entry=next(e for e in json.loads((ROOT/'lessons/manifest.json').read_text())['lessons'] if e['id']==50)
assert entry['checkpoint'] and entry['labPath']==f'labs/{SLUG}.ipynb'
output={'status':'PASS','local_links':links,'inline_pngs_per_notebook':5,'student_tasks':4,
    'executed_teacher_cells':sum(c.cell_type=='code' for c in teacher.cells),
    'local_selected_runs':27,'larger_selected_runs':9,'browser':'NOT_CHECKED','live_colab':'NOT_CHECKED'}
(HERE/'_delivery_l050_results.json').write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))
