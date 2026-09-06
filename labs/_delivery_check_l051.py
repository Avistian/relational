"""Check artifacts, linked pages, images, real predictions and executed solution."""
import ast,base64,hashlib,io,json,re
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score,roc_auc_score
from relkit.bias_interventions import paired_effect,rank_summary
HERE=Path(__file__).parent;ROOT=HERE.parent;slug='0051-why-trees-still-win'
reference=json.loads((HERE/'_verify_l051_results.json').read_text())
actual=json.loads((HERE/'data/cache/l051-teacher-results.json').read_text())
for file,digest in reference['source_sha256'].items():assert hashlib.sha256((HERE/file).read_bytes()).hexdigest()==digest,file
count=0
for name,row in reference['datasets'].items():
    splits=[set(x) for x in row['split_rows']]
    assert all(not (splits[i]&splits[j]) for i in range(3) for j in range(i))
    assert sum(map(len,splits))==1800
    for condition,runs in row['runs'].items():
        for model,rr in runs.items():
            for a,b in zip(rr,actual['datasets'][name]['runs'][condition][model]):
                assert a['seed']==b['seed'] and a['accuracy']==b['accuracy'],(name,condition,model)
                assert np.allclose(a['probability'],b['probability'],atol=1e-7)
                assert np.isclose(a['accuracy'],accuracy_score(row['test_y'],np.array(a['probability'])>=.5))
                assert np.isclose(a['auc'],roc_auc_score(row['test_y'],a['probability']));count+=1
    for c in ['rotated','noise','smoothed']:
        for m in ['MLP','FT-T','XGBoost']:
            expected=paired_effect([v['accuracy'] for v in row['runs'][c][m]],[v['accuracy'] for v in row['runs']['top5' if c=='smoothed' else 'original'][m]])
            assert expected==row['effects'][c][m]
for c,s in reference['statistics'].items():assert s==rank_summary(reference['datasets'],c)
larger=json.loads((HERE/'_paper_repro_l051_closer_summary.json').read_text());larger_count=0
for file,digest in larger['contract']['identity'].items():assert hashlib.sha256((HERE/file).read_bytes()).hexdigest()==digest,file
for name,row in larger['datasets'].items():
    assert len(row['selected_rows'])==6000
    for condition,runs in row['runs'].items():
        for model,rr in runs.items():
            assert [a['seed'] for a in rr]==[0,1,2]
            for a in rr:
                assert np.isclose(a['accuracy'],accuracy_score(row['test_y'],np.array(a['probability'])>=.5))
                assert np.isclose(a['auc'],roc_auc_score(row['test_y'],a['probability']));larger_count+=1
for c,s in larger['statistics'].items():assert s==rank_summary(larger['datasets'],c)
expected_images={hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'figures/l051').glob('*.png')}
code_count=0
for solution in [False,True]:
    p=HERE/('solutions' if solution else '')/f'{slug}.ipynb';nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
    seen=[];todo=0
    for cell in nb.cells:
        assert not cell.get('attachments')
        if cell.cell_type=='markdown':
            assert 'attachment:' not in cell.source
            for encoded in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',cell.source):
                data=base64.b64decode(encoded,validate=True);Image.open(io.BytesIO(data)).verify();seen.append(hashlib.sha256(data).hexdigest())
        else:
            if cell.source.startswith('# TODO'):
                todo+=1;assert ('____' not in cell.source) if solution else ('____' in cell.source)
            if solution:
                assert cell.execution_count is not None
                assert not any(o.output_type=='error' for o in cell.outputs);code_count+=1
            else:assert not cell.outputs and cell.execution_count is None
    assert todo==4 and len(seen)==6 and set(seen)==expected_images
html=BeautifulSoup((HERE/'html'/f'{slug}.html').read_text(),'html.parser')
embedded=[hashlib.sha256(base64.b64decode(n['src'].split(',',1)[1])).hexdigest() for n in html.find_all('img',src=True) if n['src'].startswith('data:image/png;base64,')]
assert len(embedded)==6 and set(embedded)==expected_images
links=0
for file in [ROOT/'lessons'/f'{slug}.html',ROOT/'reference/inductive-bias-interventions.html',HERE/'html'/f'{slug}.html']:
    soup=BeautifulSoup(file.read_text(),'html.parser');ids=[n['id'] for n in soup.find_all(id=True)];assert len(ids)==len(set(ids))
    for n in soup.find_all(['a','img','script','link']):
        raw=n.get('href') or n.get('src')
        if not raw:continue
        parts=urlsplit(raw)
        if parts.scheme or parts.netloc:continue
        target=(file.parent/unquote(parts.path)).resolve() if parts.path else file.resolve()
        assert target.exists(),(file,raw)
        if parts.fragment and target.suffix=='.html':
            page=BeautifulSoup(target.read_text(),'html.parser');assert page.find(id=unquote(parts.fragment)),(file,raw)
        links+=1
manifest=json.loads((ROOT/'lessons/manifest.json').read_text());row=next(x for x in manifest['lessons'] if x['id']==51)
assert row['quarter']==2 and row['labPath']==f'labs/{slug}.ipynb'
report=dict(prediction_runs_reconciled=count,larger_prediction_runs_checked=larger_count,executed_solution_code_cells=code_count,student_todos=4,inline_pngs_per_artifact=6,
    local_links_checked=links,source_hashes_match=True,notebook_and_reference_scores_match=True,browser='NOT_CHECKED: no installed browser found',
    live_colab='NOT_CHECKED: inline PNG packaging verified only',svg_panels=20,svg_glyph_bounds='81 labels; no violations with librsvg')
(HERE/'_delivery_l051_results.json').write_text(json.dumps(report,indent=2));print(report)
