"""L048 delivery checks: notebook independence, images, links and provenance.

Not a live browser or Colab test. Run after _build, solution execution and HTML export.
"""
import ast
import base64
import hashlib
import io
import json
import re
from pathlib import Path
from urllib.parse import urlsplit,unquote
import nbformat
from bs4 import BeautifulSoup
from PIL import Image

HERE=Path(__file__).parent
ROOT=HERE.parent
IMAGE=re.compile(r'!\[[^\]]*\]\((data:image/png;base64,[A-Za-z0-9+/=]+)\)')


def image_hash(uri):
    data=base64.b64decode(uri.split(',',1)[1],validate=True)
    with Image.open(io.BytesIO(data)) as im:
        assert im.format=='PNG';im.verify()
    return hashlib.sha256(data).hexdigest()


def main():
    expected=sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'figures/l048').glob('*.png'))
    assert len(expected)==10
    student=nbformat.read(HERE/'0048-dcnv2.ipynb',as_version=4)
    teacher=nbformat.read(HERE/'solutions/0048-dcnv2.ipynb',as_version=4)
    for nb in (student,teacher):
        nbformat.validate(nb)
        markdown='\n'.join(c.source for c in nb.cells if c.cell_type=='markdown')
        assert 'attachment:' not in markdown
        assert sorted(image_hash(uri) for uri in IMAGE.findall(markdown))==expected
        assert not any(c.get('attachments') for c in nb.cells)
        codes=[c for c in nb.cells if c.cell_type=='code']
        assert '@colab-bootstrap' in codes[0].source
        assert all(not any(o.output_type=='error' for o in c.outputs) for c in codes)
    codes=[c for c in student.cells if c.cell_type=='code']
    assert all(c.execution_count is None and not c.outputs for c in codes)
    todos=[c for c in codes if c.source.startswith('# TODO')]
    assert len(todos)==4 and all('____' in c.source for c in todos)
    assert all(2<=c.source.count('____')<=3 for c in todos)
    for name in ('cross_step','lowrank_step','combine_paths','mix_step'):
        assert sum(bool(re.search(r'^def '+name+r'\(',c.source,re.M)) for c in codes)==1, name+' must not be overwritten'
    canonical=ast.parse((HERE/'relkit/dcnv2.py').read_text())
    expected_nodes={n.name:ast.dump(n,include_attributes=False) for n in canonical.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    actual={}
    for cell in teacher.cells:
        if cell.cell_type=='code' and '@colab-bootstrap' not in cell.source:
            for n in ast.parse(cell.source).body:
                if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in expected_nodes:
                    actual[n.name]=ast.dump(n,include_attributes=False)
    for name in expected_nodes:assert actual[name]==expected_nodes[name],name+' differs from canonical implementation'
    assert all(c.execution_count is not None for c in teacher.cells if c.cell_type=='code')
    streams='\n'.join(o.get('text','') for c in teacher.cells for o in c.get('outputs',[]) if o.output_type=='stream')
    assert 'Student implementation on model path' in streams and '"lesson": 48' in streams
    checks=re.findall(r'PASS · ([^\n]+)',streams)
    assert len(checks)>=20
    assert 'RUN_PAPER_REPRO=False' in '\n'.join(c.source for c in codes)
    prepared=BeautifulSoup((HERE/'html/0048-dcnv2.html').read_text(),'html.parser')
    assert sorted(image_hash(im['src']) for im in prepared.select('img[src^="data:image/png;base64,"]'))==expected
    links=0
    for path in (ROOT/'lessons/0048-dcnv2.html',ROOT/'reference/dcnv2-reproduction.html',HERE/'html/0048-dcnv2.html'):
        soup=BeautifulSoup(path.read_text(),'html.parser')
        for node in soup.select('[href], [src]'):
            ref=node.get('href',node.get('src',''));u=urlsplit(ref)
            if u.scheme or u.netloc or not u.path:continue
            target=(path.parent/unquote(u.path)).resolve()
            assert target.exists(),f'{path}: missing {ref}'
            if u.fragment and target.suffix=='.html':
                linked=BeautifulSoup(target.read_text(),'html.parser')
                assert linked.find(id=unquote(u.fragment)),f'Missing fragment {ref}'
            links+=1
    entry=next(x for x in json.loads((ROOT/'lessons/manifest.json').read_text())['lessons'] if x['id']==48)
    assert entry['labPath']=='labs/0048-dcnv2.ipynb' and entry['published']
    for manifest_name in ('_sources_l048.json','_scaleup_l048_manifest.json'):
        manifest=json.loads((HERE/manifest_name).read_text())
        if 'source_sha256' in manifest:
            for file,digest in manifest['source_sha256'].items():assert hashlib.sha256((HERE/file).read_bytes()).hexdigest()==digest,file
        else:
            assert manifest['model_sha256']==hashlib.sha256((HERE/'relkit/dcnv2.py').read_bytes()).hexdigest()
            assert manifest['runner_sha256']==hashlib.sha256((HERE/'_paper_repro_l048.py').read_bytes()).hexdigest()
    print(f'PASS: {len(expected)} images in both notebooks and HTML; 4 live blank tasks; {len(checks)} executed notebook checks; canonical source parity; {links} local links; provenance.')
    report={'notebook_checks_passed':len(checks),'images':len(expected),'student_tasks':4,'local_links':links,
            'teacher_all_code_executed':True,'canonical_notebook_source_matches':True,
            'browser':'NOT_CHECKED: temporary browser download declined',
            'colab':'Inline data-URI packaging checked; live Colab UI not checked'}
    (HERE/'_package_l048_results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
