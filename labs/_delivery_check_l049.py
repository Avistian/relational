"""Validate lesson links, notebook packaging and source/results consistency."""
import base64
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote,urlsplit
import nbformat
from bs4 import BeautifulSoup
from PIL import Image

HERE=Path(__file__).parent;ROOT=HERE.parent;SLUG='0049-excelformer-trompt'

def check():
    stats={}
    for file in [ROOT/'lessons'/f'{SLUG}.html',ROOT/'reference/claim-audit.html',HERE/'html'/f'{SLUG}.html']:
        soup=BeautifulSoup(file.read_text(),'html.parser');ids=[t['id'] for t in soup.select('[id]')]
        assert len(ids)==len(set(ids)),('duplicate ids',file)
        for tag,attr in [('a','href'),('img','src'),('script','src'),('link','href')]:
            for n in soup.find_all(tag):
                url=n.get(attr,'');parts=urlsplit(url)
                if not url or parts.scheme or parts.netloc:continue
                target=(file.parent/unquote(parts.path)).resolve() if parts.path else file
                assert target.exists(),(file,url,target)
                if parts.fragment and target.suffix=='.html':
                    dest=soup if target==file else BeautifulSoup(target.read_text(),'html.parser')
                    assert dest.find(id=parts.fragment) or dest.find(id=unquote(parts.fragment)) or dest.find(attrs={'name':unquote(parts.fragment)}),(file,url,'missing anchor')
        stats[file.name+'-'+file.parent.name]='links PASS'
    student=nbformat.read(HERE/f'{SLUG}.ipynb',as_version=4)
    teacher=nbformat.read(HERE/'solutions'/f'{SLUG}.ipynb',as_version=4)
    for nb in (student,teacher):nbformat.validate(nb)
    codes=[c for c in student.cells if c.cell_type=='code']
    assert '@colab-bootstrap' in codes[0].source
    assert sum(c.source.startswith('# TODO') and '____' in c.source for c in codes)==4,'Three code exercises plus written EXIT'
    assert all(c.execution_count is None and not c.outputs for c in codes),'Student must stay blank'
    source_images={hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'figures/l049').glob('*.png')}
    embedded=[]
    for c in student.cells:
        if c.cell_type=='markdown':
            assert 'attachment:' not in c.source
            for data in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',c.source):
                raw=base64.b64decode(data,validate=True)
                assert hashlib.sha256(raw).hexdigest() in source_images,'Stale embedded figure'
                embedded.append(raw)
    assert len(embedded)==7, len(embedded)
    html=(HERE/'html'/f'{SLUG}.html').read_text()
    for raw in embedded:assert base64.b64encode(raw).decode() in html,'Prepared HTML lost image'
    for c in teacher.cells:
        if c.cell_type=='code':
            assert c.execution_count is not None, 'Teacher code not executed'
            assert not any(o.output_type=='error' for o in c.outputs),'Teacher notebook error'
    r=json.loads((HERE/'_verify_l049_results.json').read_text())
    assert hashlib.sha256((HERE/'relkit/claim_models.py').read_bytes()).hexdigest()==r['model_sha256']
    manifest=json.loads((ROOT/'lessons/manifest.json').read_text())
    entry=next(v for v in manifest['lessons'] if v['id']==49)
    assert entry['labPath']==f'labs/{SLUG}.ipynb' and entry['published']
    for p in (HERE/'figures/l049').glob('*.png'):
        with Image.open(p) as im:im.verify()
    stats.update(student_blanks=4,inline_pngs=len(embedded),teacher_code_cells=sum(c.cell_type=='code' for c in teacher.cells),
                 browser='NOT_CHECKED',live_colab='NOT_CHECKED',status='PASS')
    print(json.dumps(stats,indent=2))
    return stats

if __name__=='__main__':check()
