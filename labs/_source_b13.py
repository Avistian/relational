"""Audit paper-preserved tag and Hub inventory; download source, never weights."""
import hashlib,io,json,zipfile
from pathlib import Path
from bs4 import BeautifulSoup
from _prepare_b13 import fetch
P=Path(__file__).resolve().parent;S=P/'sources/b13'
def main():
    commit=json.loads(fetch('https://api.github.com/repos/stanford-star/plurel/commits/v1.0.0'))['sha']
    archive=S/'plurel-v1.0.0.zip'
    if not archive.exists():archive.write_bytes(fetch('https://codeload.github.com/stanford-star/plurel/zip/'+commit))
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            rel=Path(*Path(name).parts[1:])
            if not name.endswith('/') and (rel.suffix in ['.py','.md','.toml','.yaml','.yml','.json','.lock'] or rel.name.startswith('LICENSE')):
                dest=S/'plurel-paper'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
    hub=json.loads(fetch('https://huggingface.co/api/models/stanford-star/rt-plurel?blobs=true'))
    (S/'plurel-hub-inventory.json').write_text(json.dumps(hub,indent=2)+'\n')
    (S/'plurel-paper-release.json').write_text(json.dumps(dict(commit=commit,tag='v1.0.0',archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),hub_revision=hub.get('sha'),note='Author README identifies this tag as paper code. This does not alone identify all Table1 training runs.'),indent=2)+'\n')
    soup=BeautifulSoup((S/'plurel-v1.html').read_text(),'html.parser')
    # Math annotations duplicate rendered numbers; retain rendered text only.
    for node in soup.select('annotation'):node.decompose()
    table=soup.find(id='S3.T1');rows=[]
    for tr in table.select('tr'):
        cells=[x.get_text(' ',strip=True) for x in tr.select('td')]
        if cells and cells[0].startswith('rel-'):rows.append(cells)
    assert len(rows)==18
    (S/'plurel-table1.json').write_text(json.dumps(dict(source='https://arxiv.org/html/2602.04029v1#S3.T1',columns=['database','task','real_only_percent','synthetic_real_percent','reported_gain_points','synthetic_only_percent'],rows=rows,note='Published rounded cells, not measured B13 predictions; printed gains may differ from subtraction of rounded values.'),indent=2)+'\n')
    print('Paper tag',commit,'; table rows',len(rows),'; Hub files',len(hub.get('siblings',[])))
if __name__=='__main__':main()
