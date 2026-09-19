"""Keep the shared walkthrough portable when a lesson's existing builder runs.

Only explanatory cells/assets change. Saved execution is retained only when the
entire ordered code-cell source is identical; it is never labeled a fresh run.
"""
from pathlib import Path
import base64
import copy
import hashlib
import re
import nbformat
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parents[1]
_SAVED={}

def snapshot(number):
    slug=next((ROOT/'lessons/content').glob(f'{number:04}-*.md')).stem
    for folder in ['labs','labs/solutions']:
        p=ROOT/folder/f'{slug}.ipynb'
        if p.exists():_SAVED[str(p)]=nbformat.read(p,as_version=4)

def portable(text):
    def image(m):
        key=m.group(1)
        data=base64.b64encode((ROOT/'assets/architectures'/f'{key}.png').read_bytes()).decode()
        return 'src="data:image/png;base64,'+data+'"'
    text=re.sub(r'src="(?:\.\./)+assets/architectures/([^"/]+)\.svg"',image,text)
    # Bare lesson links are correct in lessons/, but not in a downloaded notebook.
    text=re.sub(r'\]\((\d{4}-[^)]+\.html(?:#[^)]*)?)\)',r'](https://avistian.github.io/relational/lessons/\1)',text)
    text=re.sub(r'href="(\d{4}-[^"]+\.html)"',r'href="https://avistian.github.io/relational/lessons/\1"',text)
    text=text.replace('href="../reference/0071-0090-model-map.html"','href="https://avistian.github.io/relational/reference/0071-0090-model-map.html"')
    return text

def finalize(number):
    slug=next((ROOT/'lessons/content').glob(f'{number:04}-*.md')).stem
    page=ROOT/'lessons'/f'{slug}.html'
    html=page.read_text()
    css='<link rel="stylesheet" href="../assets/learning-walkthrough.css">'
    if css not in html:html=html.replace('</head>',css+'</head>')
    page.write_text(html)
    manuscript=(ROOT/'lessons/content'/f'{slug}.md').read_text()
    block=manuscript.split('<!-- depth-walkthrough:start -->',1)[1].split('<!-- depth-walkthrough:end -->',1)[0]
    for folder in ['labs','labs/solutions']:
        p=ROOT/folder/f'{slug}.ipynb';nb=nbformat.read(p,as_version=4)
        if not any('THE BIG PICTURE' in c.source for c in nb.cells if c.cell_type=='markdown'):
            nb.cells.insert(1,nbformat.v4.new_markdown_cell(portable('<!-- depth-walkthrough:start -->'+block+'<!-- depth-walkthrough:end -->')))
        for cell in nb.cells:
            if cell.cell_type=='markdown':cell.source=portable(cell.source)
        previous=_SAVED.get(str(p))
        if previous:
            old=[c for c in previous.cells if c.cell_type=='code'];new=[c for c in nb.cells if c.cell_type=='code']
            if [c.source for c in old]!=[c.source for c in new]:
                raise RuntimeError(f'{slug}: code changed during an editorial rebuild; inspect before preserving outputs')
            for a,b in zip(old,new):
                b.outputs=copy.deepcopy(a.outputs);b.execution_count=a.execution_count;b.metadata=copy.deepcopy(a.metadata)
            # Retain runtime/provenance records from prior execution.
            nb.metadata.update(copy.deepcopy(previous.metadata))
        # Stable, unique IDs independent of prose cell insertion.
        seen={}
        for cell in nb.cells:
            digest=hashlib.sha256((cell.cell_type+'\0'+cell.source).encode()).hexdigest()[:16]
            count=seen.get(digest,0);seen[digest]=count+1
            cell.id=f'l{number:03}-{digest}-{count}'
        nbformat.validate(nb);nbformat.write(nb,p)
    # A readable student preview: outputs, if present, are preserved author records.
    nb=nbformat.read(ROOT/'labs'/f'{slug}.ipynb',as_version=4)
    html,_=HTMLExporter().from_notebook_node(nb)
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(html,'html.parser')
    for anchor in soup.find_all('a',href=True):
        href=anchor['href']
        if not href.startswith(('http:','https:','#','data:','mailto:')):anchor['href']='../'+href
    link=soup.new_tag('link',rel='stylesheet',href='../../assets/learning-walkthrough.css');soup.head.append(link)
    (ROOT/'labs/html'/f'{slug}.html').write_text('\n'.join(line.rstrip() for line in str(soup).splitlines())+'\n')

    record=ROOT/'labs'/f'_execution_l{number:03}_results.json'
    if record.exists():
        import json
        saved=json.loads(record.read_text())
        if saved.get('prose_refresh_code_unchanged'):
            saved['notebook_sha256']=hashlib.sha256((ROOT/'labs/solutions'/f'{slug}.ipynb').read_bytes()).hexdigest()
            record.write_text(json.dumps(saved,indent=2)+'\n')
