"""Refresh prose/figures while preserving executed solution code/output pairs."""
import os
from urllib.parse import urlsplit,unquote
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from _build_l054 import build,ROOT,SLUG

student=build(False,write=False);nbformat.write(student,ROOT/f'{SLUG}.ipynb')
path=ROOT/'solutions'/f'{SLUG}.ipynb';old=nbformat.read(path,as_version=4);new=build(True,write=False)
prior=[c for c in old.cells if c.cell_type=='code'];fresh=[c for c in new.cells if c.cell_type=='code']
assert len(prior)==len(fresh)
for a,b in zip(prior,fresh):
    assert a.source==b.source,'Code changed; execute again before preserving outputs'
    assert a.execution_count is not None,'Unexecuted solution cell'
    assert not any(o.output_type=='error' for o in a.outputs)
    b.outputs=a.outputs;b.execution_count=a.execution_count
nbformat.write(new,path)
html,_=HTMLExporter(template_name='lab').from_notebook_node(student)
soup=BeautifulSoup(html,'html.parser')
for node in soup.find_all(id=True):node['id']=unquote(node['id'])
for node in soup.find_all('a',href=True):
    parts=urlsplit(node['href'])
    if parts.scheme or parts.netloc or not parts.path:continue
    node['href']=os.path.relpath((ROOT/parts.path).resolve(),ROOT/'html')+('#'+parts.fragment if parts.fragment else '')
(ROOT/'html'/f'{SLUG}.html').write_text(str(soup))
print('Rendered student HTML; refreshed figures in executed solution')
