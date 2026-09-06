"""Refresh prose/figures without discarding executed solution outputs; render student HTML."""
import os
from urllib.parse import urlsplit,unquote
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from _build_l052 import build,HERE,SLUG
student=build(False,write=False);nbformat.write(student,HERE/f'{SLUG}.ipynb')
path=HERE/'solutions'/f'{SLUG}.ipynb';old=nbformat.read(path,as_version=4);new=build(True,write=False)
a=[c for c in old.cells if c.cell_type=='code'];b=[c for c in new.cells if c.cell_type=='code']
assert len(a)==len(b)
for prev,fresh in zip(a,b):
    assert prev.source==fresh.source,'Code changed: execute again before preserving outputs'
    assert prev.execution_count is not None,'Unexecuted solution cell'
    assert not any(o.output_type=='error' for o in prev.outputs)
    fresh.outputs=prev.outputs;fresh.execution_count=prev.execution_count
nbformat.write(new,path)
html,_=HTMLExporter(template_name='lab').from_notebook_node(student)
soup=BeautifulSoup(html,'html.parser')
for node in soup.find_all(id=True):node['id']=unquote(node['id'])
for node in soup.find_all('a',href=True):
    parts=urlsplit(node['href'])
    if parts.scheme or parts.netloc or not parts.path:continue
    node['href']=os.path.relpath((HERE/parts.path).resolve(),HERE/'html')+('#'+parts.fragment if parts.fragment else '')
(HERE/'html'/f'{SLUG}.html').write_text(str(soup));print('Prepared student HTML and refreshed executed solution')
