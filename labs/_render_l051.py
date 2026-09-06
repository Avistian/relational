"""Refresh only notebook prose/figures while preserving verified code outputs."""
from pathlib import Path
import os
from urllib.parse import urlsplit,unquote
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from _build_l051 import build,HERE,SLUG
student=build(False);nbformat.write(student,HERE/f'{SLUG}.ipynb')
path=HERE/'solutions'/f'{SLUG}.ipynb';old=nbformat.read(path,as_version=4);new=build(True)
a=[c for c in old.cells if c.cell_type=='code'];b=[c for c in new.cells if c.cell_type=='code']
assert len(a)==len(b)
for prev,fresh in zip(a,b):
    assert prev.source==fresh.source,'Code changed: re-execute before preserving outputs'
    assert prev.execution_count is not None,'Teacher solution has not executed all cells'
    fresh.outputs=prev.outputs;fresh.execution_count=prev.execution_count
if 'execution_verification' in old.metadata:new.metadata['execution_verification']=old.metadata['execution_verification']
nbformat.write(new,path)
html,_=HTMLExporter(template_name='lab').from_notebook_node(student)
soup=BeautifulSoup(html,'html.parser')
# nbconvert percent-encodes some heading IDs; use decoded DOM IDs with URL-encoded fragments.
for node in soup.find_all(id=True):node['id']=unquote(node['id'])
for node in soup.find_all('a',href=True):
    parts=urlsplit(node['href'])
    if parts.scheme or parts.netloc or not parts.path:continue
    node['href']=os.path.relpath((HERE/parts.path).resolve(),HERE/'html')+('?' + parts.query if parts.query else '')+('#'+parts.fragment if parts.fragment else '')
(HERE/'html'/f'{SLUG}.html').write_text(str(soup));print('Refreshed prose and rendered prepared HTML; executed code preserved.')
