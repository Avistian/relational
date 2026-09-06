"""Refresh notebook markdown while preserving executed teacher outputs; render student HTML."""
from pathlib import Path
import os
from urllib.parse import urlsplit
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from _build_l050 import build,SLUG,HERE
student=build(False);nbformat.write(student,HERE/f'{SLUG}.ipynb')
path=HERE/'solutions'/f'{SLUG}.ipynb'
old=nbformat.read(path,as_version=4);new=build(True)
a=[c for c in old.cells if c.cell_type=='code'];b=[c for c in new.cells if c.cell_type=='code']
assert len(a)==len(b)
for prev,fresh in zip(a,b):
    assert prev.source==fresh.source,'Code changed: execute solution again before preserving outputs.'
    fresh.outputs=prev.outputs;fresh.execution_count=prev.execution_count
nbformat.write(new,path)
html,_=HTMLExporter(template_name='lab').from_notebook_node(student)
soup=BeautifulSoup(html,'html.parser')
for node in soup.find_all('a',href=True):
    href=node['href'];parts=urlsplit(href)
    if parts.scheme or parts.netloc or not parts.path:continue
    node['href']=os.path.relpath((HERE/parts.path).resolve(),HERE/'html')+('?' + parts.query if parts.query else '')+('#'+parts.fragment if parts.fragment else '')
(HERE/'html'/f'{SLUG}.html').write_text(str(soup))
print('Refreshed notebook prose/images without changing executed code; rendered student HTML.')
